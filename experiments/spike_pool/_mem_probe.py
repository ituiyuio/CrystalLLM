import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import importlib.util
import torch.nn.functional as F
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod
_sc = _load("spike_llm_chunk", os.path.join(os.path.dirname(__file__), "spike_llm_chunk.py"))
torch.cuda.set_per_process_memory_fraction(0.95, 0)
model = _sc.make_chunk_model(4096, 4096, 64, 16, 4100, cpu_offload=True,
                             chunk_size=31, mode='shared', gate_source='token',
                             input_drive='adjunct', bptt_pool=True)
model.pool.attach_prefetcher()
model = model.to('cuda')
torch.cuda.empty_cache()
print(f"static alloc = {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
B, L = 256, 64
ids = torch.randint(0, 4100, (B, L), device='cuda')
pool = model.pool
route = model.preissue(ids)
E = model.embed(ids)
S = E[:, 0]
blocks_gpu = route['idx_gpu'][0]
stage_i8, scale_gpu = pool._prefetcher.get(0, route['evs'][0])
with torch.no_grad():
    pool._chunk_bf16[0].copy_(stage_i8)
W_bf16 = pool._chunk_bf16[0]
scale_k = scale_gpu
K = blocks_gpu.shape[0]
S_old = S
for t in range(31):
    target_t = E[:, 1+t, :]
    S_for_bmm = S_old.to(torch.bfloat16).unsqueeze(0).expand(K,-1,-1,-1).permute(0,2,1,3).reshape(K, pool.d_model, B)
    a0 = torch.cuda.memory_allocated()/1e9
    W_leaf = W_bf16.clone().requires_grad_(True)
    deltas_bf16 = torch.bmm(W_leaf, S_for_bmm)
    delta_S = (deltas_bf16 * scale_k.bfloat16().view(-1,1,1)).sum(dim=0).T.float()
    a1 = torch.cuda.memory_allocated()/1e9
    inj = target_t.float()
    rms_inj = inj.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6
    rms_w = delta_S.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6
    S_new = torch.sigmoid(pool.iss_gamma_raw)*S_old + pool.iss_gain*(inj/rms_inj) + pool.iss_gain_w*(delta_S/rms_w)
    a2 = torch.cuda.memory_allocated()/1e9
    if t % 5 == 0 or t == 30:
        print(f"  pos {t:2d}: pre={a0:.2f} bmm+{a1-a0:.3f} state+{a2-a1:.3f} total={a2:.2f}", flush=True)
    S_old = S_new
print(f"end chunk0 loop: {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
