import os, sys, torch
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
_here = os.path.dirname(os.path.abspath(__file__))
import importlib.util
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod
_sc = _load("spike_llm_chunk", os.path.join(_here, "spike_llm_chunk.py"))
torch.manual_seed(0)
m = _sc.make_chunk_model(4096, 4096, 64, 16, 1000, cpu_offload=False,
                         chunk_size=31, mode='shared', gate_source='token',
                         input_drive='adjunct', ce_residual=True).to('cuda')
opt = torch.optim.AdamW([p for n, p in m.named_parameters()
                         if p.requires_grad], lr=1e-4)
for s in range(2):
    ids = torch.randint(0, 1000, (256, 64), device='cuda')
    loss = m(ids)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"step {s} OK loss={loss.item():.2f} "
          f"alloc={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
print("PASS")
