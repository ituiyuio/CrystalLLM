import os, sys, math, torch
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
_here = os.path.dirname(os.path.abspath(__file__))
import importlib.util
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod
_sl = _load("spike_llm", os.path.join(_here, "spike_llm.py"))
_sc = _load("spike_llm_chunk", os.path.join(_here, "spike_llm_chunk.py"))
BPE = "D:/CrystaLLM/experiments/v49_pre/bpe_train_2000_s42.npy"
tok = _sl.get_bpe_data(BPE)
B, L, STEPS, WARM = 256, 64, 60, 50

def sched_f(step):
    if step < WARM: return step / WARM
    prog = (step - WARM) / max(1, STEPS - WARM)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

def run(tag, sync_pool_lr):
    torch.manual_seed(0)
    m = _sc.make_chunk_model(4096, 4096, 64, 16, 4100, cpu_offload=True,
                             chunk_size=31, mode='shared', gate_source='token',
                             input_drive='adjunct', ce_residual=True)
    m.pool.attach_prefetcher()
    m = m.to('cuda')
    pool_base = 5e-4
    m.pool.cfg['lr_base'] = pool_base
    opt = torch.optim.AdamW([p for n, p in m.named_parameters()
                             if p.requires_grad], lr=5e-4)
    pf = m.pool._prefetcher
    inp_next, _ = _sl.get_batch(tok, B, L, 'cuda')
    route_next = m.preissue(inp_next)
    l0 = None
    for s in range(STEPS):
        inp, route = inp_next, route_next
        inp_next, _ = _sl.get_batch(tok, B, L, 'cuda')
        route_next = m.preissue(inp_next)
        f = sched_f(s)
        for g in opt.param_groups:
            g['lr'] = 5e-4 * f
        if sync_pool_lr:
            m.pool.cfg['lr_base'] = pool_base * f   # 池子同步 warmup
        loss = m(inp, route=route)
        opt.zero_grad(); loss.backward(); opt.step()
        if s == 0: l0 = loss.item() / 63
        if s == 0: pf.join_pending()
    lf = loss.item() / 63
    print(f"{tag:36s} loss {l0:.3f} -> {lf:.3f} "
          f"({'LEARNS' if lf < l0 - 0.05 else 'FLAT'})", flush=True)
    pf.shutdown(); del m; torch.cuda.empty_cache()

run("P1 head-warmup only (复现死平?)", sync_pool_lr=False)
run("P2 pool 同步 warmup (机制验证)", sync_pool_lr=True)
