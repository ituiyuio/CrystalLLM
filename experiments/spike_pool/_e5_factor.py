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
B, L, STEPS = 256, 64, 40

def run(tag, fixed_batch=False, cpu_pool=True, warmup=True, pool_lr=5e-4):
    torch.manual_seed(0)
    m = _sc.make_chunk_model(4096, 4096, 64, 16, 4100, cpu_offload=cpu_pool,
                             chunk_size=31, mode='shared', gate_source='token',
                             input_drive='adjunct', ce_residual=True)
    if cpu_pool:
        m.pool.attach_prefetcher()
    m = m.to('cuda')
    m.pool.cfg['lr_base'] = pool_lr
    opt = torch.optim.AdamW([p for n, p in m.named_parameters()
                             if p.requires_grad], lr=5e-4)
    ids_fixed, _ = _sl.get_batch(tok, B, L, 'cuda')
    pf = getattr(m.pool, '_prefetcher', None)
    inp_next, _ = _sl.get_batch(tok, B, L, 'cuda')
    route_next = m.preissue(inp_next) if pf else None
    l0 = None
    for s in range(STEPS):
        inp, route = inp_next, route_next
        if fixed_batch:
            inp = ids_fixed
        else:
            inp_next, _ = _sl.get_batch(tok, B, L, 'cuda')
        route_next = m.preissue(inp_next) if pf else None
        loss = m(inp, route=route)
        opt.zero_grad(); loss.backward(); opt.step()
        if s == 0: l0 = loss.item() / 63
        if pf and s == 0:
            pf.join_pending()
    lf = loss.item() / 63
    print(f"{tag:34s} loss {l0:.3f} -> {lf:.3f} "
          f"({'LEARNS' if lf < l0 - 0.05 else 'FLAT'})", flush=True)
    if pf: pf.shutdown()
    del m; torch.cuda.empty_cache()

run("L0 ladder-exact (fresh+cpu+warm)")                      # 预期 FLAT
run("L1 fixed batch", fixed_batch=True)                      # 换固定 batch
run("L2 gpu pool (no pf/lookahead)", cpu_pool=False)         # 换 GPU 池
run("L3 no warmup (const lr, fresh)", warmup=False)          # 换恒定 LR
