import os, sys, torch
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import importlib.util
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod
_sc = _load("spike_llm_chunk", os.path.join(_here, "spike_llm_chunk.py"))

def try_cfg(tag, **kw):
    torch.manual_seed(0)
    try:
        m = _sc.make_chunk_model(256, 256, 16, 4, 128, cpu_offload=False,
                                 chunk_size=8, mode='shared',
                                 gate_source='token', **kw).to('cuda')
        ids = torch.randint(0, 128, (8, 24), device='cuda')
        loss = m(ids)
        loss.backward()
        print(f"{tag}: OK (loss={loss.item():.3f})", flush=True)
        del m; torch.cuda.empty_cache()
        return True
    except RuntimeError as ex:
        print(f"{tag}: CRASH — {str(ex)[:80]}", flush=True)
        torch.cuda.empty_cache()
        return False

try_cfg("A baseline(no drive)", input_drive=False)
try_cfg("B adjunct only", input_drive='adjunct')
try_cfg("C sum only", input_drive=True)
try_cfg("D adjunct+ce_res", input_drive='adjunct', ce_residual=True)
try_cfg("E ce_res only", ce_residual=True)

# --- 阶段2: CPU 池 + 预取 + lookahead + 多步 ---
def try_cfg2(tag, steps=3, **kw):
    torch.manual_seed(0)
    try:
        m = _sc.make_chunk_model(256, 256, 16, 4, 128, cpu_offload=True,
                                 chunk_size=8, mode='shared',
                                 gate_source='token', **kw)
        m.pool.attach_prefetcher()
        m = m.to('cuda')
        opt = torch.optim.AdamW([p for n, p in m.named_parameters()
                                 if p.requires_grad], lr=1e-4)
        ids = torch.randint(0, 128, (8, 24), device='cuda')
        inp_next = ids.clone()
        route_next = m.preissue(inp_next)
        for s in range(steps):
            route = route_next
            inp_next = torch.randint(0, 128, (8, 24), device='cuda')
            route_next = m.preissue(inp_next)
            loss = m(ids, route=route)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"{tag}: OK x{steps} (loss={loss.item():.3f})", flush=True)
        m.pool._prefetcher.shutdown(); del m; torch.cuda.empty_cache()
        return True
    except RuntimeError as ex:
        print(f"{tag}: CRASH — {str(ex)[:80]}", flush=True)
        torch.cuda.empty_cache()
        return False

try_cfg2("F cpu+pf+lookahead baseline", input_drive=False)
try_cfg2("G cpu+pf+lookahead adjunct", input_drive='adjunct')
try_cfg2("H cpu+pf+lookahead adjunct+ce", input_drive='adjunct', ce_residual=True)

# --- 阶段3: d=4096 全尺寸, 逐步变参 ---
def try_cfg3(tag, B=256, L=64, steps=1, d=4096, **kw):
    torch.manual_seed(0)
    try:
        m = _sc.make_chunk_model(d, d, 16, 4, 1000, cpu_offload=False,
                                 chunk_size=31, mode='shared',
                                 gate_source='token', **kw).to('cuda')
        opt = torch.optim.AdamW([p for n, p in m.named_parameters()
                                 if p.requires_grad], lr=1e-4)
        for s in range(steps):
            ids = torch.randint(0, 1000, (B, L), device='cuda')
            loss = m(ids)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"{tag}: OK x{steps}", flush=True)
        del m; torch.cuda.empty_cache()
        return True
    except RuntimeError as ex:
        print(f"{tag}: CRASH — {str(ex)[:70]}", flush=True)
        torch.cuda.empty_cache()
        return False

try_cfg3("I d4096 B256 L64 adjunct", input_drive='adjunct')
try_cfg3("J d4096 B64 L64 adjunct", B=64, input_drive='adjunct')
try_cfg3("K d4096 B256 L32 adjunct", L=32, input_drive='adjunct')
try_cfg3("L d4096 B256 L64 ce_res", ce_residual=True)

# --- 阶段4: 全尺寸 + CPU 池 + 预取 + lookahead (ladder E5 完整组合) ---
def try_cfg4(tag, B=256, L=64, steps=2, **kw):
    torch.manual_seed(0)
    try:
        m = _sc.make_chunk_model(4096, 4096, 64, 16, 1000, cpu_offload=True,
                                 chunk_size=31, mode='shared',
                                 gate_source='token', **kw)
        m.pool.attach_prefetcher()
        m = m.to('cuda')
        opt = torch.optim.AdamW([p for n, p in m.named_parameters()
                                 if p.requires_grad], lr=1e-4)
        inp_next = torch.randint(0, 1000, (B, L), device='cuda')
        route_next = m.preissue(inp_next)
        for s in range(steps):
            route = route_next
            inp_next = torch.randint(0, 1000, (B, L), device='cuda')
            route_next = m.preissue(inp_next)
            loss = m(inp_next if False else route and inp_next or inp_next) if False else m(inp_next if False else route['idx_gpu'] and inp_next or inp_next)
            loss = m(inp_next, route=None)  # 简化: 只测 backward 链
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"{tag}: OK x{steps}", flush=True)
        m.pool._prefetcher.shutdown(); del m; torch.cuda.empty_cache()
        return True
    except RuntimeError as ex:
        print(f"{tag}: CRASH — {str(ex)[:70]}", flush=True)
        torch.cuda.empty_cache()
        return False

try_cfg4("M full+cpu+pf adjunct", input_drive='adjunct')
try_cfg4("N full+cpu+pf adjunct+ce", input_drive='adjunct', ce_residual=True)

def try_cfg5(tag, B=256, L=64, steps=2, **kw):
    torch.manual_seed(0)
    try:
        m = _sc.make_chunk_model(4096, 4096, 64, 16, 1000, cpu_offload=True,
                                 chunk_size=31, mode='shared',
                                 gate_source='token', **kw)
        m.pool.attach_prefetcher()
        m = m.to('cuda')
        opt = torch.optim.AdamW([p for n, p in m.named_parameters()
                                 if p.requires_grad], lr=1e-4)
        inp_next = torch.randint(0, 1000, (B, L), device='cuda')
        route_next = m.preissue(inp_next)
        for s in range(steps):
            route = route_next
            inp_next = torch.randint(0, 1000, (B, L), device='cuda')
            route_next = m.preissue(inp_next)
            loss = m(inp_next, route=route)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"{tag}: OK x{steps}", flush=True)
        m.pool._prefetcher.shutdown(); del m; torch.cuda.empty_cache()
        return True
    except RuntimeError as ex:
        print(f"{tag}: CRASH — {str(ex)[:70]}", flush=True)
        torch.cuda.empty_cache()
        return False

try_cfg5("O ladder-exact adjunct+ce", input_drive='adjunct', ce_residual=True)
