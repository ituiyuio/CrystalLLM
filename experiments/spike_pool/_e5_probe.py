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
                         if p.requires_grad], lr=5e-4)
ids = torch.randint(0, 1000, (256, 64), device='cuda')
for s in range(31):
    loss = m(ids)
    opt.zero_grad(); loss.backward(); opt.step()
    if s % 5 == 0:
        with torch.no_grad():
            gamma = torch.sigmoid(m.iss_gamma_raw)
            E = m.embed(ids)
            S = E[:, 0]
            for t in range(1, 63):
                inj = E[:, t]
                inj = inj / (inj.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)
                S = gamma * S + m.iss_gain * inj
            print(f"s={s:3d} loss={loss.item()/63:.3f} gamma={float(gamma):.3f} "
                  f"g={float(m.iss_gain):.4f} g_w={float(m.iss_gain_w):.4f} "
                  f"|S(e1-form)|={S.norm()/256**0.5:.2f}", flush=True)
