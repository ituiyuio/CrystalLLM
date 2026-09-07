"""
F36 ISS 验证: S_t = γ·S_{t-1} + g·RMS_norm(Δ_t)

[1] 手算对齐   : S_new == sigmoid(γ_raw)·S0 + g·(Δ/rms), Δ = sum_k s_k·(W_k@S0)
[2] γ/g 可训练 : CE backward 后 iss_gain.grad / iss_gamma_raw.grad 非空非零
[3] 尺度不变性 : coef 0.4 vs 0.004 -> 同一稳态带, 都不钉 128
                 (旧世界: 0.4 两步钉死 128, 0.004 躺在 2.2, 差 60 倍)
[4] F33 仍干净 : eval/generate 不改 W_pool / update_buffer
[5] commit 流动: pool LR 放大后 W_pool 真的被改 (mult 是自由旋钮的证明)
[6] d=4096+coef0.4 (旧配置 2 步钉死): 稳态带内运行, 不撞 512/不钉 128
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import math
import torch
import spike_llm as SL

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""),
          flush=True)


def build(D, NB, TK, coef, V, seed):
    torch.manual_seed(seed)
    pool = SL.make_pool(d_model=D, d_inner=D, num_blocks=NB, top_k=TK,
                        scale_coef=coef)
    model = SL.SpikeLLM(pool, vocab_size=V).to('cuda')
    return pool, model


def markov_tokens(V, N=20000, seed=7):
    g = torch.Generator().manual_seed(seed)
    ids = [torch.randint(0, V, (1,), generator=g).item()]
    for _ in range(1, N):
        ids.append((ids[-1] * 7 + 3) % V)
    return torch.tensor(ids, dtype=torch.long, device='cuda')


def trajectory(pool, model, inp, n=40):
    with torch.no_grad():
        S = model.embed(inp[:, 0])
        norms = [S.norm(dim=-1).mean().item()]
        for t in range(1, min(n, inp.shape[1])):
            S = pool.forward_step_state(model.embed(inp[:, t]), S)
            norms.append(S.norm(dim=-1).mean().item())
    pins = sum(1 for x in norms if abs(x - 128.0) < 0.5)
    return norms, pins


# ===================== Part 1: d=256 =====================
D, NB, TK, V = 256, 32, 8, 500
print("=" * 60)
print(f"=== Part 1: d={D}, blocks={NB}, top_k={TK}, ISS (gamma=0.95, gain=0.05) ===")
print("=" * 60, flush=True)

token_ids = markov_tokens(V)
pool, model = build(D, NB, TK, 0.4, V, 1234)

# ---- [1] 手算对齐 ----
gl = []
_og = pool._compute_gates


def _rec(step, S=None):
    idx = _og(step, S)
    gl.append(idx.detach().cpu().tolist())
    return idx


pool._compute_gates = _rec
pool.cfg['lr_base'] = 0.0
S0 = torch.randn(1, D, device='cuda') * 0.01
tgt = torch.randn(1, D, device='cuda') * 0.01
S_new = pool.forward_step_state(tgt, S0)
delta = torch.zeros(D, device='cuda')
for k in gl[-1]:
    delta += pool.scale_pool[k] * (pool.W_pool[k].float() @ S0[0])
rms = delta.pow(2).mean().sqrt() + 1e-6
gam = torch.sigmoid(pool.iss_gamma_raw.detach())
g = pool.iss_gain.detach()
expected = gam * S0[0] + g * delta / rms
rel = (S_new[0] - expected).norm().item() / (expected.norm().item() + 1e-12)
check("[1] S_new = γ·S0 + g·Δ/rms 手算对齐", rel < 0.02, f"rel_err={rel:.2e}")

# ---- [2] γ/g 有梯度 + [5] commit 流动 ----
# pool_lr_mult 语义: 池子 LR 单独放大. 0.05 时 per-call 更新 ~9e-4 且每块只在
# K/N=25% 调用中活跃, 840 调用累积 0.19 < 0.5 阈值 -> 不提交 (节奏问题非 bug);
# 0.5 才能在 60 步内看到 commit. 4096 真实跑同理由需要 mult ~1000
pool.cfg['lr_base'] = 0.5
opt = torch.optim.AdamW(
    [p for n, p in model.named_parameters() if 'embed' in n or 'output' in n or 'iss_' in n],
    lr=1e-3, betas=(0.9, 0.95))
inp, _ = SL.get_batch(token_ids, 4, 16, 'cuda')
loss = model(inp)
loss.backward()
ok_g = pool.iss_gain.grad is not None and torch.isfinite(pool.iss_gain.grad).all() \
    and pool.iss_gain.grad.abs().item() > 0
ok_gam = pool.iss_gamma_raw.grad is not None and torch.isfinite(pool.iss_gamma_raw.grad).all()
check("[2] iss_gain.grad 非零有限", ok_g, f"grad={pool.iss_gain.grad.item():.3e}")
check("[2] iss_gamma_raw.grad 非空有限", ok_gam,
      f"grad={pool.iss_gamma_raw.grad.item():.3e}")

w0 = pool.W_pool.clone()
losses = []
for step in range(60):
    inp, _ = SL.get_batch(token_ids, 4, 16, 'cuda')
    loss = model(inp)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item() / 15)
check("[5] 60 步后 W_pool 有 commit", not torch.equal(w0, pool.W_pool),
      f"loss {losses[0]:.3f} -> {losses[-1]:.3f}")

# ---- [4] eval 干净 ----
w_snap = pool.W_pool.clone()
ub_snap = pool.update_buffer.clone()
_ = SL.estimate_loss(model, token_ids, 4, 16, n_batches=2)
_ = SL.generate(model, torch.randint(0, V, (3,)), max_new_tokens=8)
check("[4] eval/generate 不改 W_pool", torch.equal(w_snap, pool.W_pool))
check("[4] eval/generate 不改 update_buffer", torch.equal(ub_snap, pool.update_buffer))

del pool, model, opt
torch.cuda.empty_cache()

# ---- [3] 尺度不变性: coef 0.4 vs 0.004 同稳态带 ----
print("\n=== [3] 尺度不变性: coef 0.4 vs 0.004 (同种子同数据) ===", flush=True)
finals = {}
for coef in [0.4, 0.004]:
    pool_c, model_c = build(D, NB, TK, coef, V, 777)
    pool_c.cfg['lr_base'] = 0.05
    opt_c = torch.optim.AdamW(
        [p for n, p in model_c.named_parameters() if 'embed' in n or 'output' in n or 'iss_' in n],
        lr=1e-3, betas=(0.9, 0.95))
    for step in range(10):
        inp, _ = SL.get_batch(token_ids, 4, 16, 'cuda')
        loss = model_c(inp)
        opt_c.zero_grad()
        loss.backward()
        opt_c.step()
    torch.manual_seed(999)
    inp_probe, _ = SL.get_batch(token_ids, 1, 41, 'cuda')
    norms, pins = trajectory(pool_c, model_c, inp_probe)
    finals[coef] = norms[-1]
    print(f"  coef={coef}: final_norm={norms[-1]:.2f}, max={max(norms):.2f}, "
          f"pins={pins}, 轨迹末 10: {[f'{n:.1f}' for n in norms[-10:]]}", flush=True)
    del pool_c, model_c, opt_c
    torch.cuda.empty_cache()

gam = 0.95
g0 = 0.05
lo = g0 * math.sqrt(D) / math.sqrt(1 - gam ** 2) * 0.7   # 方向随机下界
hi = g0 * math.sqrt(D) / (1 - gam) * 1.3                 # 方向对齐上界
f1, f2 = finals[0.4], finals[0.004]
check("[3] coef=0.4 在稳态带内 (不钉 128 不撞 512)", lo <= f1 <= hi,
      f"final={f1:.2f}, band=[{lo:.1f},{hi:.1f}]")
check("[3] coef=0.004 在同一稳态带", lo <= f2 <= hi, f"final={f2:.2f}")
check("[3] 尺度不变: 两配置终值差 < 50%", abs(f1 - f2) / max(f1, f2) < 0.5,
      f"|{f1:.2f}-{f2:.2f}| (旧世界差 60x)")

# ===================== Part 2: d=4096 + coef 0.4 =====================
print("\n" + "=" * 60)
print("=== Part 2: d=4096, blocks=64, coef=0.4 (旧配置 2 步钉死 128) ===")
print("=" * 60, flush=True)
poolB, modelB = build(4096, 64, 16, 0.4, V, 42)
poolB.cfg['lr_base'] = 5e-3
optB = torch.optim.AdamW(
    [p for n, p in modelB.named_parameters() if 'embed' in n or 'output' in n or 'iss_' in n],
    lr=5e-5, betas=(0.9, 0.95))
lossesB = []
for step in range(12):
    inp, _ = SL.get_batch(token_ids, 2, 16, 'cuda')
    loss = modelB(inp)
    optB.zero_grad()
    loss.backward()
    optB.step()
    lossesB.append(loss.item() / 15)
torch.manual_seed(999)
inpB, _ = SL.get_batch(token_ids, 1, 41, 'cuda')
normsB, pinsB = trajectory(poolB, modelB, inpB)
loB = g0 * math.sqrt(4096) / math.sqrt(1 - gam ** 2) * 0.7
hiB = g0 * math.sqrt(4096) / (1 - gam) * 1.3
print(f"  12 steps: loss {lossesB[0]:.3f} -> {lossesB[-1]:.3f} | "
      f"S_norm final={normsB[-1]:.1f}, max={max(normsB):.1f}, pins={pinsB}", flush=True)
print(f"  轨迹: {[f'{n:.1f}' for n in normsB]}", flush=True)
check("[6] 4096+coef0.4 不钉 128 不撞墙", pinsB == 0 and max(normsB) < 400,
      f"final={normsB[-1]:.1f}, band=[{loB:.0f},{hiB:.0f}]")
check("[6] 4096 稳态带内", loB <= normsB[-1] <= hiB, f"final={normsB[-1]:.1f}")

del poolB, modelB, optB
torch.cuda.empty_cache()

print("\n" + "=" * 60)
print(f"RESULT: {sum(results)}/{len(results)} passed")
print("=" * 60, flush=True)
