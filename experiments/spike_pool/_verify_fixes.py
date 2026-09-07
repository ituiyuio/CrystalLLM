"""
F30-F35 修复验证脚本 (机制级断言, 不看 loss 收敛 -- 那是重跑 4096 的事)

F30 lock bug   : >190 次调用后门控仍是 Bernoulli (多块轮转, 不是永远只有块 0)
F31 scale 双乘 : 手算 delta = sum_k scale_k * (W_k @ S) 与 S_new - S_old 对齐 (单乘)
F32 内容寻址   : _compute_gates 收到外部 S (非 self.S zeros)
F33 eval 干净  : estimate_loss / generate 不改 W_pool / update_buffer
F34 topk 平局  : 训练中 unique 激活块数 > Top_K (休眠块随机轮换)
F35 LR 接线    : cfg['lr_base'] 外部可控, update_buffer 响应; 大 lr 下 commit 改 W_pool
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import spike_llm as SL

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""),
          flush=True)


# ===================== Part A: d=256 机制验证 =====================
D, NB, TK, V = 256, 32, 8, 500
print("=" * 60)
print(f"=== Part A: d={D}, blocks={NB}, top_k={TK} ===")
print("=" * 60, flush=True)

pool = SL.make_pool(d_model=D, d_inner=D, num_blocks=NB, top_k=TK)
model = SL.SpikeLLM(pool, vocab_size=V).to('cuda')
pool.cfg['lr_base'] = 1e-3   # F35: 外部控制池子 LR

gate_log = []
_orig_gates = pool._compute_gates


def _rec_gates(step, S=None):
    idx = _orig_gates(step, S)
    gate_log.append((S is not None, idx.detach().cpu().tolist()))
    return idx


pool._compute_gates = _rec_gates

# 可预测 Markov 链: next = (cur*7+3) % V, 看池子参与后 loss 是否更低 (信息性, 不断言)
ids = [torch.randint(0, V, (1,)).item()]
for _ in range(1, 20000):
    ids.append((ids[-1] * 7 + 3) % V)
token_ids = torch.tensor(ids, dtype=torch.long, device='cuda')

opt = torch.optim.AdamW(
    [p for n, p in model.named_parameters() if 'embed' in n or 'output' in n],
    lr=1e-3, betas=(0.9, 0.95))
ub_before = pool.update_buffer.clone()
losses = []
for step in range(60):
    inp, _ = SL.get_batch(token_ids, 4, 16, 'cuda')
    loss = model(inp)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item() / 15)

n_calls = len(gate_log)
uniq = len({b for _, idxs in gate_log for b in idxs})
after190 = [idxs for i, (_, idxs) in enumerate(gate_log) if i >= 190]
uniq_after190 = len({b for idxs in after190 for b in idxs})
k_mean = sum(len(idxs) for _, idxs in gate_log) / max(1, n_calls)
uniform = float(torch.log(torch.tensor(float(V))))
print(f"  calls={n_calls} | K_mean={k_mean:.1f} | unique_blocks={uniq}/{NB} | "
      f"unique_after_call190={uniq_after190} | loss {losses[0]:.3f} -> {losses[-1]:.3f} "
      f"(uniform={uniform:.3f})", flush=True)

check("F30 lock 关闭: >190 调用后仍多块轮转", uniq_after190 > 1, f"unique={uniq_after190}")
check("F34 topk 平局: unique 总数 > top_k (休眠块轮换)", uniq > TK, f"{uniq} > {TK}")
check("F32 外部 S 传进 _compute_gates", all(hs for hs, _ in gate_log))
check("训练时 update_buffer 有累积 (更新通路活)", not torch.equal(ub_before, pool.update_buffer))

# F33: eval / generation 不改权重
w_snap = pool.W_pool.clone()
ub_snap = pool.update_buffer.clone()
_ = SL.estimate_loss(model, token_ids, 4, 16, n_batches=2)
_ = SL.generate(model, torch.randint(0, V, (3,)), max_new_tokens=8)
check("F33 eval/generate 不改 W_pool", torch.equal(w_snap, pool.W_pool))
check("F33 eval/generate 不改 update_buffer", torch.equal(ub_snap, pool.update_buffer))

# 对照: grad-enabled 调用确实会累积
_ = model(SL.get_batch(token_ids, 4, 16, 'cuda')[0])
check("对照: 训练前向会累积 update_buffer", not torch.equal(ub_snap, pool.update_buffer))

del pool, model, opt
torch.cuda.empty_cache()

# ===================== F31: 手算前向对齐 (新池子, lr=0 只看前向) =====================
print("\n=== F31: 单乘 scale 手算验证 ===", flush=True)
pool2 = SL.make_pool(d_model=D, d_inner=D, num_blocks=NB, top_k=TK)
pool2.cfg['lr_base'] = 0.0
_gl = []
_og = pool2._compute_gates


def _rec2(step, S=None):
    idx = _og(step, S)
    _gl.append(idx.detach().cpu().tolist())
    return idx


pool2._compute_gates = _rec2
S0 = torch.randn(1, D, device='cuda') * 0.01
tgt = torch.randn(1, D, device='cuda') * 0.01
S_new = pool2.forward_step_state(tgt, S0)
idx = _gl[-1]
delta = torch.zeros(D, device='cuda')
for k in idx:
    delta += pool2.scale_pool[k] * (pool2.W_pool[k].float() @ S0[0])
expected = S0[0] + delta
rel_err = (S_new[0] - expected).norm().item() / (expected.norm().item() + 1e-12)
check("F31 delta = scale*(W@S) 单乘 (双乘会差 scale 倍)", rel_err < 0.02, f"rel_err={rel_err:.2e}")

# commit 通路: 大 lr 下 W_pool 真的被改
pool2.cfg['lr_base'] = 500.0
w2 = pool2.W_pool.clone()
_ = pool2.forward_step_state(tgt, S0)
check("commit 路径: 大 lr 下 W_pool 被改", not torch.equal(w2, pool2.W_pool))
del pool2
torch.cuda.empty_cache()

# ===================== Part B: d=4096 短程 (默认 scale, 看 S 动力学) =====================
print("\n" + "=" * 60)
print("=== Part B: d=4096, blocks=64, top_k=16, 8 步 + S_norm 轨迹 ===")
print("=" * 60, flush=True)
poolB = SL.make_pool(d_model=4096, d_inner=4096, num_blocks=64, top_k=16)
modelB = SL.SpikeLLM(poolB, vocab_size=V).to('cuda')
poolB.cfg['lr_base'] = 5e-5
_glB = []
_ogB = poolB._compute_gates


def _recB(step, S=None):
    idx = _ogB(step, S)
    _glB.append(idx.detach().cpu().tolist())
    return idx


poolB._compute_gates = _recB
optB = torch.optim.AdamW(
    [p for n, p in modelB.named_parameters() if 'embed' in n or 'output' in n],
    lr=5e-5, betas=(0.9, 0.95))
lossesB = []
for step in range(8):
    inp, _ = SL.get_batch(token_ids, 2, 32, 'cuda')
    loss = modelB(inp)
    optB.zero_grad()
    loss.backward()
    optB.step()
    lossesB.append(loss.item() / 31)
uniqB = len({b for idxs in _glB for b in idxs})
print(f"  8 steps done | unique_blocks={uniqB}/64 | loss {lossesB[0]:.3f} -> {lossesB[-1]:.3f} "
      f"(uniform={uniform:.3f})", flush=True)

# S_norm 轨迹 (no_grad -> F33 保证不更新, 纯观察池子前向动力学)
inpB, _ = SL.get_batch(token_ids, 1, 24, 'cuda')
with torch.no_grad():
    S = modelB.embed(inpB[:, 0])
    norms = [S.norm(dim=-1).mean().item()]
    for t in range(1, 23):
        S = poolB.forward_step_state(modelB.embed(inpB[:, t]), S)
        norms.append(S.norm(dim=-1).mean().item())
print(f"  S_norm 轨迹 (cap={poolB.cfg['S_Norm_Cap']}, target=128): "
      f"{[f'{n:.1f}' for n in norms]}", flush=True)

del poolB, modelB, optB
torch.cuda.empty_cache()

print("\n" + "=" * 60)
print(f"RESULT: {sum(results)}/{len(results)} passed")
print("=" * 60, flush=True)
