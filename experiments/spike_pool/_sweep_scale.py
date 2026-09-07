"""
scale_coef 扫参 (F30-F35 修复后的第一个真实调参实验)

背景: 修复后 4096 池子活了但过热 -- S_norm 每步撞 512 截断被钉回 128.
理论: 有效线性递推 S_t = (I + M) S_{t-1}, M = sum_k scale_k * W_k (INT8 随机块),
      lambda_max(M) ≈ 2*sqrt(d_model) * scale * std_INT8(74) * sqrt(K)
                  ≈ 189 * scale_coef   (d=4096, K=16, scale = 0.005*coef)
      稳定边界 coef ≈ 1/189 ≈ 0.005. 扫 [0.04, 0.01, 0.004, 0.001] 覆盖
      热区/边界/稳定区/冷区, 用 S_norm 轨迹探针量化.

每个 config:
  1. 固定种子建池 (64 blocks, K=16 动力学与 256 相同) + 短程训练 30 步
     (让 sleepers 通过 commit 拿到真实权重, 不是纯初始随机探针)
  2. no_grad 前向轨迹 24 步 (F33 保证不更新), 记录:
     - S_norm 轨迹 (是否钉死 128 = 每步截断)
     - delta/S 比率 (手算 sum_k scale_k W_k @ S, 池子每步注入多少)
  3. 分类: PINNED(每步撞墙) / STABLE(有贡献不撞墙) / COLD(贡献可忽略)
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import spike_llm as SL

D, NB, TK = 4096, 64, 16
SEED = 1234


def s_norm_trajectory(pool, model, inp, n_steps=24):
    """S_norm 监控探针: no_grad 前向链, 返回 (norms, delta_ratios, pinned)

    delta_ratio 用手算 sum_k scale_k*(W_k @ S_prev) 拿到截断前的真实注入量;
    norms 是 forward_step_state 返回的截断后值 (128.0 连续出现 = 每步撞墙)
    """
    gate_log = []
    _og = pool._compute_gates

    def _rec(step, S=None):
        idx = _og(step, S)
        gate_log.append(idx)
        return idx

    pool._compute_gates = _rec
    with torch.no_grad():
        S = model.embed(inp[:, 0])
        norms = [S.norm(dim=-1).mean().item()]
        ratios = []
        for t in range(1, min(n_steps, inp.shape[1])):
            S_prev = S
            S = pool.forward_step_state(model.embed(inp[:, t]), S)
            norms.append(S.norm(dim=-1).mean().item())
            # 截断前 delta (用当步实际激活块)
            delta = torch.zeros_like(S_prev)
            for k in gate_log[-1].detach().cpu().tolist():
                delta += pool.scale_pool[k] * (pool.W_pool[k].float() @ S_prev[0])
            ratios.append((delta.norm().item() / (S_prev.norm().item() + 1e-12)))
    pool._compute_gates = _og
    pinned = sum(1 for n in norms if abs(n - 128.0) < 0.5) >= 3
    return norms, ratios, pinned


def run_config(coef, token_ids, inp_probe):
    torch.manual_seed(SEED)
    pool = SL.make_pool(d_model=D, d_inner=D, num_blocks=NB, top_k=TK,
                        scale_coef=coef)
    model = SL.SpikeLLM(pool, vocab_size=4100).to('cuda')
    pool.cfg['lr_base'] = 5e-5
    opt = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if 'embed' in n or 'output' in n],
        lr=5e-5, betas=(0.9, 0.95))

    losses = []
    for step in range(30):
        inp, _ = SL.get_batch(token_ids, 2, 16, 'cuda')
        loss = model(inp)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item() / 15)

    norms, ratios, pinned = s_norm_trajectory(pool, model, inp_probe)
    mean_ratio = sum(ratios) / max(1, len(ratios))
    final_norm = norms[-1]
    max_norm = max(norms)

    if pinned or max_norm > 400:
        cls = "PINNED(撞墙)"
    elif final_norm > 1.5 and mean_ratio > 2e-3:
        cls = "STABLE"
    else:
        cls = "COLD(死水)"

    lam = 189.0 * coef  # 理论谱半径
    print(f"\n--- scale_coef={coef} (scale={0.01 * coef * 0.5:.2e}, 理论lambda≈{lam:.2f}) ---",
          flush=True)
    print(f"  loss: {losses[0]:.3f} -> {losses[-1]:.3f} | "
          f"S_norm: {norms[0]:.1f} -> max {max_norm:.1f} -> final {final_norm:.1f} | "
          f"delta/S = {mean_ratio:.3f} | {cls}", flush=True)
    print(f"  轨迹: {[f'{n:.1f}' for n in norms]}", flush=True)

    del pool, model, opt
    torch.cuda.empty_cache()
    return {'coef': coef, 'cls': cls, 'final_norm': final_norm,
            'max_norm': max_norm, 'ratio': mean_ratio,
            'loss_first': losses[0], 'loss_last': losses[-1]}


if __name__ == "__main__":
    token_ids = SL.get_bpe_data(
        "D:/CrystaLLM/experiments/v49_pre/bpe_train_2000_s42.npy")
    torch.manual_seed(SEED)
    inp_probe, _ = SL.get_batch(token_ids, 1, 25, 'cuda')  # 固定探针序列

    rows = []
    for coef in [0.04, 0.01, 0.004, 0.001]:
        rows.append(run_config(coef, token_ids, inp_probe))

    print("\n" + "=" * 70)
    print(f"{'coef':>6} | {'lambda':>7} | {'max_norm':>8} | {'final':>6} | "
          f"{'delta/S':>7} | {'loss_end':>8} | 分类")
    print("-" * 70)
    for r in rows:
        print(f"{r['coef']:>6} | {189 * r['coef']:>7.2f} | {r['max_norm']:>8.1f} | "
              f"{r['final_norm']:>6.1f} | {r['ratio']:>7.3f} | {r['loss_last']:>8.3f} | {r['cls']}")
    print("=" * 70, flush=True)
