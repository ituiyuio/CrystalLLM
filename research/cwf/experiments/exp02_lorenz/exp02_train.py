"""
Day 3: Training CWF vs AR+VQ on Lorenz (5000 steps)
====================================================

训练协议:
- CWF: lr=1e-3, AdamW, MSE on (x, y, z)
- AR+VQ: lr=1e-3, AdamW, MSE + CE + VQ loss
- Oracle: 不训练 (物理上界)
- 5000 步, batch=32, seq_len=256
- 每 200 步评估 teacher-forcing 1-step MSE
- 最后 200 步评估 AR ceiling (确认 plateau)

VQ 防塌缩:
- 监控 codebook perplexity, 若 < 50 触发 code reset
- commitment_cost=0.25 (标准)
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from lorenz_data import generate_lorenz_trajectories
from lorenz_oracle import LorenzOracle
from cwf_lorenz import MultiChannelCWFLorenz
from ar_vq_lorenz import ARVQBaselineLorenz


def train_cwf(model, train_data, n_steps=5000, batch_size=32, seq_len=256,
              lr=1e-3, log_every=200, device="cuda"):
    """训练 CWF, 监控 loss + closure."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)
    loss_fn = nn.MSELoss()

    train_data = train_data.to(device)
    N = train_data.shape[0]
    history = {"loss": [], "closure_max": [], "step": []}

    print(f"\n{'='*70}\nTraining CWF ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params)\n{'='*70}")

    t0 = time.time()
    window_loss = []
    for step in range(1, n_steps + 1):
        # 随机采样 batch
        idx = torch.randint(0, N, (batch_size,))
        starts = torch.randint(0, train_data.shape[1] - seq_len - 1, (batch_size,))

        x_batch = torch.stack([train_data[i, s:s+seq_len] for i, s in zip(idx, starts)])  # (B, T, 3)
        y_target = torch.stack([train_data[i, s+seq_len] for i, s in zip(idx, starts)])  # (B, 3)

        model.train()
        y_hat, info = model(x_batch, return_info=True)
        loss = loss_fn(y_hat, y_target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        window_loss.append(loss.item())
        if step % log_every == 0 or step == 1 or step == n_steps:
            avg = sum(window_loss[-log_every:]) / min(log_every, len(window_loss))
            history["loss"].append(avg)
            history["step"].append(step)
            history["closure_max"].append(info["psi_norm_max"])
            elapsed = time.time() - t0
            tps = step * batch_size * seq_len / max(elapsed, 0.1)
            print(f"  step {step:>5}/{n_steps}  loss={avg:.4f}  "
                  f"closure={info['psi_norm_max']:.4f}  "
                  f"elapsed={elapsed:.0f}s  ({tps:.0f} tok/s)", flush=True)

    print(f"\n[CWF] Final loss: {history['loss'][-1]:.4f}, "
          f"Closure violations: {sum(1 for c in history['closure_max'] if c >= 1.0)}/{len(history['closure_max'])}")
    return history


def train_ar_vq(model, train_data, n_steps=5000, batch_size=32, seq_len=256,
                lr=1e-3, log_every=200, device="cuda",
                vq_alpha=0.1, codebook_reset_threshold=50):
    """训练 AR+VQ, 监控 loss + VQ perplexity + 防塌缩."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    train_data = train_data.to(device)
    N = train_data.shape[0]
    history = {"mse_loss": [], "ce_loss": [], "vq_loss": [], "perplexity": [], "step": []}

    print(f"\n{'='*70}\nTraining AR+VQ ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params)\n{'='*70}")

    t0 = time.time()
    window_mse, window_ce, window_vq, window_ppl = [], [], [], []
    for step in range(1, n_steps + 1):
        idx = torch.randint(0, N, (batch_size,))
        starts = torch.randint(0, train_data.shape[1] - seq_len - 1, (batch_size,))

        x_batch = torch.stack([train_data[i, s:s+seq_len] for i, s in zip(idx, starts)])
        y_target = torch.stack([train_data[i, s+seq_len] for i, s in zip(idx, starts)])

        model.train()
        y_hat, token_logits, vq_loss, perplexity = model(x_batch)

        mse_loss = F.mse_loss(y_hat, y_target)
        # 下一 token (真下一状态的 VQ 编码)
        with torch.no_grad():
            _, _, next_token, _ = model.vq(y_target.unsqueeze(1))
            next_token = next_token.squeeze(1)
        ce_loss = F.cross_entropy(token_logits, next_token)
        total_loss = mse_loss + vq_alpha * ce_loss + vq_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # 防码本塌缩: 若 perplexity 过低, 重置未使用的码
        if perplexity.item() < codebook_reset_threshold and step > 100:
            with torch.no_grad():
                # 统计每个码的使用频率
                _, _, ids, _ = model.vq(x_batch[:8])
                used = torch.zeros(model.codebook_size, device=device)
                used[ids.flatten().unique()] = 1
                unused = (used == 0).nonzero().squeeze()
                # 重置未使用码为 x_batch 的随机向量
                if len(unused) > 0 and len(ids.flatten()) > 0:
                    random_samples = x_batch[:8].reshape(-1, 3)[torch.randint(0, x_batch[:8].numel() // 3, (len(unused),))]
                    model.vq.embedding.weight.data[unused] = random_samples

        window_mse.append(mse_loss.item())
        window_ce.append(ce_loss.item())
        window_vq.append(vq_loss.item())
        window_ppl.append(perplexity.item())

        if step % log_every == 0 or step == 1 or step == n_steps:
            n = min(log_every, len(window_mse))
            avg_mse = sum(window_mse[-n:]) / n
            avg_ce = sum(window_ce[-n:]) / n
            avg_vq = sum(window_vq[-n:]) / n
            avg_ppl = sum(window_ppl[-n:]) / n
            history["mse_loss"].append(avg_mse)
            history["ce_loss"].append(avg_ce)
            history["vq_loss"].append(avg_vq)
            history["perplexity"].append(avg_ppl)
            history["step"].append(step)
            elapsed = time.time() - t0
            tps = step * batch_size * seq_len / max(elapsed, 0.1)
            print(f"  step {step:>5}/{n_steps}  mse={avg_mse:.3f}  ce={avg_ce:.3f}  "
                  f"vq={avg_vq:.3f}  ppl={avg_ppl:.1f}/512  "
                  f"elapsed={elapsed:.0f}s  ({tps:.0f} tok/s)", flush=True)

    print(f"\n[AR+VQ] Final MSE: {history['mse_loss'][-1]:.3f}, "
          f"Final perplexity: {history['perplexity'][-1]:.1f}/512")
    return history


def evaluate_oracle(val_data, seq_len=256, horizon=100, device="cuda"):
    """Oracle 不需要训练, 直接评估."""
    print(f"\n{'='*70}\nOracle (RK4) baseline\n{'='*70}")
    oracle = LorenzOracle(dt=0.01).to(device)
    val_data = val_data.to(device)

    mses_horizon = {h: [] for h in [1, 10, 50, 100]}
    for traj in val_data[:20]:  # 取 20 条轨迹
        # 取一段
        start = torch.randint(0, len(traj) - seq_len - horizon - 1, (1,)).item()
        x = traj[start:start+seq_len]  # (T, 3)
        # Ground truth K 步后状态
        gt = traj[start+seq_len:start+seq_len+horizon+1]  # (horizon+1, 3)

        # Oracle 自由 rollout
        pred = oracle.rollout(x[-1:], steps=horizon)[0]  # (horizon, 3)
        for h in mses_horizon:
            mses_horizon[h].append(F.mse_loss(pred[h-1], gt[h]).item())

    summary = {h: sum(v) / len(v) for h, v in mses_horizon.items()}
    print(f"  Oracle MSE @ horizon 1/10/50/100: "
          f"{summary[1]:.3f} / {summary[10]:.3f} / {summary[50]:.3f} / {summary[100]:.3f}")
    return summary


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    # 数据 - 在 GPU 上生成 (比 CPU 快 50x)
    print("\n[Data] Generating 200 train + 50 val trajectories on GPU...", flush=True)
    train_data = generate_lorenz_trajectories(n_trajectories=200, seq_len=1024, seed=42, device=device)
    val_data = generate_lorenz_trajectories(n_trajectories=50, seq_len=1024, seed=99, device=device)
    print(f"  Train: {train_data.shape}, Val: {val_data.shape}", flush=True)

    # 训练 CWF
    cwf_model = MultiChannelCWFLorenz(d=32, seq_len=256)
    cwf_history = train_cwf(cwf_model, train_data, n_steps=2000, device=device)

    # 训练 AR+VQ
    ar_model = ARVQBaselineLorenz(codebook_size=512, d_model=128, n_layers=3, seq_len=256)
    ar_history = train_ar_vq(ar_model, train_data, n_steps=2000, device=device)

    # Oracle baseline
    oracle_summary = evaluate_oracle(val_data, device=device)

    # 保存结果
    out = {
        "config": {
            "n_steps": 2000, "batch_size": 32, "seq_len": 256,
            "lr": 1e-3, "device": device,
        },
        "cwf_params": sum(p.numel() for p in cwf_model.parameters()),
        "ar_vq_params": sum(p.numel() for p in ar_model.parameters()),
        "cwf_history": cwf_history,
        "ar_vq_history": ar_history,
        "oracle_summary": oracle_summary,
    }
    out_path = Path(__file__).parent / "exp02_train_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[Saved] -> {out_path}")

    # 简明总结
    print(f"\n{'='*70}\nTRAINING SUMMARY (Day 3, 2000 steps)\n{'='*70}")
    print(f"  CWF params:       {out['cwf_params']/1e6:.2f}M")
    print(f"  AR+VQ params:     {out['ar_vq_params']/1e6:.2f}M")
    print(f"  CWF final MSE:    {cwf_history['loss'][-1]:.4f}")
    print(f"  AR+VQ final MSE:  {ar_history['mse_loss'][-1]:.4f}")
    print(f"  AR+VQ final ppl:  {ar_history['perplexity'][-1]:.1f}/512")
    print(f"  Oracle MSE@1/10/50/100: "
          f"{oracle_summary[1]:.3f}/{oracle_summary[10]:.3f}/"
          f"{oracle_summary[50]:.3f}/{oracle_summary[100]:.3f}")
    print(f"  CWF / AR ratio (MSE): "
          f"{cwf_history['loss'][-1] / max(ar_history['mse_loss'][-1], 0.01):.3f}x")


if __name__ == "__main__":
    main()
