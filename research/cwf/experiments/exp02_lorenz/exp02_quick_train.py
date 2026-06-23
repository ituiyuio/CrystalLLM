"""
Day 3.5: Quick Re-train + Save Checkpoints
==========================================

Day 3 训练没保存 checkpoint. 重新跑一个 500 步的精简版,
保存模型权重供 Day 4 评估使用.

500 步训练 ~ 12 分钟 (CWF) + 1 分钟 (AR) = ~13 分钟
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


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    # 数据
    print("\n[Data] Generating trajectories...", flush=True)
    train_data = generate_lorenz_trajectories(n_trajectories=200, seq_len=1024, seed=42, device=device)
    print(f"  Train: {train_data.shape}", flush=True)

    # 训练 CWF (500 步)
    print("\n" + "=" * 70 + "\nTraining CWF (500 steps, save checkpoint)\n" + "=" * 70, flush=True)
    cwf = MultiChannelCWFLorenz(d=32, seq_len=256).to(device)
    optimizer = torch.optim.AdamW(cwf.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    loss_fn = nn.MSELoss()

    N = train_data.shape[0]
    seq_len = 256
    t0 = time.time()
    for step in range(1, 501):
        idx = torch.randint(0, N, (32,))
        starts = torch.randint(0, train_data.shape[1] - seq_len - 1, (32,))
        x_batch = torch.stack([train_data[i, s:s+seq_len] for i, s in zip(idx, starts)])
        y_target = torch.stack([train_data[i, s+seq_len] for i, s in zip(idx, starts)])

        cwf.train()
        y_hat, info = cwf(x_batch, return_info=True)
        loss = loss_fn(y_hat, y_target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cwf.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0 or step == 1 or step == 500:
            elapsed = time.time() - t0
            print(f"  step {step}/500  loss={loss.item():.4f}  "
                  f"closure={info['psi_norm_max']:.4f}  elapsed={elapsed:.0f}s", flush=True)

    cwf_ckpt = Path(__file__).parent / "results" / "cwf_lorenz_500.pt"
    cwf_ckpt.parent.mkdir(exist_ok=True)
    torch.save({"model_state": cwf.state_dict(), "config": {"d": 32, "seq_len": 256}}, cwf_ckpt)
    print(f"\n[CWF] Saved -> {cwf_ckpt}", flush=True)

    # 训练 AR+VQ (500 步)
    print("\n" + "=" * 70 + "\nTraining AR+VQ (500 steps, save checkpoint)\n" + "=" * 70, flush=True)
    ar = ARVQBaselineLorenz(codebook_size=512, d_model=128, n_layers=3, seq_len=256).to(device)
    optimizer = torch.optim.AdamW(ar.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

    t0 = time.time()
    for step in range(1, 501):
        idx = torch.randint(0, N, (32,))
        starts = torch.randint(0, train_data.shape[1] - seq_len - 1, (32,))
        x_batch = torch.stack([train_data[i, s:s+seq_len] for i, s in zip(idx, starts)])
        y_target = torch.stack([train_data[i, s+seq_len] for i, s in zip(idx, starts)])

        ar.train()
        y_hat, token_logits, vq_loss, perplexity = ar(x_batch)

        mse_loss = F.mse_loss(y_hat, y_target)
        with torch.no_grad():
            _, _, next_token, _ = ar.vq(y_target.unsqueeze(1))
            next_token = next_token.squeeze(1)
        ce_loss = F.cross_entropy(token_logits, next_token)
        total_loss = mse_loss + 0.1 * ce_loss + vq_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(ar.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # 防码本塌缩
        if perplexity.item() < 100 and step > 50:
            with torch.no_grad():
                _, _, ids, _ = ar.vq(x_batch[:8])
                used = torch.zeros(ar.codebook_size, device=device)
                used[ids.flatten().unique()] = 1
                unused = (used == 0).nonzero().squeeze()
                if len(unused) > 0:
                    samples = x_batch[:8].reshape(-1, 3)[torch.randint(0, x_batch[:8].numel() // 3, (len(unused),))]
                    ar.vq.embedding.weight.data[unused] = samples

        if step % 100 == 0 or step == 1 or step == 500:
            elapsed = time.time() - t0
            print(f"  step {step}/500  mse={mse_loss.item():.3f}  ce={ce_loss.item():.3f}  "
                  f"ppl={perplexity.item():.1f}/512  elapsed={elapsed:.0f}s", flush=True)

    ar_ckpt = Path(__file__).parent / "results" / "ar_vq_lorenz_500.pt"
    torch.save({"model_state": ar.state_dict(),
                "config": {"codebook_size": 512, "d_model": 128, "n_layers": 3, "seq_len": 256}},
               ar_ckpt)
    print(f"\n[AR+VQ] Saved -> {ar_ckpt}", flush=True)
    print("\n[OK] Quick re-train complete.")


if __name__ == "__main__":
    main()
