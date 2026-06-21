"""Exp 13 (CMT ablation Fix-4): cmt_real_init_v2 - M5 虚部梯度冻结修复.

修复内容:
  - 应用 RealInitV2 到 cmt_full 模型: imag 权重 N(0.1, 0.02) 有偏初始化
  - 修复 M5: 虚部信号在初始化时被随机抵消 (E[c_g * μ_g] ≈ 0), 偏置打破对称性
  - 不改变架构, 仅改变初始化

架构: 与 Exp 8 完全一致, 仅 imag 权重初始化不同

参考文档:
  - docs/notes/2026-06-21-wave-function-scalpel.md §📐 M5
  - docs/superpowers/specs/2026-06-21-cmt-ablation-fix-design.md §3.4
"""
import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from experiments.v49_pre.exp_runner import VOCAB_SIZE
from experiments.v49_pre.cmt_v2 import apply_real_init_v2
from experiments.v49_pre.exp8_cmt_full import CMT50M, build_cmt_50m
from experiments.v49_pre.data_loader import build_subset_loader
from experiments.v49_pre.metrics import MetricsCollector, format_metrics


def count_active_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_ppl_device_aware(model, val_loader, device, max_batches=20):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device)
            x_in, y = x[:, :-1], x[:, 1:]
            logits = model(x_in)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
            total_tokens += y.numel()
    model.train()
    return math.exp(total_loss / max(total_tokens, 1))


def measure_imag_energy_ratio(model, val_loader, device, n_samples=8, T=512):
    d = model.config.d_model
    model.eval()
    with torch.no_grad():
        x = next(iter(val_loader))[0][:n_samples, :T].to(device)
        pos = torch.arange(T, device=device).unsqueeze(0).expand(n_samples, T)
        z_in = model.token_emb(x) + model.pos_emb(pos)
        input_imag = z_in[..., d:].abs().mean().item()
        z = z_in
        for layer in model.layers:
            z = layer(z)
        output_imag = z[..., d:].abs().mean().item()
    model.train()
    return input_imag, output_imag, output_imag / max(input_imag, 1e-8)


def train_step_inline(model, batch, optimizer, loss_fn=None):
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    x = batch[:, :-1]
    y = batch[:, 1:]
    logits = model(x)
    loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def run_training(model, n_steps=10000, batch_size=8, seq_len=512,
                 learning_rate=1e-4, eval_every=2000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loader = build_subset_loader(batch_size=batch_size, seq_len=seq_len, shuffle=True)
    metrics = MetricsCollector()
    metrics.start()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    val_ppls = []
    for step in range(1, n_steps + 1):
        batch = next(iter(loader))[0].to(device)
        tokens = batch.numel()
        loss = train_step_inline(model, batch, optimizer)
        metrics.record_step(tokens=tokens)
        metrics.update_peak_memory()
        if step % eval_every == 0 or step == n_steps:
            val_ppl = evaluate_ppl_device_aware(model, loader, device)
            val_ppls.append((step, val_ppl))
            print(f"Step {step}: loss={loss:.4f}, val_ppl={val_ppl:.4f}, "
                  f"{format_metrics(metrics.to_dict())}")
    return metrics, val_ppls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=640)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--kan_dim", type=int, default=96)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    print(f"=== Exp 13: Fix-4 (RealInitV2) - M5 虚部梯度冻结修复 ===")
    print(f"配置: d_model={args.d_model}, n_layers={args.n_layers}, "
          f"n_heads={args.n_heads}, kan_dim={args.kan_dim}, "
          f"batch={args.batch_size}, T={args.seq_len}, lr={args.learning_rate}")
    print(f"架构与 Exp 8 完全一致, 仅 imag 权重初始化改为 N(0.1, 0.02)\n")

    # 加载 cmt_full 模型 (与 Exp 8 同), 但应用 RealInitV2
    model = build_cmt_50m(
        d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, kan_dim=args.kan_dim,
    )
    # 应用虚部偏置初始化
    apply_real_init_v2(model)
    print(f"已应用 RealInitV2: imag 权重 N(0.1, 0.02), 其他 xavier_uniform")

    n_params = count_active_params(model)
    print(f"CMT-Fix4 模型参数数: {n_params:,}")

    metrics, val_ppls = run_training(
        model, n_steps=args.n_steps, batch_size=args.batch_size,
        seq_len=args.seq_len, learning_rate=args.learning_rate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_loader = build_subset_loader(batch_size=args.batch_size, seq_len=args.seq_len)
    in_imag, out_imag, imag_ratio = measure_imag_energy_ratio(model, val_loader, device)
    print(f"\nImag energy: input={in_imag:.4f}, output={out_imag:.4f}, ratio={imag_ratio:.4f}")

    result = {
        "exp_id": "exp13_cmt_real_init_v2",
        "fix_target": "M5 (虚部梯度冻结)",
        "config": vars(args),
        "n_params": n_params,
        "metrics": metrics.to_dict(),
        "val_ppls": val_ppls,
        "imag_energy": {"input": in_imag, "output": out_imag, "ratio": imag_ratio},
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到 {args.output}")
    print(format_metrics(metrics.to_dict()))
    if val_ppls:
        print(f"\n=== Val PPL 曲线 ===")
        for step, ppl in val_ppls:
            print(f"  step {step}: {ppl:.4f}")
        final_ppl = val_ppls[-1][1]
        print(f"\nFinal val PPL @ step {args.n_steps}: {final_ppl:.4f}")
        print(f"\n=== 与 Exp 8 对比 ===")
        print(f"  Exp 8 (xavier init):     32.58")
        print(f"  Exp 13 (RealInitV2):     {final_ppl:.4f}")
        if final_ppl < 25:
            print(f"  [OK] PPL < 25, Fix-4 有效, 虚部有偏初始化解决 M5")
        elif final_ppl < 30:
            print(f"  [PARTIAL] 有所改善")
        else:
            print(f"  [X] PPL ≥ 30, Fix-4 无效, M5 非主要失败机制")


if __name__ == "__main__":
    main()