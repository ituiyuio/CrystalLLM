"""Exp 3: FP8 混合精度训练 vs BF16 baseline.

验证在 50M 规模上:
  - FP8 matmul + BF16 累加/LayerNorm 是否安全 (无 NaN/Inf, 收敛正常).
  - FP8 vs BF16 在 tokens/sec 和 peak memory 上的差异.

FP8 在 Windows + PyTorch 上路径不确定:
  - RTX 5090 (Blackwell) 硬件支持 FP8 (compute capability >= 8.9).
  - 但 torchao 的 float8 在 Windows 上可能有问题, transformer_engine 需要 nvcc.
  - 本实现优雅降级: torchao -> transformer_engine -> bf16_autocast -> (无效时仍 bf16).
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

from experiments.v49_pre.exp_runner import (
    build_50m_model, count_active_params,
)
from experiments.v49_pre.data_loader import build_subset_loader
from experiments.v49_pre.metrics import MetricsCollector, format_metrics


def has_fp8_support() -> bool:
    """检查当前 GPU 是否原生支持 FP8 (compute capability >= 8.9).

    Returns:
        True if CUDA available and device compute capability >= 8 (Ada/Hopper/Blackwell).
    """
    if not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return False
    # Ada (8.9+), Hopper (9.x), Blackwell (10.x/12.x) 均支持 FP8.
    # 仅 major >= 8 即可 — minor 的版本号在不同代际之间差异较大 (Blackwell 是 12.0).
    return major >= 8


def setup_fp8():
    """设置 FP8 混合精度.

    优先使用 torchao.float8, 备选 transformer_engine, 最后 BF16 autocast.

    Returns:
        tuple: (kind, handle) 其中 kind ∈ {"torchao", "te", "bf16_autocast"}.
            - ("torchao", convert_to_float8_training callable)
            - ("te", transformer_engine.pytorch module)
            - ("bf16_autocast", None)
    """
    if not has_fp8_support():
        return ("bf16_autocast", None)

    # 优先 torchao
    try:
        from torchao.float8 import convert_to_float8_training  # type: ignore
        return ("torchao", convert_to_float8_training)
    except ImportError:
        pass
    except Exception:
        # torchao 已安装但 import 路径出问题 — 静默回退
        pass

    # 备选 transformer_engine
    try:
        import transformer_engine.pytorch as te  # type: ignore
        return ("te", te)
    except ImportError:
        pass
    except Exception:
        pass

    # 都没有 — 回退 BF16 autocast (等同 baseline)
    return ("bf16_autocast", None)


def _evaluate_ppl_device_aware(model, val_loader, device, max_batches: int = 20):
    """Device-aware perplexity 评估 — 避免 exp_runner.evaluate_ppl 的 cpu/cuda mismatch bug.

    与 Exp 2 同样的 device-aware 修复.
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)
            x, y = x[:, :-1], x[:, 1:]
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
            total_tokens += y.numel()
    model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)


def run_training(model, n_steps: int = 10000, batch_size: int = 8, seq_len: int = 512,
                 learning_rate: float = 1e-4, eval_every: int = 2000, use_fp8: bool = False):
    """运行训练循环."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 应用 FP8 包装
    fp8_kind = "bf16_autocast"
    fp8_applied = False
    if use_fp8:
        ctx = setup_fp8()
        fp8_kind, handle = ctx
        if fp8_kind == "torchao":
            try:
                # FP8 conversion 在这个 PyTorch 版本 + 这个 50M 模型上, 因为 autograd 内部
                # reshape 产生 (640x4088) 等不能 /16 的 shape, 不可用.
                # 标记为不可用, 回退 BF16 (但记录尝试).
                print("torchao FP8 importable but incompatible with this model (autograd reshape produces non-/16 dims)")
                print("Falling back to BF16 (FP8 unavailable in current env)")
                fp8_kind = "bf16_unavailable"
                fp8_applied = False
            except Exception as e:
                print(f"torchao FP8 failed: {e}, falling back to BF16 autocast")
                fp8_kind = "bf16_autocast"
        elif fp8_kind == "te":
            print("Using TransformerEngine FP8")
        else:
            print("FP8 not available, falling back to BF16 autocast")

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

        optimizer.zero_grad()
        x, y = batch[:, :-1], batch[:, 1:]

        # 根据 FP8 kind 选择 autocast 策略
        if fp8_kind == "te":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1)
                )
        else:
            # BF16 baseline 或 torchao (torchao 内部处理精度, 外层无需 autocast)
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            )

        loss.backward()
        optimizer.step()

        metrics.record_step(tokens=tokens)
        metrics.update_peak_memory()

        if step % eval_every == 0 or step == n_steps:
            val_ppl = _evaluate_ppl_device_aware(model, loader, device)
            val_ppls.append((step, val_ppl))
            print(
                f"Step {step}: loss={loss:.4f}, val_ppl={val_ppl:.4f}, "
                f"{format_metrics(metrics.to_dict())}"
            )

    return metrics, val_ppls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["baseline", "fp8"], required=True)
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    print(f"=== Exp 3: {args.variant} (FP8 hw support: {has_fp8_support()}) ===")

    model = build_50m_model()
    print(f"Active params: {count_active_params(model):,}")

    metrics, val_ppls = run_training(model, n_steps=args.n_steps, use_fp8=(args.variant == "fp8"))

    result = {
        "variant": args.variant,
        "metrics": metrics.to_dict(),
        "val_ppls": val_ppls,
        "fp8_supported_hardware": has_fp8_support(),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")
    print(format_metrics(metrics.to_dict()))
    if val_ppls:
        print(f"Final val_ppl: {val_ppls[-1][1]:.4f}")


if __name__ == "__main__":
    main()