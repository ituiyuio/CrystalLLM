"""Exp 5: 课程学习 (按 loss 排序, 易→难) vs 随机 shuffle."""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.v49_pre.data_loader import (
    SUBSET_SIZE,
    _load_vocab,
    _make_token_windows,
    build_subset_loader,
    load_v28_full,
)
from experiments.v49_pre.exp_runner import (
    build_50m_model,
    count_active_params,
    evaluate_ppl,
)
from experiments.v49_pre.metrics import MetricsCollector, format_metrics


def sort_by_difficulty(losses: list) -> list:
    """返回按 loss 从小到大排列的样本索引 (易→难)."""
    return sorted(range(len(losses)), key=lambda i: losses[i])


def _evaluate_ppl_on_device(model, val_loader, device):
    """在指定 device 上计算 perplexity (DataLoader 输出在 CPU 上)."""
    import math
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)
            x, y = x[:, :-1], x[:, 1:]
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += y.numel()
    model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)


def estimate_difficulty(model, samples, seq_len: int = 512,
                        max_samples: int = 1000, device: str = "cpu",
                        stoi: dict | None = None) -> list:
    """用模型估计每个 sample 的 loss (作为难度指标).

    samples: list[str] (raw text) 或 list[list[int]] (pre-tokenized).
    为了保持 PoC 速度, 只用 max_samples 个样本估计.

    Returns:
        list[float]: 每个 sample 的 loss (与输入顺序一致).
    """
    if stoi is None:
        stoi, _ = _load_vocab()

    model = model.to(device)
    model.eval()
    losses = []
    with torch.no_grad():
        for s in samples[:max_samples]:
            # 如果是 raw text 则 tokenize
            if isinstance(s, str):
                ids = [stoi.get(c, 0) for c in s]
            else:
                ids = list(s)
            if len(ids) < seq_len + 1:
                continue
            x = torch.tensor(ids[:seq_len], dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(ids[1:seq_len + 1], dtype=torch.long, device=device).unsqueeze(0)
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
            losses.append(loss.item())
    model.train()
    return losses


def build_curriculum_subset_loader(batch_size: int = 8, seq_len: int = 512,
                                    difficulty_scores: list | None = None,
                                    seed: int = 42):
    """构建按难度排序的 loader (易→难).

    复用 build_subset_loader 的逻辑 (10k subset, char-tokenize + window),
    但应用按难度的 permutation 后不再 shuffle (shuffle=False).
    """
    if difficulty_scores is None:
        # 没有难度分数, 退化为随机
        return build_subset_loader(batch_size=batch_size, seq_len=seq_len, shuffle=True, seed=seed)

    stoi, _ = _load_vocab()
    texts = load_v28_full()

    # difficulty_scores 长度 = SUBSET_SIZE (与 build_subset_loader 选择 indices 一致)
    sorted_indices = sort_by_difficulty(difficulty_scores)
    sorted_indices = sorted_indices[:SUBSET_SIZE]

    rng = np.random.default_rng(seed)
    # 用与 build_subset_loader 相同的窗口化逻辑
    windows = _make_token_windows(texts, sorted_indices, stoi, seq_len, rng)

    dataset = TensorDataset(torch.from_numpy(windows))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def run_training_with_curriculum(model, n_steps: int = 10000, batch_size: int = 8,
                                  seq_len: int = 512, learning_rate: float = 1e-4,
                                  eval_every: int = 1000, use_curriculum: bool = False):
    """运行训练循环."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if use_curriculum:
        # 先用 1k step 训练 baseline 模型估计难度
        print("Estimating difficulty scores (1k warmup)...")
        warmup_loader = build_subset_loader(batch_size=batch_size, seq_len=seq_len, shuffle=True)
        model.train()
        warmup_it = iter(warmup_loader)
        for warmup_step in range(1000):
            try:
                batch = next(warmup_it)[0].to(device)
            except StopIteration:
                warmup_it = iter(warmup_loader)
                batch = next(warmup_it)[0].to(device)
            optimizer.zero_grad()
            x, y = batch[:, :-1], batch[:, 1:]
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
            loss.backward()
            optimizer.step()

        # 估计每个 sample 的难度 (用 100 个样本在 GPU 上, 避免 10k CPU inference 卡死)
        full = load_v28_full()
        scores = estimate_difficulty(
            model, full, seq_len=seq_len,
            max_samples=100, device=device,
        )
        print(f"Estimated difficulty for {len(scores)} samples")

        loader = build_curriculum_subset_loader(
            batch_size=batch_size, seq_len=seq_len,
            difficulty_scores=scores,
        )
        # 把模型移回 GPU 继续训练
        model = model.to(device)
    else:
        loader = build_subset_loader(batch_size=batch_size, seq_len=seq_len, shuffle=True)

    metrics = MetricsCollector()
    metrics.start()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    val_ppls = []
    start_step = 1001 if use_curriculum else 1
    for step in range(start_step, n_steps + 1):
        try:
            batch = next(iter(loader))[0].to(device)
        except StopIteration:
            # curriculum loader 用完后回到 baseline
            loader = build_subset_loader(batch_size=batch_size, seq_len=seq_len, shuffle=True)
            batch = next(iter(loader))[0].to(device)

        tokens = batch.numel()
        optimizer.zero_grad()
        x, y = batch[:, :-1], batch[:, 1:]
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        loss.backward()
        optimizer.step()

        metrics.record_step(tokens=tokens)
        metrics.update_peak_memory()

        eval_step = step - 1000 if use_curriculum else step
        if eval_step % eval_every == 0 or step == n_steps:
            val_ppl = _evaluate_ppl_on_device(model, loader, device)
            val_ppls.append((step, val_ppl))
            print(
                f"Step {step}: loss={loss:.4f}, val_ppl={val_ppl:.4f}, "
                f"{format_metrics(metrics.to_dict())}"
            )

    return metrics, val_ppls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["baseline", "curriculum"], required=True)
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    print(f"=== Exp 5: {args.variant} ===")

    model = build_50m_model()
    print(f"Active params: {count_active_params(model):,}")

    metrics, val_ppls = run_training_with_curriculum(
        model, n_steps=args.n_steps, use_curriculum=(args.variant == "curriculum")
    )

    result = {
        "variant": args.variant,
        "metrics": metrics.to_dict(),
        "val_ppls": val_ppls,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print(format_metrics(metrics.to_dict()))
    print(f"Final val_ppl: {val_ppls[-1][1]:.4f}")


if __name__ == "__main__":
    main()
