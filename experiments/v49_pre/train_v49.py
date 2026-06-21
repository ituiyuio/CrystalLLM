"""V49 正式训练: CMT-Fixed 50M 全量数据 + 8-bit AdamW + held-out 评估.

承接:
  - Exp 14/15: CMT-Fixed 架构 (LieRE_NoContext + WaveAttentionSoftmax + ComplexKANFFN_TrueMul)
  - Exp 4 (8-bit AdamW): v49_pre 唯一 PASS 的实验, 11% mem savings
  - verify_cmt_fixed.py: CMTFixed50M 实现 (复用)
  - Memory 2026-06-21-cmt-ablation-fix: 在 held-out v28_test 上验证泛化能力

架构: CMTFixed50M (CMTFixed50M from verify_cmt_fixed.py:58-86)
规模: 50M active params, d_model=640, n_layers=8, n_heads=8
数据: v28_train (69k 样本, 88M chars), v28_val (held-out)
优化: 8-bit AdamW (bitsandbytes), 梯度 clip
训练: 30k steps (formal), batch=8, T=512, lr=1e-4 cosine
评估: val PPL @ 1k/2k/5k/10k/15k/20k/25k/30k + 终态 test PPL
输出: experiments/v49_pre/results/v49_{timestamp}.json + .log
"""
import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn

from experiments.v49_pre.cmt_v2 import (
    LieRE_NoContext,
    WaveAttentionSoftmax,
    ComplexKANFFN_TrueMul,
)
from experiments.v49_pre.data_loader import (
    V28_TRAIN_PATH,
    VOCAB_PATH,
    _load_vocab,
    load_v28_full,
)
from experiments.v49_pre.metrics import MetricsCollector, format_metrics
from experiments.v49_pre.verify_cmt_fixed import (
    ComplexLayerNorm,
    CMTBlockFixed,
    CMTFixed50M,
    generate_text,
    tokens_to_text,
)


# ---------------------------------------------------------------------------
# 优化器: 8-bit AdamW (with fallback)
# ---------------------------------------------------------------------------
def build_optimizer(model, lr, weight_decay=0.01, use_8bit=True):
    """8-bit AdamW (bitsandbytes) 或 fallback 到 torch AdamW."""
    if use_8bit:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                model.parameters(), lr=lr, weight_decay=weight_decay,
                betas=(0.9, 0.95), eps=1e-8,
            )
            print(f"[OK] 使用 bnb.optim.AdamW8bit (8-bit)")
            return optimizer, "adamw_8bit"
        except ImportError:
            print(f"[WARN] bitsandbytes 不可用, fallback 到 torch.optim.AdamW")
        except Exception as e:
            print(f"[WARN] 8-bit AdamW 初始化失败 ({e}), fallback")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
        betas=(0.9, 0.95), eps=1e-8,
    )
    return optimizer, "adamw_32bit"


# ---------------------------------------------------------------------------
# 数据加载: 完整 v28_train + v28_val
# ---------------------------------------------------------------------------
def build_v28_train_loader(batch_size, seq_len, seed=42, val_fraction=0.0,
                            full_train=True, max_train_samples=None):
    """构建 v28_train DataLoader (可用全量或子集).

    Args:
        full_train: True = 全部 69k 样本; False = 10k subset
        max_train_samples: 限制样本数 (debug 用)
    """
    stoi, _ = _load_vocab()
    texts = load_v28_full()
    rng = np.random.default_rng(seed)

    n_samples = len(texts)
    if max_train_samples is not None:
        n_samples = min(n_samples, max_train_samples)
    if not full_train:
        n_samples = min(n_samples, 10000)

    indices = rng.choice(len(texts), size=n_samples, replace=False)

    # 预 tokenize 并切成 window
    all_ids = []
    for i in indices:
        text = texts[int(i)]
        ids = [stoi.get(c, 0) for c in text]
        all_ids.extend(ids)
    n_windows = len(all_ids) // seq_len
    all_ids = all_ids[: n_windows * seq_len]
    arr = np.asarray(all_ids, dtype=np.int64).reshape(n_windows, seq_len)
    perm = rng.permutation(n_windows)
    arr = arr[perm]

    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(torch.from_numpy(arr))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader, n_samples, n_windows


def build_v28_val_loader(batch_size, seq_len, n_batches=20, seed=99):
    """从 v28_val (held-out) 评估."""
    stoi, _ = _load_vocab()
    val_path = V28_TRAIN_PATH.parent / "v28_val.parquet"
    if not val_path.exists():
        print(f"[WARN] {val_path} 不存在, 评估跳过")
        return None
    df = pd_read_parquet(val_path)
    texts = df["text"].tolist()
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(texts), size=min(len(texts), 1000), replace=False)

    all_ids = []
    for i in indices:
        text = texts[int(i)]
        ids = [stoi.get(c, 0) for c in text]
        all_ids.extend(ids)
    n_windows = len(all_ids) // seq_len
    all_ids = all_ids[: n_windows * seq_len]
    arr = np.asarray(all_ids, dtype=np.int64).reshape(n_windows, seq_len)

    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(torch.from_numpy(arr))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader


def pd_read_parquet(path):
    """避免在函数顶 import pandas, 仅在 val 用时按需导入."""
    import pandas as pd
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_ppl(model, val_loader, device, max_batches=20):
    """在 val_loader 上计算 val PPL."""
    if val_loader is None:
        return None
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        x = batch[0].to(device) if isinstance(batch, (tuple, list)) else batch.to(device)
        x_in, y = x[:, :-1], x[:, 1:]
        logits = model(x_in)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item()
        total_tokens += y.numel()
    model.train()
    return math.exp(total_loss / max(total_tokens, 1))


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------
def train_step(model, batch, optimizer, device, max_grad_norm=1.0):
    """单步 next-token prediction 训练, 梯度 clip."""
    x = batch[0].to(device) if isinstance(batch, (tuple, list)) else batch.to(device)
    x_in, y = x[:, :-1], x[:, 1:]
    logits = model(x_in)
    loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return loss.item()


def cosine_lr_schedule(step, total_steps, warmup_steps, base_lr, min_lr_ratio=0.1):
    """Cosine LR schedule with linear warmup."""
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * (min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress)))


def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 1. 数据
    print(f"\n[1] 加载数据...")
    train_loader, n_train_samples, n_windows = build_v28_train_loader(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seed=args.seed,
        full_train=args.full_train,
        max_train_samples=args.max_train_samples,
    )
    val_loader = build_v28_val_loader(
        batch_size=args.batch_size, seq_len=args.seq_len, n_batches=20,
    )
    print(f"  train: {n_train_samples} samples, {n_windows} windows")
    print(f"  val: {'ready' if val_loader is not None else 'N/A'}")

    # 2. 模型
    print(f"\n[2] 构建 CMT-Fixed 50M...")
    torch.manual_seed(42)
    model = CMTFixed50M(
        vocab_size=_load_vocab()[1],
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量: {n_params:,} ({n_params/1e6:.1f}M)")

    # 3. 优化器
    print(f"\n[3] 构建优化器...")
    optimizer, opt_type = build_optimizer(model, lr=args.lr, use_8bit=args.use_8bit)
    print(f"  type: {opt_type}, lr: {args.lr}")

    # 4. 训练循环
    print(f"\n[4] 开始训练 ({args.n_steps} steps)...")
    metrics = MetricsCollector()
    metrics.start()

    val_ppls = []
    losses_window = []
    train_iter = iter(train_loader)
    t_start = time.time()
    last_log_step = 0
    for step in range(1, args.n_steps + 1):
        # LR schedule
        cur_lr = cosine_lr_schedule(step, args.n_steps, args.warmup_steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        loss = train_step(model, batch, optimizer, device, max_grad_norm=args.grad_clip)
        losses_window.append(loss)
        tokens = batch[0].numel() if isinstance(batch, (tuple, list)) else batch.numel()
        metrics.record_step(tokens=tokens)
        metrics.update_peak_memory()

        # 日志
        if step % args.log_every == 0 or step == 1:
            recent = sum(losses_window[-args.log_every:]) / max(len(losses_window[-args.log_every:]), 1)
            elapsed = time.time() - t_start
            tps = metrics.tokens_per_sec
            peak_mem = metrics.peak_memory_mb
            print(
                f"  step {step:5d}/{args.n_steps} | loss={recent:.4f} | "
                f"lr={cur_lr:.2e} | tps={tps:.0f} | mem={peak_mem:.0f}MB | "
                f"elapsed={elapsed:.0f}s", flush=True,
            )
            last_log_step = step

        # 评估
        if step % args.eval_every == 0 or step == args.n_steps:
            val_ppl = evaluate_ppl(model, val_loader, device, max_batches=20)
            val_ppls.append((step, val_ppl))
            print(f"  [EVAL] step {step:5d} | val_ppl={val_ppl:.4f}", flush=True)

        # Checkpoint
        if args.checkpoint_every and step % args.checkpoint_every == 0 and step < args.n_steps:
            ckpt_path = Path(args.output).with_suffix(f".step{step}.pt")
            torch.save({
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_ppl": val_ppls[-1][1] if val_ppls else None,
                "args": vars(args),
            }, ckpt_path)
            print(f"  [CKPT] saved {ckpt_path.name}", flush=True)

    # 5. 最终评估 + checkpoint
    print(f"\n[5] 训练完成, 最终评估...")
    final_val_ppl = evaluate_ppl(model, val_loader, device, max_batches=50)
    print(f"  final val_ppl (50 batches): {final_val_ppl:.4f}")

    # 文本生成
    print(f"\n[6] 文本生成测试...")
    stoi, _ = _load_vocab()
    itos = {v: k for k, v in stoi.items()}
    sample_text = "The quick brown fox"
    prompt_tokens = [stoi.get(c, 0) for c in sample_text]
    for temp in [0.7, 1.0]:
        gen = generate_text(model, prompt_tokens, max_new_tokens=100,
                            temperature=temp, top_k=50)
        print(f"  T={temp}: {tokens_to_text(gen, itos)!r}")

    # 保存最终 checkpoint
    final_ckpt = Path(args.output).with_suffix(".final.pt")
    torch.save({
        "step": args.n_steps,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_ppl": final_val_ppl,
        "args": vars(args),
    }, final_ckpt)
    print(f"  [CKPT] saved final {final_ckpt.name}", flush=True)

    # 7. 汇总
    elapsed = time.time() - t_start
    summary = {
        "exp_id": "v49_formal_training",
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "n_params": n_params,
        "n_train_samples": n_train_samples,
        "n_windows": n_windows,
        "optimizer_type": opt_type,
        "metrics": metrics.to_dict(),
        "val_ppls": val_ppls,
        "final_val_ppl": final_val_ppl,
        "total_time_s": elapsed,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 结果已保存: {args.output}")
    print(f"     Checkpoint: {final_ckpt}")
    print(f"     总耗时: {elapsed/60:.1f} min ({elapsed/3600:.2f} h)")
    return summary


def main():
    parser = argparse.ArgumentParser(description="V49 正式训练: CMT-Fixed 50M")
    parser.add_argument("--n_steps", type=int, default=30000, help="训练总步数 (默认 30k 正式)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=640)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率 (CMT 模型用 3e-4 比 1e-4 收敛更快)")
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--use_8bit", action="store_true", default=True, help="使用 8-bit AdamW")
    parser.add_argument("--no_8bit", action="store_false", dest="use_8bit")
    parser.add_argument("--full_train", action="store_true", default=True, help="用全量 v28_train (vs 10k subset)")
    parser.add_argument("--max_train_samples", type=int, default=None, help="限制训练样本数 (debug)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--checkpoint_every", type=int, default=10000, help="0=不保存中间 ckpt")
    parser.add_argument("--output", type=str, required=True, help="结果 JSON 路径")
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
