"""Exp 20: CMT + BPE 5k step sanity check.

核心问题: 双相学习曲线 (Phase 1 underfit → Phase 2 memorize) 是 char-level 特有问题,
还是架构 vs next-token 任务的根本 mismatch?

方法: 保持 cmt_three_knives_fixed 三模块 + SmallCMTModel, 仅换 tokenization (BPE vocab=4100).
若 Phase 中间过渡打开 → BPE 修复 mismatch, CMT 有真 LM 潜力.
若仍 Phase 1 → Phase 2 跳变 → 问题在架构, BPE 无效.

实验设计 (与 Exp 19 完全对齐, 只换 tokenization):
  - 模型: SmallCMTModel (d_model=128, 2 层, 4 头, kan_dim=64) — 与 Exp 19 同
  - 数据: BPE-encoded v28_train 2k 样本子集 — 替代 char-level
  - 步数: 5000 — 与 Exp 19 同
  - lr: 1e-4 cosine + 200 warmup — 与 Exp 19 同
  - eval: 每 500 step

对比基线:
  - Exp 19 (char-level): final val_ppl 18.02 (underfit_in_progress)
  - Exp 16 (char-level 30k): final val_ppl 1.0097 (memorizer)
  - V49 baseline (char-level): val_ppl 2.36 (真 LM)

参考:
  - experiments/v49_pre/exp19_sanity_5k.py (char-level 同结构)
  - experiments/v49_pre/cmt_three_knives_fixed.py (三模块)
  - crystalllm/data/processed/bpe_data_loader.py (BPE 数据)
  - docs/experiments/2026-06-22-exp19-sanity-5k-results.md (对比基线)
"""
import argparse
import io
import json
import math
import sys
from pathlib import Path

# Force UTF-8 stdout for Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.v49_pre.bpe_data_loader import (
    build_bpe_loader,
    decode_tokens,
    encode_text,
    get_bpe_vocab_size,
    load_bpe_tokenizer,
)
from experiments.v49_pre.cmt_three_knives_fixed import (
    ComplexKANFFN_TrueComplex,
    LieRE_Fixed,
    WaveAttentionSoftmax,
)


# ---------------------------------------------------------------------------
# BPE 数据加载 (复用 exp19 接口, 改 BPE)
# ---------------------------------------------------------------------------
def evaluate_ppl_heldout_bpe(model, val_parquet_path: str, enc, device,
                              seq_len: int = 512, max_texts: int = 20,
                              max_windows_per_text: int = 3):
    """在 v28_val 上评估 PPL — BPE 版."""
    df = pd.read_parquet(val_parquet_path)
    texts = df["text"].tolist()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_tokens = 0.0, 0
    n_texts = min(len(texts), max_texts)
    model.eval()
    with torch.no_grad():
        for i in range(n_texts):
            text = texts[i]
            ids = enc.encode_ordinary(text)
            n_windows = len(ids) // seq_len
            if n_windows == 0:
                continue
            ids = ids[: n_windows * seq_len]
            arr = np.asarray(ids, dtype=np.int64).reshape(-1, seq_len)
            n_eval = min(n_windows, max_windows_per_text)
            for j in range(n_eval):
                x = torch.from_numpy(arr[j:j+1]).to(device)
                x_in, y = x[:, :-1], x[:, 1:]
                logits = model(x_in)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                total_loss += loss.item()
                total_tokens += y.numel()
    model.train()
    if total_tokens == 0:
        return None
    return math.exp(total_loss / total_tokens)


# ---------------------------------------------------------------------------
# SmallCMTModel (从 exp19 复用, 接受 BPE vocab_size)
# ---------------------------------------------------------------------------
class SmallCMTModel(nn.Module):
    """小尺寸 CMT 模型 — 用 cmt_three_knives_fixed 的三模块."""

    def __init__(self, vocab_size: int, d_model: int = 128,
                 n_layers: int = 2, n_heads: int = 4,
                 kan_dim: int = 64, max_seq_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, 2 * d_model)
        self.pos_emb = nn.Embedding(max_seq_len, 2 * d_model)
        self.layers = nn.ModuleList([
            _SmallCMTBlock(d_model, n_heads, kan_dim, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f_real = nn.LayerNorm(d_model)
        self.ln_f_imag = nn.LayerNorm(d_model)
        self.head = nn.Linear(2 * d_model, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        z = self.token_emb(x) + self.pos_emb(pos)
        for layer in self.layers:
            z = layer(z)
        d = self.d_model
        z = torch.cat([self.ln_f_real(z[..., :d]), self.ln_f_imag(z[..., d:])], dim=-1)
        return self.head(z)


class _SmallCMTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, kan_dim: int, dropout: float):
        super().__init__()
        self.pe = LieRE_Fixed(d_model)
        self.ln1_real = nn.LayerNorm(d_model)
        self.ln1_imag = nn.LayerNorm(d_model)
        self.attn = WaveAttentionSoftmax(d_model, n_heads=n_heads)
        self.ln2_real = nn.LayerNorm(d_model)
        self.ln2_imag = nn.LayerNorm(d_model)
        self.ffn = ComplexKANFFN_TrueComplex(d_model, kan_dim, dropout=dropout)

    def forward(self, z):
        d = z.size(-1) // 2
        z = z + self.pe(z)
        z_norm = torch.cat([self.ln1_real(z[..., :d]), self.ln1_imag(z[..., d:])], dim=-1)
        z = z + self.attn(z_norm)
        z_norm = torch.cat([self.ln2_real(z[..., :d]), self.ln2_imag(z[..., d:])], dim=-1)
        z = z + self.ffn(z_norm)
        return z


# ---------------------------------------------------------------------------
# 5 维生成评估 (BPE 版)
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_generation_diversity_bpe(model, enc, device,
                                   prompts=None, max_new=100, temps=(0.8, 1.0), top_k=50):
    if prompts is None:
        prompts = [
            ("english_simple", "The quick brown fox "),
            ("english_story", "Once upon a time, "),
            ("code_python", "def fibonacci(n):\n    "),
        ]
    results = {}
    for pname, prompt in prompts:
        prompt_tokens = enc.encode_ordinary(prompt)
        results[pname] = {}
        for temp in temps:
            gen_tokens = []
            for _ in range(max_new):
                ctx = torch.tensor(
                    [list(prompt_tokens + gen_tokens)[-model.max_seq_len:]],
                    device=device, dtype=torch.long,
                )
                logits = model(ctx)
                logits = logits[0, -1, :] / temp
                if top_k > 0:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[-1]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                gen_tokens.append(next_token)
            gen_text = enc.decode(gen_tokens)
            # BPE diversity: 按 subword 计算
            n_total = len(gen_tokens)
            n_unique = len(set(gen_tokens))
            diversity = n_unique / max(n_total, 1)
            results[pname][f"T{temp}"] = {
                "diversity": diversity,
                "n_unique": n_unique,
                "n_total": n_total,
                "text_sample": gen_text[:80],
            }
    return results


def detect_repetition_run(text, min_run=5):
    import re
    pattern = re.compile(r"(.)\1{" + str(min_run - 1) + r",}")
    runs = pattern.findall(text)
    return len(runs) > 0


def is_locally_coherent(text):
    common = ["the", "and", "for", "is", "to", "in", "of", "return",
              "function", "def", "if", "int", "void"]
    text_lower = text.lower()
    hits = sum(1 for w in common if f" {w} " in f" {text_lower} "
               or f" {w}(" in f" {text_lower}" or f" {w}\n" in f" {text_lower}")
    return hits >= 2


def measure_imag_energy_ratio(model, loader, device, n_samples: int = 4, T: int = 256):
    d = model.d_model
    model.eval()
    with torch.no_grad():
        x = next(iter(loader))[0][:n_samples, :T].to(device)
        pos = torch.arange(T, device=device).unsqueeze(0).expand(n_samples, T)
        z_in = model.token_emb(x) + model.pos_emb(pos)
        input_imag = z_in[..., d:].abs().mean().item()
        z = z_in
        for layer in model.layers:
            z = layer(z)
        output_imag = z[..., d:].abs().mean().item()
    model.train()
    return input_imag, output_imag, output_imag / max(input_imag, 1e-8)


# ---------------------------------------------------------------------------
# 主训练循环
# ---------------------------------------------------------------------------
def run_bpe_sanity(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    bpe_vocab = get_bpe_vocab_size()
    enc = load_bpe_tokenizer()
    print(f"BPE vocab size: {bpe_vocab}")

    model = SmallCMTModel(
        vocab_size=bpe_vocab,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        kan_dim=args.kan_dim,
        max_seq_len=args.seq_len,
        dropout=0.1,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== Exp 20: CMT + BPE 5k sanity check ===")
    print(f"模型: d_model={args.d_model}, n_layers={args.n_layers}, "
          f"n_heads={args.n_heads}, kan_dim={args.kan_dim}")
    print(f"参数量: {n_params:,} (对比 Exp 19 char-level 同模型)")
    print(f"数据: BPE-encoded {args.subset_size} 样本子集 (vocab={bpe_vocab})")
    print(f"训练: {args.n_steps} step, batch={args.batch_size}, T={args.seq_len}, lr={args.lr}\n")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    warmup = 200
    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, args.n_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress)) * 0.9 + 0.1
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # BPE 数据 (cache 加速)
    loader = build_bpe_loader(batch_size=args.batch_size, seq_len=args.seq_len,
                              subset_size=args.subset_size, seed=args.seed)

    val_ppl_curve = []
    train_losses = []
    loss_fn = nn.CrossEntropyLoss()

    print(f"{'Step':>5} | {'train_loss':>10} | {'val_ppl':>8} | {'notes':>30}")
    print("-" * 70)

    for step in range(1, args.n_steps + 1):
        try:
            batch = next(iter(loader))[0].to(device)
        except StopIteration:
            loader = build_bpe_loader(batch_size=args.batch_size, seq_len=args.seq_len,
                                      subset_size=args.subset_size, seed=args.seed)
            batch = next(iter(loader))[0].to(device)

        x, y = batch[:, :-1], batch[:, 1:]
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_losses.append(loss.item())

        if step % args.eval_every == 0 or step == args.n_steps:
            val_ppl = evaluate_ppl_heldout_bpe(
                model, args.val_parquet, enc, device,
                seq_len=args.seq_len, max_texts=20, max_windows_per_text=3,
            )
            val_ppl_curve.append((step, val_ppl))
            notes = ""
            if val_ppl is not None:
                if val_ppl < 1.05:
                    notes = "[MEMORIZER WARNING]"
                elif val_ppl > 5.0:
                    notes = "[UNDERFIT]"
                elif val_ppl > 1.1:
                    notes = "[non-memorizer trend]"
            print(f"{step:>5} | {loss.item():>10.4f} | "
                  f"{val_ppl if val_ppl else 0:>8.4f} | {notes:>30}")

    # 5 维评估
    print(f"\n[5-dim eval @ step {args.n_steps}]")
    in_imag, out_imag, imag_ratio = measure_imag_energy_ratio(model, loader, device)
    print(f"  imag energy ratio (in/out): {imag_ratio:.4f}")

    gen_results = eval_generation_diversity_bpe(model, enc, device)
    all_divs = []
    n_coherent = 0
    n_repetition = 0
    n_total = 0
    for pname, pdata in gen_results.items():
        for temp_key, td in pdata.items():
            all_divs.append(td["diversity"])
            n_total += 1
            if is_locally_coherent(td["text_sample"]):
                n_coherent += 1
            if detect_repetition_run(td["text_sample"]):
                n_repetition += 1
            sample_safe = repr(td["text_sample"][:60])
            print(f"  {pname} {temp_key}: div={td['diversity']:.3f}, sample={sample_safe}")

    avg_diversity = float(np.mean(all_divs)) if all_divs else 0.0
    coherent_ratio = n_coherent / n_total if n_total else 0.0
    final_ppl = val_ppl_curve[-1][1] if val_ppl_curve else None

    if len(val_ppl_curve) >= 3:
        early_ppl = np.mean([p for _, p in val_ppl_curve[:2] if p is not None])
        late_ppl = np.mean([p for _, p in val_ppl_curve[-2:] if p is not None])
        ppl_trend = "decreasing" if late_ppl < early_ppl * 0.95 else \
                    "stable_or_rising" if late_ppl > early_ppl * 1.05 else "stable"
        ppl_drop_ratio = (early_ppl - late_ppl) / early_ppl if early_ppl > 0 else 0
    else:
        ppl_trend = "insufficient_data"
        ppl_drop_ratio = 0

    # 决策 (核心: 对比 Exp 19 看 BPE 是否打破双相学习曲线)
    print(f"\n=== 决策 (与 Exp 19 char-level 对比) ===")
    exp19_ppl = 18.02  # 已知 baseline
    if final_ppl is None:
        decision = "[FAIL] val_ppl 计算失败"
    elif final_ppl < 1.05 and avg_diversity < 0.05:
        decision = "[MEMORIZER] BPE 也未打破 Phase 1→2 跳变, 架构 vs next-token 任务根本不匹配"
    elif final_ppl > 5.0 and ppl_trend == "decreasing":
        decision = "[UNDERFIT_IN_PROGRESS] BPE 仍未走完 Phase 1, 需更长训练"
    elif 1.1 < final_ppl <= 3.0 and avg_diversity > 0.05:
        decision = "[BPE_UNLOCKED] 真 LM 窗口打开! CMT + BPE 是有效组合, 可尝试 30k step"
    elif final_ppl < 1.05 and avg_diversity > 0.05:
        decision = "[BOUNDARY] PPL 偏低但 diversity 尚可, BPE 解锁部分能力"
    else:
        decision = "[MIXED] 介于多种状态, 需进一步诊断"

    print(f"  final val_ppl: {final_ppl}")
    print(f"  ppl curve trend: {ppl_trend} (drop={ppl_drop_ratio*100:.1f}%)")
    print(f"  avg diversity: {avg_diversity:.4f}")
    print(f"  coherent: {n_coherent}/{n_total} ({coherent_ratio*100:.0f}%)")
    print(f"  repetition: {n_repetition}/{n_total}")
    print(f"  imag energy ratio: {imag_ratio:.4f}")
    print(f"  对比 Exp 19 (char-level): val_ppl {exp19_ppl}")
    print(f"  {decision}")

    if args.output:
        result = {
            "exp_id": "exp20_bpe_sanity_5k",
            "config": vars(args),
            "n_params": n_params,
            "bpe_vocab_size": bpe_vocab,
            "val_ppl_curve": val_ppl_curve,
            "ppl_trend": ppl_trend,
            "ppl_drop_ratio": ppl_drop_ratio,
            "final_ppl": final_ppl,
            "avg_diversity": avg_diversity,
            "coherent_ratio": coherent_ratio,
            "n_repetition": n_repetition,
            "imag_energy": {"input": in_imag, "output": out_imag, "ratio": imag_ratio},
            "exp19_baseline_ppl": exp19_ppl,
            "decision": decision,
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到 {out_path}")

    return decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=5000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--kan_dim", type=int, default=64)
    parser.add_argument("--subset_size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_parquet",
                        default="crystalllm/data/processed/v28_val.parquet")
    parser.add_argument("--output",
                        default="experiments/v49_pre/results/exp20_bpe_sanity_5k.json")
    args = parser.parse_args()
    run_bpe_sanity(args)


if __name__ == "__main__":
    main()
