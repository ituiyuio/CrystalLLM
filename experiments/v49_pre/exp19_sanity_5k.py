"""Exp 19: 5k step 小数据 sanity check — 验证修正版 CMT 是否展现非 memorizer 趋势.

承接: cmt_three_knives_fixed.py (5/5 验证通过) + Exp 18 H1_HARDENED (CMT 30k 仍 memorizer).

目的: 不再花 30k step, 用 5k step + 小数据快速判断:
  - 修正版 CMT 是否能在短训练内展现"非 memorizer 趋势"?
  - val_ppl 曲线是否还在下降 (未收敛到 memorization)?
  - diversity / coherent 是否脱离 memorizer 区间?

实验设计:
  - 模型: d_model=128, n_layers=2, n_heads=4, kan_dim=64 (≈ 5M params)
          (远比 50M 主线小, 但保留三刀架构完整)
  - 数据: 2k 样本子集 (v28_train 中随机抽, seed=42)
  - 步数: 5000 (对比 Exp 16 的 30000)
  - 评估: 每 500 step 算 val_ppl, 5000 step 时跑 5 维评估
  - 决策: val_ppl > 1.1 + diversity > 0.05 → 有非 memorizer 趋势
          val_ppl < 1.05 → memorizer (即使修正后)
          val_ppl > 5 + 仍下降 → 严重 underfit (需更长训练)

参考:
  - experiments/v49_pre/cmt_three_knives_fixed.py (修正版三刀)
  - docs/experiments/2026-06-22-three-knives-code-fix.md (修正记录)
  - docs/experiments/2026-06-22-exp18-a1-long.md (CMT 30k 失败先例)
"""
import argparse
import io
import json
import math
import sys
from pathlib import Path

# Force UTF-8 stdout for Windows console (avoid GBK UnicodeEncodeError)
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

from experiments.v49_pre.cmt_three_knives_fixed import (
    ComplexBSplineKAN_TrueComplex,
    ComplexKANFFN_TrueComplex,
    WaveAttentionSoftmax,
    LieRE_Fixed,
)
from experiments.v49_pre.data_loader import _load_vocab, load_v28_full, _make_token_windows
from experiments.v49_pre.exp_runner import VOCAB_SIZE


# ---------------------------------------------------------------------------
# 数据加载: 小子集 (2k 样本)
# ---------------------------------------------------------------------------
def build_small_loader(batch_size: int = 8, seq_len: int = 512,
                       subset_size: int = 2000, seed: int = 42):
    """2k 样本子集 DataLoader."""
    from torch.utils.data import DataLoader, TensorDataset
    stoi, _ = _load_vocab()
    texts = load_v28_full()
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(texts), size=subset_size, replace=False)
    windows = _make_token_windows(texts, indices, stoi, seq_len, rng)
    dataset = TensorDataset(torch.from_numpy(windows))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


def evaluate_ppl_heldout_small(model, val_parquet_path: str, stoi, device,
                                seq_len: int = 512, max_texts: int = 20,
                                max_windows_per_text: int = 3):
    """在 v28_val (held-out) 上评估 PPL — 小规模版本."""
    df = pd.read_parquet(val_parquet_path)
    texts = df["text"].tolist()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_tokens = 0.0, 0
    n_texts = min(len(texts), max_texts)
    model.eval()
    with torch.no_grad():
        for i in range(n_texts):
            text = texts[i]
            ids = [stoi.get(c, 0) for c in text]
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
# 小型 CMT 模型 (包装 cmt_three_knives_fixed 的三模块)
# ---------------------------------------------------------------------------
class SmallCMTModel(nn.Module):
    """小尺寸 CMT 模型 — 用 cmt_three_knives_fixed 的三模块.

    结构:
      token_emb (vocab → 2*d) + pos_emb → N × CMTBlock_ThreeKnives → LN → head → logits
    """

    def __init__(self, vocab_size: int, d_model: int = 128,
                 n_layers: int = 2, n_heads: int = 4,
                 kan_dim: int = 64, max_seq_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        # Embedding: 输出 cat[real | imag], shape (vocab, 2*d_model)
        self.token_emb = nn.Embedding(vocab_size, 2 * d_model)
        self.pos_emb = nn.Embedding(max_seq_len, 2 * d_model)
        # 三刀 block 堆叠
        self.layers = nn.ModuleList([
            _SmallCMTBlock(d_model, n_heads, kan_dim, dropout)
            for _ in range(n_layers)
        ])
        # 末尾 LN (real/imag 分别)
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
        # Embedding + PE
        z = self.token_emb(x) + self.pos_emb(pos)
        for layer in self.layers:
            z = layer(z)
        # 末尾 LN
        d = self.d_model
        z = torch.cat([self.ln_f_real(z[..., :d]), self.ln_f_imag(z[..., d:])], dim=-1)
        return self.head(z)


class _SmallCMTBlock(nn.Module):
    """单层 small CMT block (用 cmt_three_knives_fixed 三模块).

    结构: PE → LN → Attn → residual → LN → FFN → residual
    """

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
        # PE 注入
        z = z + self.pe(z)
        # Attention block
        z_norm = torch.cat([self.ln1_real(z[..., :d]), self.ln1_imag(z[..., d:])], dim=-1)
        z = z + self.attn(z_norm)
        # FFN block
        z_norm = torch.cat([self.ln2_real(z[..., :d]), self.ln2_imag(z[..., d:])], dim=-1)
        z = z + self.ffn(z_norm)
        return z


# ---------------------------------------------------------------------------
# 5 维生成评估 (简化版, 适配 5k step)
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_generation_diversity(model, stoi, itos, device,
                               prompts=None, max_new=100, temps=(0.8, 1.0), top_k=50):
    if prompts is None:
        prompts = [
            ("english_simple", "The quick brown fox "),
            ("english_story", "Once upon a time, "),
            ("code_python", "def fibonacci(n):\n    "),
        ]
    results = {}
    for pname, prompt in prompts:
        prompt_tokens = [stoi.get(c, 0) for c in prompt]
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
            gen_text = "".join(itos.get(t, "?") for t in gen_tokens)
            n_total = len(gen_text)
            n_unique = len(set(gen_text))
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
    """测量 input/output imag energy ratio — 验证端到端复数信号流."""
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
def run_sanity_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 模型
    model = SmallCMTModel(
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        kan_dim=args.kan_dim,
        max_seq_len=args.seq_len,
        dropout=0.1,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== Exp 19: 5k sanity check ===")
    print(f"模型: d_model={args.d_model}, n_layers={args.n_layers}, "
          f"n_heads={args.n_heads}, kan_dim={args.kan_dim}")
    print(f"参数量: {n_params:,} (对比主线 50M)")
    print(f"数据: {args.subset_size} 样本子集 (seed={args.seed})")
    print(f"训练: {args.n_steps} step, batch={args.batch_size}, T={args.seq_len}, lr={args.lr}\n")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Cosine schedule with warmup
    from torch.optim.lr_scheduler import LambdaLR
    warmup = 200
    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, args.n_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress)) * 0.9 + 0.1
    scheduler = LambdaLR(optimizer, lr_lambda)

    # 数据
    loader = build_small_loader(batch_size=args.batch_size, seq_len=args.seq_len,
                                subset_size=args.subset_size, seed=args.seed)
    stoi, _ = _load_vocab()
    itos = {v: k for k, v in stoi.items()}

    # 训练循环 + 周期性评估
    val_ppl_curve = []  # [(step, ppl), ...]
    train_losses = []
    loss_fn = nn.CrossEntropyLoss()

    print(f"{'Step':>5} | {'train_loss':>10} | {'val_ppl':>8} | {'notes':>30}")
    print("-" * 70)

    for step in range(1, args.n_steps + 1):
        try:
            batch = next(iter(loader))[0].to(device)
        except StopIteration:
            loader = build_small_loader(batch_size=args.batch_size, seq_len=args.seq_len,
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

        # 周期评估
        if step % args.eval_every == 0 or step == args.n_steps:
            val_ppl = evaluate_ppl_heldout_small(
                model, args.val_parquet, stoi, device,
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
    print(f"    input  imag mean |.|: {in_imag:.4f}")
    print(f"    output imag mean |.|: {out_imag:.4f}")

    gen_results = eval_generation_diversity(model, stoi, itos, device)
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
            sample_str = repr(td['text_sample'][:50])
            try:
                print(f"  {pname} {temp_key}: div={td['diversity']:.3f}, sample={sample_str}")
            except UnicodeEncodeError:
                sample_safe = sample_str.encode('ascii', 'replace').decode('ascii')
                print(f"  {pname} {temp_key}: div={td['diversity']:.3f}, sample={sample_safe}")

    avg_diversity = float(np.mean(all_divs)) if all_divs else 0.0
    coherent_ratio = n_coherent / n_total if n_total else 0.0
    final_ppl = val_ppl_curve[-1][1] if val_ppl_curve else None

    # 趋势判断: val_ppl 是否仍在下降?
    if len(val_ppl_curve) >= 3:
        early_ppl = np.mean([p for _, p in val_ppl_curve[:2] if p is not None])
        late_ppl = np.mean([p for _, p in val_ppl_curve[-2:] if p is not None])
        ppl_trend = "decreasing" if late_ppl < early_ppl * 0.95 else \
                    "stable_or_rising" if late_ppl > early_ppl * 1.05 else "stable"
        ppl_drop_ratio = (early_ppl - late_ppl) / early_ppl if early_ppl > 0 else 0
    else:
        ppl_trend = "insufficient_data"
        ppl_drop_ratio = 0

    # 决策
    print(f"\n=== 决策 ===")
    if final_ppl is None:
        decision = "[FAIL] val_ppl 计算失败"
    elif final_ppl < 1.05 and avg_diversity < 0.05:
        decision = "[MEMORIZER] 即便 5k step, 修正版 CMT 仍 memorizer — 架构 vs 任务不匹配"
    elif final_ppl > 5.0 and ppl_trend == "decreasing":
        decision = "[UNDERFIT_IN_PROGRESS] 仍在下降, 需更长训练 (16k-30k step)"
    elif 1.1 < final_ppl <= 3.0 and avg_diversity > 0.05:
        decision = "[NON_MEMORIZER_TREND] 修正版 CMT 有戏, 可尝试 30k step 正式实验"
    elif final_ppl < 1.05 and avg_diversity > 0.05:
        decision = "[BOUNDARY] PPL 偏低但 diversity 尚可, 需细查 (可能 train/val 分布差异)"
    else:
        decision = "[MIXED] 介于多种状态, 需进一步诊断"

    print(f"  final val_ppl: {final_ppl}")
    print(f"  ppl curve trend: {ppl_trend} (drop={ppl_drop_ratio*100:.1f}%)")
    print(f"  avg diversity: {avg_diversity:.4f}")
    print(f"  coherent: {n_coherent}/{n_total} ({coherent_ratio*100:.0f}%)")
    print(f"  repetition: {n_repetition}/{n_total}")
    print(f"  imag energy ratio: {imag_ratio:.4f}")
    print(f"  {decision}")

    # 保存
    if args.output:
        result = {
            "exp_id": "exp19_sanity_5k",
            "config": vars(args),
            "n_params": n_params,
            "val_ppl_curve": val_ppl_curve,
            "ppl_trend": ppl_trend,
            "ppl_drop_ratio": ppl_drop_ratio,
            "final_ppl": final_ppl,
            "avg_diversity": avg_diversity,
            "coherent_ratio": coherent_ratio,
            "n_repetition": n_repetition,
            "imag_energy": {"input": in_imag, "output": out_imag, "ratio": imag_ratio},
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
                        default="experiments/v49_pre/results/exp19_sanity_5k.json")
    args = parser.parse_args()
    run_sanity_training(args)


if __name__ == "__main__":
    main()
