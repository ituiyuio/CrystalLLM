"""Exp 21 (G): Baseline Transformer + BPE 5k sanity — 排除 BPE 混淆变量.

目的: Exp 20 显示 CMT + BPE 有 LM 信号 (coherent 2/6, bits/char 3.32). 但这是
CMT 架构的功劳, 还是 BPE 单独就有的? 必须用 baseline Transformer + BPE 做对照.

如果 baseline + BPE 比 baseline + char-level 显著更好 → BPE 单独就是改进方向, CMT 没功劳.
如果 baseline + BPE ~ baseline + char-level (val_ppl 2.36) → CMT + BPE 是真实信号.

实验设计 (与 Exp 19/20 完全对齐):
  - 模型: 标准 Transformer (无 CMT), 同规模 (d_model=128, 2 层, 4 头)
  - 数据: BPE-encoded v28_train 2k 样本子集 (与 Exp 20 同)
  - 步数: 5000, lr=1e-4 cosine + 200 warmup
  - eval: 每 500 step

对比:
  - Exp 19 char-level CMT (2.11M): bits/char 4.17, coherent 0/6
  - Exp 20 BPE CMT (3.05M): bits/char 3.32, coherent 2/6
  - Exp 21 (本) BPE baseline: 期望 ~baseline char-level bits/char
  - V49 1.2B baseline (1.2B char): bits/char 1.24
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
from experiments.v49_pre.exp_runner import TransformerBlock


# ---------------------------------------------------------------------------
# SmallBaselineModel (标准 Transformer, 同规模于 SmallCMTModel)
# ---------------------------------------------------------------------------
class SmallBaselineModel(nn.Module):
    """~3M 标准 Transformer (无 CMT, 标准 attention+FFN).

    规模对齐 SmallCMTModel (d_model=128, n_layers=2) 以便与 Exp 19/20 公平对比.
    """

    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 2,
                 n_heads: int = 4, d_ff: int = 512, max_seq_len: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, max_seq_len=max_seq_len)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # tie weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.token_emb(x) + self.pos_emb(pos)
        for layer in self.layers:
            h = layer(h)
        h = self.ln_f(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# BPE val PPL eval
# ---------------------------------------------------------------------------
def evaluate_ppl_heldout_bpe(model, val_parquet_path: str, enc, device,
                              seq_len: int = 512, max_texts: int = 20,
                              max_windows_per_text: int = 3):
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=5000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--subset_size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_parquet",
                        default="crystalllm/data/processed/v28_val.parquet")
    parser.add_argument("--output",
                        default="experiments/v49_pre/results/exp21_baseline_bpe_5k.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    bpe_vocab = get_bpe_vocab_size()
    enc = load_bpe_tokenizer()
    print(f"BPE vocab size: {bpe_vocab}")

    model = SmallBaselineModel(
        vocab_size=bpe_vocab,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
        dropout=0.1,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== Exp 21: Baseline + BPE 5k sanity ===")
    print(f"模型: 标准 Transformer (无 CMT)")
    print(f"  d_model={args.d_model}, n_layers={args.n_layers}, "
          f"n_heads={args.n_heads}, d_ff={args.d_ff}")
    print(f"参数量: {n_params:,}")
    print(f"对比: SmallCMTModel BPE 3.05M, V49 1.2B 1214M")
    print(f"数据: BPE-encoded {args.subset_size} 样本子集")
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
                bits_per_token = math.log2(val_ppl)
                bits_per_char = bits_per_token / 3.0  # BPE compression ~3x
                if val_ppl < 1.05:
                    notes = "[MEMORIZER]"
                elif val_ppl > 100:
                    notes = "[UNDERFIT]"
                else:
                    notes = f"[bits/char={bits_per_char:.2f}]"
            print(f"{step:>5} | {loss.item():>10.4f} | "
                  f"{val_ppl if val_ppl else 0:>8.4f} | {notes:>30}")

    # 5 维评估
    print(f"\n[5-dim eval @ step {args.n_steps}]")
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

    final_bits_per_token = math.log2(final_ppl) if final_ppl else None
    final_bits_per_char = final_bits_per_token / 3.0 if final_bits_per_token else None

    if len(val_ppl_curve) >= 3:
        early_ppl = np.mean([p for _, p in val_ppl_curve[:2] if p is not None])
        late_ppl = np.mean([p for _, p in val_ppl_curve[-2:] if p is not None])
        ppl_trend = "decreasing" if late_ppl < early_ppl * 0.95 else \
                    "stable_or_rising" if late_ppl > early_ppl * 1.05 else "stable"
    else:
        ppl_trend = "insufficient_data"

    # 4-way 对比
    exp19_ppl, exp19_bpc, exp19_coh = 18.02, 4.17, 0
    exp20_ppl, exp20_bpc, exp20_coh = 990.45, 3.32, 2

    print(f"\n=== 4-way 对比 ===")
    print(f"  模型                       | val_ppl | bits/char | coherent")
    print(f"  {'-'*60}")
    print(f"  CMT + char (Exp 19, 2.1M)  | {exp19_ppl:>7.2f} | {exp19_bpc:>9.2f} | {exp19_coh}/6")
    print(f"  CMT + BPE  (Exp 20, 3.1M)  | {exp20_ppl:>7.2f} | {exp20_bpc:>9.2f} | {exp20_coh}/6")
    print(f"  Base+ BPE  (Exp 21, {n_params/1e6:.1f}M) | {final_ppl:>7.2f} | {final_bits_per_char:>9.2f} | {n_coherent}/6")

    print(f"\n=== 决策 ===")
    if final_ppl is None:
        decision = "[FAIL]"
    elif final_bits_per_char < 3.32:
        decision = "[BPE_IS_KEY] baseline + BPE < CMT + BPE bits/char → BPE 单独就够, CMT 没功劳"
    elif final_bits_per_char > 3.32 + 0.5:
        decision = "[CMT_PLUS_BPE] baseline + BPE 显著差于 CMT + BPE → CMT 确实有改进 (注意 1.2B scale gap)"
    else:
        decision = "[CMT_MARGINAL] BPE 贡献为主, CMT 边际改进"

    print(f"  final val_ppl: {final_ppl}")
    print(f"  bits/char: {final_bits_per_char:.2f}")
    print(f"  ppl trend: {ppl_trend}")
    print(f"  diversity: {avg_diversity:.4f}")
    print(f"  coherent: {n_coherent}/{n_total}")
    print(f"  repetition: {n_repetition}/{n_total}")
    print(f"  {decision}")

    if args.output:
        result = {
            "exp_id": "exp21_baseline_bpe_5k",
            "config": vars(args),
            "n_params": n_params,
            "bpe_vocab_size": bpe_vocab,
            "val_ppl_curve": val_ppl_curve,
            "ppl_trend": ppl_trend,
            "final_ppl": final_ppl,
            "bits_per_char": final_bits_per_char,
            "avg_diversity": avg_diversity,
            "coherent": n_coherent,
            "n_repetition": n_repetition,
            "comparison": {
                "exp19_cmt_char": {"ppl": exp19_ppl, "bpc": exp19_bpc, "coherent": exp19_coh},
                "exp20_cmt_bpe": {"ppl": exp20_ppl, "bpc": exp20_bpc, "coherent": exp20_coh},
                "exp21_base_bpe": {"ppl": final_ppl, "bpc": final_bits_per_char, "coherent": n_coherent},
            },
            "decision": decision,
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到 {out_path}")


if __name__ == "__main__":
    main()
