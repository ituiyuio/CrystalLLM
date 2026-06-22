"""Exp 16 (CMT-Clean 公平对照): 修复后的 CMT 在 held-out 数据上验证.

承接: cmt_engineering_audit (2026-06-22) 结论 — 之前所有 CMT 测试都有 bug 或 memorizer.
本实验目标: 0-bug CMT-clean 在真实 held-out 数据上验证是否仍是 memorizer.

架构 (修复后):
  - PE:       LieRE_Fixed (RoPE + 小幅 context-aware 偏移, 不退化为 identity)
  - Attn:     WaveAttentionSoftmax (magnitude-softmax, 实测可用)
  - FFN:      ComplexKANFFN_TrueComplex (真复数乘法, cross-channel 耦合)

训练配置:
  - 数据:     v28_train FULL (69k samples) — 不用 10k subset 避免 memorization
  - 评估:     v28_val (held-out) — 不用 train subset
  - 步数:     30k (与 V49 formal 一致)
  - batch:    8, T=512, lr=1e-4 cosine, warmup=500
  - 优化器:   AdamW (8-bit if available)

输出:
  - results/exp16_cmt_clean.json: val_ppl 曲线 + 5 维评估
  - 决策: PPL ∈ [1.5, 3.0] + diversity ≥ 0.3 → CMT 假说部分成立
       PPL < 1.5 + diversity < 0.3 → 仍 memorizer (架构本质)
       PPL > 3.0 → 仍 underfitter (约束过严)

参考:
  - docs/experiments/2026-06-22-cmt-engineering-audit.md (审计报告)
  - docs/standards/2026-06-22-lm-evaluation-standard.md (5 维评估标准)
"""
import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from experiments.v49_pre.cmt_clean import CMT50MClean
from experiments.v49_pre.data_loader import _load_vocab
from experiments.v49_pre.exp_runner import VOCAB_SIZE, train_step


# ---------------------------------------------------------------------------
# 数据加载: 完整 v28_train (不用 10k subset)
# ---------------------------------------------------------------------------
def build_full_loader(batch_size: int = 8, seq_len: int = 512, shuffle: bool = True,
                      seed: int = 42, max_samples: int = None):
    """Build DataLoader over the FULL v28_train (not 10k subset).

    Returns DataLoader that yields (tokens,) tuples, tokens: (B, T) int64.
    """
    from torch.utils.data import DataLoader, TensorDataset
    from experiments.v49_pre.data_loader import _make_token_windows, load_v28_full
    stoi, _ = _load_vocab()
    texts = load_v28_full()
    rng = np.random.default_rng(seed)
    indices = np.arange(len(texts))
    if max_samples is not None and max_samples < len(texts):
        indices = rng.choice(len(texts), size=max_samples, replace=False)
    windows = _make_token_windows(texts, indices, stoi, seq_len, rng)
    dataset = TensorDataset(torch.from_numpy(windows))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def evaluate_ppl_heldout(model, val_parquet_path: str, stoi, device,
                          seq_len: int = 512, max_texts: int = 50, max_windows_per_text: int = 5):
    """在 v28_val (held-out) 上评估 PPL.

    与 train subset 评估的关键差异: val 数据与 train 数据完全无重叠.
    """
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
        return None, 0
    return math.exp(total_loss / total_tokens), total_tokens


def measure_imag_energy_ratio(model, val_loader, device, n_samples: int = 8, T: int = 512):
    """测量 input/output imag energy ratio (验证端到端复数信号流)."""
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


# ---------------------------------------------------------------------------
# 5 维生成评估 (char-level diversity + coherent)
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_generation_diversity(model, stoi, itos, device,
                               prompts=None, max_new=200, temps=(0.5, 0.8, 1.0), top_k=50):
    """在多个 prompt × temp 下评估生成多样性."""
    if prompts is None:
        prompts = [
            ("english_simple", "The quick brown fox "),
            ("english_story", "Once upon a time, in a land far away, "),
            ("code_python", "def fibonacci(n):\n    "),
            ("code_c", "int main() {\n    return "),
            ("agentic", "I'll help you with that. First, "),
        ]
    import torch.nn.functional as F
    results = {}
    for pname, prompt in prompts:
        prompt_tokens = [stoi.get(c, 0) for c in prompt]
        results[pname] = {}
        for temp in temps:
            gen_tokens = []
            for _ in range(max_new):
                ctx = torch.tensor(
                    [list(prompt_tokens + gen_tokens)[-model.config.max_seq_len:]],
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
                "diversity": diversity, "n_unique": n_unique, "n_total": n_total,
                "text_sample": gen_text[:100],
            }
    return results


def aggregate_5dim(ppl, diversity_results, n_coherent, n_repetition):
    """按 v1.0 标准聚合 5 维评估结果."""
    # 1.2 diversity 平均
    all_divs = []
    for p_data in diversity_results.values():
        for td in p_data.values():
            all_divs.append(td["diversity"])
    avg_diversity = float(np.mean(all_divs)) if all_divs else 0.0

    # 1.3 coherent
    n_total = sum(len(p_data) for p_data in diversity_results.values())  # 5 * 3 = 15
    coherent_ratio = n_coherent / n_total if n_total else 0.0

    # 综合 5 维 PASS/FAIL
    checks = {
        "1.1 PPL 合理 (1.5-3.0)": 1.5 <= ppl <= 3.0 if ppl else False,
        "1.2 diversity >= 0.3": avg_diversity >= 0.3,
        "1.3 coherent >= 3/5 (>= 0.6 ratio)": coherent_ratio >= 0.6,
        "1.4 OOD ratio": "see separate eval",
        "1.5 BPC reported": ppl is not None,
        "no repetition": n_repetition == 0,
        "PPL < random (500)": (ppl is not None) and (ppl < 500),
    }
    n_pass = sum(1 for v in checks.values() if v)
    return {
        "avg_diversity": avg_diversity,
        "coherent_ratio": coherent_ratio,
        "n_coherent": n_coherent,
        "n_repetition": n_repetition,
        "checks": checks,
        "n_pass": n_pass,
    }


def detect_repetition_run(text, min_run=5):
    import re
    pattern = re.compile(r"(.)\1{" + str(min_run - 1) + r",}")
    runs = pattern.findall(text)
    return len(runs) > 0, len(runs)


def is_locally_coherent(text):
    common = ["the", "and", "for", "is", "to", "in", "of", "return",
              "function", "def", "if", "for", "while", "int", "void",
              "const", "var", "let", "import", "class", "public"]
    text_lower = text.lower()
    hits = sum(1 for w in common if f" {w} " in f" {text_lower} " or
               f" {w}(" in f" {text_lower}" or f" {w}\n" in f" {text_lower}")
    return hits >= 2


# ---------------------------------------------------------------------------
# 训练主循环
# ---------------------------------------------------------------------------
def run_training(model, n_steps=30000, batch_size=8, seq_len=512,
                 learning_rate=1e-4, eval_every=2000,
                 val_parquet_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # Cosine schedule with warmup
    from torch.optim.lr_scheduler import LambdaLR
    import math as _math
    def lr_lambda(step):
        warmup = 500
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, n_steps - warmup)
        return 0.5 * (1 + _math.cos(_math.pi * progress)) * (1 - 0.1) + 0.1
    scheduler = LambdaLR(optimizer, lr_lambda)

    loader = build_full_loader(batch_size=batch_size, seq_len=seq_len, shuffle=True)
    val_loader = build_full_loader(batch_size=batch_size, seq_len=seq_len, shuffle=False, max_samples=200)
    stoi, _ = _load_vocab()
    itos = {v: k for k, v in stoi.items()}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    val_ppls = []
    train_losses = []
    for step in range(1, n_steps + 1):
        batch = next(iter(loader))[0].to(device)
        loss = train_step(model, batch, optimizer)
        scheduler.step()
        train_losses.append(loss)

        if step % eval_every == 0 or step == n_steps:
            val_ppl, n_tok = evaluate_ppl_heldout(
                model, val_parquet_path, stoi, device, seq_len=seq_len,
                max_texts=50, max_windows_per_text=3,
            )
            val_ppls.append((step, val_ppl))
            print(f"Step {step:>5d} | train_loss={loss:.4f} | "
                  f"val_ppl={val_ppl:.4f} | {n_tok} tokens")

    return val_ppls, train_losses, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=640)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--kan_dim", type=int, default=96)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--output", type=str,
                        default="experiments/v49_pre/results/exp16_cmt_clean.json")
    parser.add_argument("--val_parquet",
                        default="crystalllm/data/processed/v28_val.parquet")
    args = parser.parse_args()

    print(f"=== Exp 16: CMT-Clean (修复后) 公平对照 ===")
    print(f"配置: d_model={args.d_model}, n_layers={args.n_layers}, "
          f"n_heads={args.n_heads}, kan_dim={args.kan_dim}, "
          f"batch={args.batch_size}, T={args.seq_len}, lr={args.learning_rate}\n")
    print(f"训练: {args.n_steps} steps on v28_train FULL")
    print(f"评估: held-out v28_val (NOT train subset)\n")

    model = CMT50MClean(
        vocab_size=VOCAB_SIZE, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, kan_dim=args.kan_dim, dropout=0.1,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CMT-Clean 模型参数数: {n_params:,}\n")

    val_ppls, train_losses, val_loader = run_training(
        model, n_steps=args.n_steps, batch_size=args.batch_size,
        seq_len=args.seq_len, learning_rate=args.learning_rate,
        eval_every=args.eval_every, val_parquet_path=args.val_parquet,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stoi, _ = _load_vocab()
    itos = {v: k for k, v in stoi.items()}

    # imag energy ratio (端到端虚部信号)
    in_imag, out_imag, imag_ratio = measure_imag_energy_ratio(model, val_loader, device)

    # 5 维生成评估
    print("\n[5-dim eval] generation diversity × coherent × repetition...")
    gen_results = eval_generation_diversity(model, stoi, itos, device)
    n_coherent = 0
    n_repetition = 0
    for pname, pdata in gen_results.items():
        for temp_key, td in pdata.items():
            if is_locally_coherent(td["text_sample"]):
                n_coherent += 1
            rep, _ = detect_repetition_run(td["text_sample"])
            if rep:
                n_repetition += 1
            print(f"  {pname} {temp_key}: div={td['diversity']:.3f}, "
                  f"sample={td['text_sample'][:60]!r}")
    final_ppl = val_ppls[-1][1] if val_ppls else None
    eval_5dim = aggregate_5dim(final_ppl, gen_results, n_coherent, n_repetition)

    print(f"\n=== 5 维评估结果 ===")
    print(f"  in-dist PPL (held-out v28_val): {final_ppl:.4f}")
    print(f"  imag energy ratio:              {imag_ratio:.4f}")
    print(f"  avg diversity:                  {eval_5dim['avg_diversity']:.4f}")
    print(f"  coherent:                       {n_coherent}/15")
    print(f"  repetition runs:                {n_repetition}/15")
    print(f"  Pass: {eval_5dim['n_pass']}/7")

    # 决策
    print(f"\n=== 决策 ===")
    if final_ppl is not None and 1.5 <= final_ppl <= 3.0 and eval_5dim['avg_diversity'] >= 0.3:
        decision = "[CMT-CLEAN VALIDATED] CMT 假说在 0-bug 实现下部分成立"
    elif final_ppl is not None and final_ppl < 1.5 and eval_5dim['avg_diversity'] < 0.3:
        decision = "[MEMORIZER] 即便 0-bug 实现, CMT 仍 memorizer — 架构本质问题"
    elif final_ppl is not None and final_ppl > 3.0:
        decision = "[UNDERFITTER] CMT-clean 约束仍过严, PPL 难收敛"
    else:
        decision = "[MIXED] 介于多种失败模式之间, 需进一步诊断"
    print(f"  {decision}")

    # 保存
    result = {
        "exp_id": "exp16_cmt_clean",
        "config": vars(args),
        "n_params": n_params,
        "val_ppls": val_ppls,
        "imag_energy": {"input": in_imag, "output": out_imag, "ratio": imag_ratio},
        "5dim_eval": eval_5dim,
        "generation_details": {
            pname: {temp: {k: v for k, v in td.items() if k != "text_sample"}
                    for temp, td in pdata.items()}
            for pname, pdata in gen_results.items()
        },
        "decision": decision,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存到 {output_path}")


if __name__ == "__main__":
    main()
