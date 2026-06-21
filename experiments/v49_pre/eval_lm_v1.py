"""LM 评估脚本 v1.0 (基于 docs/standards/2026-06-22-lm-evaluation-standard.md).

5 维指标:
  1.1 In-distribution val_ppl (v28_val)
  1.2 Generation diversity (unique_n / total_n)
  1.3 Generation quality review (5 prompts × 3 temps, 自动判定)
  1.4 OOD PPL (≥ 2 datasets)
  1.5 BPC (跨 tokenization 可比)

输入: model checkpoint path
输出: JSON 报告 + Pass/Fail 决策
"""
import argparse
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.v49_pre.data_loader import _load_vocab
from experiments.v49_pre.verify_cmt_fixed import CMTFixed50M, generate_text, tokens_to_text
from experiments.v49_pre.exp_runner import build_50m_model
from experiments.v49_pre.wave_transformer import WaveFunctionTransformer, count_params


# ===========================================================================
# OOD 数据集
# ===========================================================================
def get_ood_texts():
    """返回 OOD 测试文本 (≥ 2 个数据集)."""
    ood = {}
    # OOD-1: v46_clean_val (Python code, 已知可用)
    try:
        import pandas as pd
        df = pd.read_parquet(PROJECT_ROOT / "crystalllm/data/processed/v46_clean_val.parquet")
        ood["v46_python"] = df["text"].tolist()
    except Exception as e:
        ood["v46_python"] = [f"ERROR: {e}"]
    # OOD-2: 内置 English prose + 中文 + 不同代码语言
    ood["english_prose"] = [
        ("The Art of War " * 200)[: 512 * 5],
        ("It was the best of times, it was the worst of times, " * 100)[: 512 * 5],
        ("To be, or not to be, that is the question: " * 200)[: 512 * 5],
    ]
    ood["code_javascript"] = [
        "function fibonacci(n) {\n" * 100,
        "const express = require('express');\n" * 100,
        "async function fetchData(url) {\n" * 100,
    ]
    return ood


# ===========================================================================
# 评估函数
# ===========================================================================
@torch.no_grad()
def eval_ppl_dataset(model, texts, stoi, device, seq_len=512, max_texts=20, max_windows_per_text=5):
    """在文本列表上计算 PPL."""
    if not texts or (isinstance(texts[0], str) and "ERROR" in texts[0]):
        return None, 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_tokens = 0.0, 0
    n_texts = min(len(texts), max_texts)
    model.eval()
    for i in range(n_texts):
        text = texts[i] if isinstance(texts[i], str) else texts[i][0]
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


def gen_with_diversity(model, prompt, stoi, itos, device, max_new=200, temp=0.8, top_k=50):
    """生成并返回多样性分析."""
    prompt_tokens = [stoi.get(c, 0) for c in prompt]
    gen_tokens = []
    with torch.no_grad():
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
    # 多样性
    n_total = len(gen_text)
    n_unique = len(set(gen_text))
    diversity = n_unique / max(n_total, 1)
    return {
        "text": gen_text,
        "n_unique": n_unique,
        "n_total": n_total,
        "diversity": diversity,
        "top5_chars": Counter(gen_text).most_common(5),
    }


def detect_repetition_run(text, min_run=5):
    """检测是否有 ≥ min_run 长度的重复字符."""
    pattern = re.compile(r"(.)\1{" + str(min_run - 1) + r",}")
    runs = pattern.findall(text)
    return len(runs) > 0, len(runs)


def is_locally_coherent(text, min_word_len=3):
    """启发式: 是否包含常见英文单词 (a, the, is, ...) 或代码关键字."""
    common_words = ["the", "and", "for", "is", "to", "in", "of", "return",
                    "function", "def", "if", "for", "while", "int", "void",
                    "const", "var", "let", "import", "class", "public"]
    text_lower = text.lower()
    hits = sum(1 for w in common_words if f" {w} " in f" {text_lower} " or
               f" {w}(" in f" {text_lower}" or f" {w}\n" in f" {text_lower}")
    return hits >= 2


# ===========================================================================
# 模型加载
# ===========================================================================
def load_model(ckpt_path, model_type, device):
    """根据 model_type 加载模型 (支持任意规模 baseline)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    stoi, vocab_size = _load_vocab()
    torch.manual_seed(42)
    if model_type == "cmt_fixed":
        model = CMTFixed50M(vocab_size=vocab_size, d_model=args["d_model"],
                            n_layers=args["n_layers"], n_heads=args["n_heads"]).to(device)
    elif model_type == "baseline":
        # 支持任意规模 baseline (从 checkpoint args 读取)
        from experiments.v49_pre.exp_runner import Transformer50M
        d_model = args.get("d_model", 640)
        n_layers = args.get("n_layers", 10)
        n_heads = args.get("n_heads", 8)
        d_ff = args.get("d_ff", 2560)
        max_seq_len = args.get("seq_len", args.get("max_seq_len", 2048))
        model = Transformer50M(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, d_ff=d_ff, max_seq_len=max_seq_len,
        ).to(device)
    elif model_type == "wave":
        model = WaveFunctionTransformer(vocab_size=vocab_size, dim=args["dim"],
                                         n_layers=args["n_layers"],
                                         n_heads=args["n_heads"],
                                         max_seq_len=args["seq_len"],
                                         use_wfnorm=args.get("use_wfnorm", False)).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    n_params = sum(p.numel() if not p.is_complex() else p.numel() * 2
                   for p in model.parameters() if p.requires_grad)
    return model, ckpt, stoi, vocab_size, n_params


# ===========================================================================
# 主评估
# ===========================================================================
PROMPTS = [
    ("english_simple", "The quick brown fox "),
    ("english_story", "Once upon a time, in a land far away, "),
    ("code_python", "def fibonacci(n):\n    "),
    ("code_c", "int main() {\n    return "),
    ("agentic", "I'll help you with that. First, "),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="checkpoint path")
    parser.add_argument("--model_type", required=True,
                        choices=["cmt_fixed", "baseline", "wave"])
    parser.add_argument("--output", required=True, help="output JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    print(f"Loading {args.model_type} from {args.ckpt}...")
    model, ckpt_data, stoi, vocab_size, n_params = load_model(
        args.ckpt, args.model_type, device
    )
    itos = {v: k for k, v in stoi.items()}
    print(f"  params: {n_params:,}")
    print(f"  trained n_steps: {ckpt_data['args'].get('n_steps', '?')}\n")

    results = {
        "exp_id": "eval_lm_v1",
        "model_type": args.model_type,
        "ckpt": str(args.ckpt),
        "timestamp": datetime.now().isoformat(),
        "n_params": n_params,
        "trained_n_steps": ckpt_data["args"].get("n_steps"),
    }

    # 1.1 In-distribution PPL (v28_val)
    print("[1.1] In-distribution PPL (v28_val)...")
    import pandas as pd
    df_val = pd.read_parquet(PROJECT_ROOT / "crystalllm/data/processed/v28_val.parquet")
    in_dist_ppl, in_dist_tokens = eval_ppl_dataset(
        model, df_val["text"].tolist(), stoi, device, max_texts=50,
    )
    print(f"  in-dist PPL: {in_dist_ppl:.4f} ({in_dist_tokens:,} tokens)")
    results["in_dist_ppl"] = in_dist_ppl

    # 1.2 + 1.3 Generation diversity + quality
    print("\n[1.2 + 1.3] Generation diversity & quality (5 prompts × 3 temps)...")
    diversity_results = {}
    for prompt_name, prompt in PROMPTS:
        diversity_results[prompt_name] = {}
        for temp in [0.5, 0.8, 1.0]:
            r = gen_with_diversity(model, prompt, stoi, itos, device,
                                   max_new=200, temp=temp)
            rep_run, n_runs = detect_repetition_run(r["text"], min_run=5)
            coherent = is_locally_coherent(r["text"])
            diversity_results[prompt_name][f"T{temp}"] = {
                "diversity": r["diversity"],
                "n_unique": r["n_unique"],
                "n_total": r["n_total"],
                "top5_chars": r["top5_chars"],
                "has_repetition_run": rep_run,
                "n_repetition_runs": n_runs,
                "is_locally_coherent": coherent,
                "text": r["text"][:100],  # 截断
            }
    # 平均 diversity
    all_divs = []
    n_coherent = 0
    n_repetition = 0
    for p_name, p_data in diversity_results.items():
        for temp_key, td in p_data.items():
            all_divs.append(td["diversity"])
            if td["is_locally_coherent"]:
                n_coherent += 1
            if td["has_repetition_run"]:
                n_repetition += 1
    avg_diversity = np.mean(all_divs)
    print(f"  avg diversity: {avg_diversity:.3f}")
    print(f"  locally coherent: {n_coherent}/15")
    print(f"  with repetition runs: {n_repetition}/15")
    results["generation_diversity_avg"] = avg_diversity
    results["n_coherent"] = n_coherent
    results["n_repetition"] = n_repetition
    results["generation_details"] = diversity_results

    # 1.4 OOD PPL
    print("\n[1.4] OOD PPL (≥ 2 datasets)...")
    ood = get_ood_texts()
    ood_results = {}
    ood_ppls = []
    for name, texts in ood.items():
        if name == "v46_python" and (not texts or "ERROR" in texts[0]):
            continue
        ppl, n_tok = eval_ppl_dataset(model, texts, stoi, device, max_texts=20)
        ood_results[name] = {"ppl": ppl, "tokens": n_tok}
        if ppl:
            ood_ppls.append(ppl)
        print(f"  {name}: PPL={ppl if ppl else 'N/A'}")
    ood_ppl_avg = np.mean(ood_ppls) if ood_ppls else None
    ood_ratio = ood_ppl_avg / in_dist_ppl if (ood_ppl_avg and in_dist_ppl) else None
    print(f"  OOD PPL avg: {ood_ppl_avg}, ratio to in-dist: {ood_ratio}")
    results["ood_ppls"] = ood_results
    results["ood_ppl_avg"] = ood_ppl_avg
    results["ood_ratio"] = ood_ratio

    # 1.5 BPC
    bpc = math.log2(in_dist_ppl) / 1.0  # char-level
    print(f"\n[1.5] BPC: {bpc:.3f} (log2({in_dist_ppl:.4f}) / 1 char/token)")
    results["bpc"] = bpc

    # 综合 Pass/Fail 决策 (按标准 v1.0)
    print("\n" + "=" * 60)
    print("[Decision] 5 维 Pass/Fail:")
    checks = {
        "1.1 PPL 合理范围 (1.5-3.0)": 1.5 <= in_dist_ppl <= 3.0,
        "1.2 Diversity ≥ 0.3": avg_diversity >= 0.3,
        "1.3 ≥ 9/15 局部合理": n_coherent >= 9,  # 至少 3/5 prompts at all temps
        "1.4 OOD ratio ≤ 5x": ood_ratio is not None and ood_ratio <= 5,
        "1.5 BPC 报告": bpc is not None,
        "无字符重复循环": n_repetition == 0,
        "PPL 远低于 random (<500)": in_dist_ppl < 500,
    }
    n_pass = 0
    for name, ok in checks.items():
        marker = "[OK]" if bool(ok) else "[FAIL]"
        print(f"  {marker} {name}")
        if bool(ok):
            n_pass += 1
    print(f"\n  Pass: {n_pass}/7")
    if n_pass == 7:
        decision = "PASS"
    elif n_pass >= 4:
        decision = "PARTIAL"
    else:
        decision = "FAIL"
    print(f"  Decision: {decision}")
    results["pass_count"] = int(n_pass)
    results["decision"] = decision
    # Convert numpy types to native Python for JSON
    results["checks"] = {k: bool(v) for k, v in checks.items()}

    # 保存
    def _to_native(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_to_native)
    print(f"\n[OK] 报告: {args.output}")


if __name__ == "__main__":
    main()
