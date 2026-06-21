"""对比测试: V49 (CMT-Fixed) vs Exp 15 config (CMT-Fixed low-lr) vs baseline Transformer.

对每个模型:
  1. 加载
  2. 5 个 prompt × 3 温度 → 生成 100 chars
  3. 多样性分析 (unique chars, top-5 重复)
  4. 输出 PPL on 标准 prose (10x 重复 "The quick brown fox")

判断标准:
  - 多样性 OK (unique > 30 / 100 chars): 模型是 LM
  - 多样性差 (unique < 15 / 100 chars): 模型是 memorizer
"""
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn

from experiments.v49_pre.data_loader import _load_vocab
from experiments.v49_pre.verify_cmt_fixed import CMTFixed50M
from experiments.v49_pre.exp_runner import build_50m_model


TEST_PROMPTS = [
    ("english_simple", "The quick brown fox "),
    ("english_story", "Once upon a time, in a land far away, "),
    ("code_python", "def fibonacci(n):\n    "),
    ("code_c", "int main() {\n    return "),
    ("agentic", "I'll help you with that. First, "),
]


def load_model_from_ckpt(ckpt_path, model_type, device):
    """model_type: 'cmt_fixed' or 'baseline'"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    stoi, vocab_size = _load_vocab()

    torch.manual_seed(42)
    if model_type == "cmt_fixed":
        model = CMTFixed50M(
            vocab_size=vocab_size,
            d_model=args["d_model"],
            n_layers=args["n_layers"],
            n_heads=args["n_heads"],
        ).to(device)
    elif model_type == "baseline":
        model = build_50m_model(vocab_size=vocab_size).to(device)
    else:
        raise ValueError(model_type)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def gen_and_analyze(model, prompt, stoi, itos, device, max_new=100, temp=0.8, top_k=50):
    prompt_tokens = [stoi.get(c, 0) for c in prompt]
    generated = list(prompt_tokens)
    with torch.no_grad():
        for _ in range(max_new):
            ctx = torch.tensor(
                [generated[-model.config.max_seq_len:]],
                device=device, dtype=torch.long,
            )
            logits = model(ctx)
            logits = logits[0, -1, :] / temp
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[-1]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
    full = "".join(itos.get(t, "?") for t in generated)
    gen_part = full[len(prompt):]
    return {
        "full": full,
        "gen": gen_part,
        "n_unique": len(set(gen_part)),
        "n_total": len(gen_part),
        "top5_chars": Counter(gen_part).most_common(5),
        "top1_pct": Counter(gen_part).most_common(1)[0][1] / max(len(gen_part), 1),
    }


def diversity_score(result):
    """0-1, 越高越多樣. 用 n_unique / n_total."""
    return result["n_unique"] / max(result["n_total"], 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stoi, _ = _load_vocab()
    itos = {v: k for k, v in stoi.items()}

    models = []
    ckpt_v49 = "experiments/v49_pre/results/v49_30k_full.final.pt"
    ckpt_diag = "experiments/v49_pre/results/_diag_exp15_config.final.pt"
    ckpt_base = "experiments/v49_pre/results/_diag_baseline_10k.final.pt"
    if Path(ckpt_v49).exists():
        models.append(("v49_cmt_fixed_30k", ckpt_v49, "cmt_fixed"))
    if Path(ckpt_diag).exists():
        models.append(("diag_cmt_fixed_10k_lr1e-4", ckpt_diag, "cmt_fixed"))
    if Path(ckpt_base).exists():
        models.append(("diag_baseline_10k", ckpt_base, "baseline"))

    if not models:
        print("No checkpoints found!")
        return

    results = {}
    for name, ckpt, mtype in models:
        print(f"\n{'='*70}\n[{name}] loading {ckpt}...")
        model, ckpt_data = load_model_from_ckpt(ckpt, mtype, device)
        print(f"  n_params: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  trained n_steps: {ckpt_data['args'].get('n_steps', '?')}")
        print(f"  trained val_ppl: {ckpt_data.get('val_ppl', '?')}")
        results[name] = {}
        for prompt_name, prompt in TEST_PROMPTS:
            print(f"\n  [{prompt_name}] prompt: {prompt!r}")
            results[name][prompt_name] = {}
            for temp in [0.5, 0.8, 1.0]:
                r = gen_and_analyze(model, prompt, stoi, itos, device, max_new=100, temp=temp)
                div = diversity_score(r)
                marker = "[OK]" if div > 0.3 else "[WARN]" if div > 0.15 else "[BAD]"
                print(f"    T={temp} {marker} unique={r['n_unique']:3d}/{r['n_total']} "
                      f"top1={r['top1_pct']:.0%}  text={r['gen'][:60]!r}")
                results[name][prompt_name][f"T{temp}"] = {
                    "diversity": div,
                    "gen_text": r["gen"],
                    "n_unique": r["n_unique"],
                    "top1_pct": r["top1_pct"],
                    "top5_chars": r["top5_chars"],
                }
        del model
        torch.cuda.empty_cache()

    # 保存
    out_path = "experiments/v49_pre/results/v49_compare_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 结果保存: {out_path}")

    # 对比表
    print(f"\n{'='*70}\n对比总结 (avg diversity across prompts):")
    for name in results:
        diversities = []
        for prompt in results[name]:
            for temp_key in results[name][prompt]:
                diversities.append(results[name][prompt][temp_key]["diversity"])
        avg = np.mean(diversities)
        marker = "OK (LM)" if avg > 0.3 else "PARTIAL (semi-memorizer)" if avg > 0.15 else "FAIL (memorizer)"
        print(f"  {name:35s} avg_div={avg:.3f}  {marker}")


if __name__ == "__main__":
    main()
