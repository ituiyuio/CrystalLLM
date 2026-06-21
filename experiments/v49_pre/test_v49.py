"""V49 模型全面测试: 跨域 PPL + 文本生成质量评估.

测试目标:
  1. v28_val (in-distribution, GitHub code) — sanity check
  2. v46_clean_val (Python code, 不同语言) — 跨语言迁移
  3. English prose (in-memory) — 真 OOD
  4. 跨域生成质量 — 代码 vs 散文

关键问题: V49 PPL 1.0053 是在 code (高可预测) 上的结果. 模型是否真学到了语言模式, 还是只记忆了 v28 训练集?
"""
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from experiments.v49_pre.cmt_v2 import (
    LieRE_NoContext,
    WaveAttentionSoftmax,
    ComplexKANFFN_TrueMul,
)
from experiments.v49_pre.data_loader import _load_vocab
from experiments.v49_pre.verify_cmt_fixed import (
    ComplexLayerNorm,
    CMTBlockFixed,
    CMTFixed50M,
    generate_text,
    tokens_to_text,
)


# ---------------------------------------------------------------------------
# 测试数据: 多种来源
# ---------------------------------------------------------------------------
TEST_TEXTS = {
    "english_prose_1": (
        "The quick brown fox jumps over the lazy dog. "
        "This sentence contains every letter of the English alphabet. "
        "It is commonly used for testing typewriters and computer fonts. "
        "Dogs are often considered loyal companions to humans."
    ),
    "english_prose_2": (
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, "
        "filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole "
        "with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort."
    ),
    "code_python": (
        "def fibonacci(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    return fibonacci(n-1) + fibonacci(n-2)\n"
        "\n"
        "for i in range(10):\n"
        "    print(fibonacci(i))\n"
    ),
    "code_javascript": (
        "function fibonacci(n) {\n"
        "    if (n <= 1) return n;\n"
        "    return fibonacci(n-1) + fibonacci(n-2);\n"
        "}\n"
        "console.log(Array.from({length: 10}, (_, i) => fibonacci(i)));\n"
    ),
    "code_rust": (
        "fn fibonacci(n: u32) -> u32 {\n"
        "    if n <= 1 { return n; }\n"
        "    fibonacci(n-1) + fibonacci(n-2)\n"
        "}\n"
        "fn main() {\n"
        "    for i in 0..10 { println!(\"{}\", fibonacci(i)); }\n"
        "}\n"
    ),
    "random_chars": "xqzv pnm wkrt abcdef ghijkl mnopqr stuvwx yzABC 1234567890 !@#$%^&*()",
    "chinese_unicode": "今天天气很好,我们去公园散步,看到了很多漂亮的花。",
    "structured_json": (
        '{"name": "test", "value": 42, "items": [1, 2, 3, 4, 5], '
        '"nested": {"a": "hello", "b": "world"}}\n'
    ),
}


def load_test_datasets():
    """加载 v28_val, v46_clean_val + 内存中的英文 prose."""
    out = {}
    for name, p in [
        ("v28_val", "crystalllm/data/processed/v28_val.parquet"),
        ("v46_clean_val", "crystalllm/data/processed/v46_clean_val.parquet"),
    ]:
        try:
            df = pd.read_parquet(p)
            out[name] = df["text"].tolist()
        except Exception as e:
            out[name] = [f"ERROR: {e}"]
    return out


# ---------------------------------------------------------------------------
# 评估函数
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_text_ppl(model, text, stoi, device, seq_len=512, max_windows=20):
    """在单段文本上计算 char-level PPL."""
    ids = [stoi.get(c, 0) for c in text]
    n_windows = len(ids) // seq_len
    if n_windows == 0:
        # 文本太短, 用一个 window
        n_windows = 1
        ids = (ids + [0] * seq_len)[:seq_len]
    else:
        ids = ids[: n_windows * seq_len]
    arr = np.asarray(ids, dtype=np.int64).reshape(-1, seq_len)
    n_eval = min(n_windows, max_windows)

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    for i in range(n_eval):
        x = torch.from_numpy(arr[i:i+1]).to(device)
        x_in, y = x[:, :-1], x[:, 1:]
        logits = model(x_in)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item()
        total_tokens += y.numel()
    model.train()
    return math.exp(total_loss / max(total_tokens, 1)), n_eval


@torch.no_grad()
def evaluate_dataset_ppl(model, texts, stoi, device, seq_len=512, max_texts=20, max_windows_per_text=5):
    """在数据集上计算平均 PPL."""
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    n_texts = min(len(texts), max_texts)
    model.eval()
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
    return math.exp(total_loss / max(total_tokens, 1)), total_tokens


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------
def load_v49_checkpoint(ckpt_path, device):
    """加载 V49 final checkpoint."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    print(f"  trained for {args['n_steps']} steps, d_model={args['d_model']}, n_layers={args['n_layers']}")
    print(f"  training val_ppl: {ckpt.get('val_ppl', 'N/A')}")

    torch.manual_seed(42)  # 与训练时一致
    stoi, vocab_size = _load_vocab()
    model = CMTFixed50M(
        vocab_size=vocab_size,
        d_model=args["d_model"],
        n_layers=args["n_layers"],
        n_heads=args["n_heads"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  loaded model: {n_params:,} params ({n_params/1e6:.1f}M)")
    return model, args


# ---------------------------------------------------------------------------
# 主测试
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    stoi, _ = _load_vocab()
    itos = {v: k for k, v in stoi.items()}

    # 1. 加载模型
    ckpt_path = "experiments/v49_pre/results/v49_30k_full.final.pt"
    model, train_args = load_v49_checkpoint(ckpt_path, device)

    results = {
        "exp_id": "v49_test",
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(ckpt_path),
        "training_args": train_args,
        "datasets": {},
        "in_memory_texts": {},
        "generations": {},
    }

    # 2. 在 val/test 数据集上评估
    print("\n" + "=" * 70)
    print("[Test 1] 数据集 PPL 评估")
    print("=" * 70)
    datasets = load_test_datasets()
    for name, texts in datasets.items():
        if not texts or "ERROR" in texts[0]:
            print(f"  [{name}] SKIP ({texts[0] if texts else 'empty'})")
            continue
        ppl, total_tokens = evaluate_dataset_ppl(
            model, texts, stoi, device, seq_len=512, max_texts=50, max_windows_per_text=5,
        )
        n_chars_total = sum(len(t) for t in texts[:50])
        print(f"  [{name}] PPL={ppl:.4f} (50 texts, {total_tokens:,} tokens)")
        results["datasets"][name] = {
            "ppl": ppl,
            "total_tokens": total_tokens,
            "n_texts": min(len(texts), 50),
            "n_chars": n_chars_total,
        }

    # 3. 在内存测试文本上评估
    print("\n" + "=" * 70)
    print("[Test 2] 内存测试文本 PPL 评估 (跨域泛化)")
    print("=" * 70)
    for name, text in TEST_TEXTS.items():
        # 截断到 seq_len * max_windows 字符
        text_eval = text[: 512 * 20]
        ppl, n_windows = evaluate_text_ppl(
            model, text_eval, stoi, device, seq_len=512, max_windows=20,
        )
        # 文本中不在 vocab 的字符数
        oov = sum(1 for c in text_eval if c not in stoi)
        oov_rate = oov / max(len(text_eval), 1)
        print(f"  [{name:25s}] PPL={ppl:8.4f}  n_windows={n_windows:3d}  OOV={oov_rate:.2%}")
        results["in_memory_texts"][name] = {
            "ppl": ppl,
            "n_windows": n_windows,
            "oov_rate": oov_rate,
            "text_preview": text[:100],
        }

    # 4. 文本生成测试
    print("\n" + "=" * 70)
    print("[Test 3] 文本生成质量")
    print("=" * 70)
    generation_prompts = [
        ("code_c_prompt", "void main() {\n  int x = "),
        ("code_python_prompt", "def hello():\n    print(\""),
        ("english_prompt", "The quick brown fox "),
        ("agentic_prompt", "I'll help you with that. First, let me "),
    ]
    for name, prompt in generation_prompts:
        prompt_tokens = [stoi.get(c, 0) for c in prompt]
        print(f"\n[{name}] prompt: {prompt!r}")
        for temp in [0.5, 0.8, 1.0]:
            gen = generate_text(model, prompt_tokens, max_new_tokens=80,
                                temperature=temp, top_k=50)
            gen_text = tokens_to_text(gen, itos)
            print(f"  T={temp}: {gen_text!r}")
        results["generations"][name] = {
            "prompt": prompt,
            "samples": [
                {"temperature": t, "text": tokens_to_text(
                    generate_text(model, prompt_tokens, max_new_tokens=80,
                                  temperature=t, top_k=50), itos,
                )}
                for t in [0.5, 0.8, 1.0]
            ],
        }

    # 5. 保存结果
    output_path = "experiments/v49_pre/results/v49_test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 结果保存到: {output_path}")

    # 6. 总结
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print(f"训练 PPL (v28_val): {train_args.get('n_steps', 'N/A')} 步后 PPL≈1.005 (in-dist code)")
    if "v28_val" in results["datasets"]:
        print(f"复测 v28_val PPL: {results['datasets']['v28_val']['ppl']:.4f}")
    if "v46_clean_val" in results["datasets"]:
        v46 = results["datasets"]["v46_clean_val"]
        print(f"v46_clean_val (Python) PPL: {v46['ppl']:.4f}  — 跨语言迁移")

    # 检查 OOD 性能
    ood_ppls = []
    for name in ["english_prose_1", "english_prose_2", "code_python", "code_javascript",
                 "code_rust", "chinese_unicode", "structured_json"]:
        if name in results["in_memory_texts"]:
            ood_ppls.append((name, results["in_memory_texts"][name]["ppl"]))
    print(f"\n跨域 PPL 排序 (从最好到最差):")
    for name, ppl in sorted(ood_ppls, key=lambda x: x[1]):
        print(f"  {name:25s} PPL={ppl:8.4f}")


if __name__ == "__main__":
    main()
