# Copyright (c) 2026 Yiming Wang <yimin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""gen_samples.py — v37 zero-z 生成质量检查

对每个 (checkpoint, z_mode) 组合, 生成 10 个样本 × 50 token, 测量:
  - 非空格率 (non_space_rate)
  - 代码结构样本数 (matched_count)

用途: 辅助 PPL 决策, 看 z=0 时 decoder 是否走默认分布 (坍缩到空格).
"""
import argparse
import json
import sys
import os
import random
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"
torch.manual_seed(42); np.random.seed(42); random.seed(42)

EVAL_DIR = Path(__file__).resolve().parent
V37_DIR = EVAL_DIR.parent
PROJECT_ROOT = V37_DIR.parent.parent

sys.path.insert(0, str(V37_DIR / "pipeline"))
from zero_z_eval import load_decoder, load_val_data

KEYWORDS = ["import ", "def ", "class ", "function ", "var ", "const ", "let ",
            "void ", "return ", "if ", "else", "{", "}", "->", "()", "(int", "(char",
            "#if", "#endif", "#else", "public ", "private "]


@torch.no_grad()
def gen_one_sample(decoder, z, BOS_ID, V, itos, n_tokens=50):
    """生成一个 n_tokens 长度的样本, 返回 (text, generated_ids)"""
    cur = torch.tensor([[BOS_ID]], dtype=torch.long, device="cuda")
    generated = [BOS_ID]
    for _ in range(n_tokens):
        logits = decoder(z, cur)
        logits_t = logits[:, -1, :]
        probs = F.softmax(logits_t, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1).item())
        generated.append(next_id)
        cur = torch.tensor([generated], dtype=torch.long, device="cuda")
    text = "".join(itos.get(t, "<unk>") for t in generated[1:])
    return text, generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=["v25", "v36"], required=True)
    parser.add_argument("--z_mode", choices=["encoded", "zero"], required=True)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--n_tokens", type=int, default=50)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    print(f"=== v37 gen_samples ===")
    print(f"  checkpoint: {args.checkpoint}, z_mode: {args.z_mode}")
    print(f"  n_samples: {args.n_samples}, n_tokens: {args.n_tokens}")

    decoder, V, D_Z, T = load_decoder(args.checkpoint, device="cuda")
    decoder.eval()
    val_texts, val_z, stoi, itos, V_check = load_val_data()
    assert V_check == V
    SPACE_ID = stoi.get(" ", -1)
    BOS_ID = stoi.get("<bos>", 1)

    non_space_rates = []
    samples = []
    matched = 0
    for i in range(args.n_samples):
        if args.z_mode == "encoded":
            z = val_z[i:i+1]
        else:
            z = torch.zeros(1, D_Z, device="cuda")
        text, gen_ids = gen_one_sample(decoder, z, BOS_ID, V, itos, n_tokens=args.n_tokens)
        samples.append(text)
        non_space_count = sum(1 for t in gen_ids[1:] if t != SPACE_ID)
        rate = non_space_count / args.n_tokens
        non_space_rates.append(rate)
        has_kw = any(kw in text for kw in KEYWORDS)
        if has_kw: matched += 1
        print(f"  sample {i}: non_space={rate:.2%} has_kw={has_kw} text={repr(text[:60])}")

    avg_rate = sum(non_space_rates) / len(non_space_rates)
    print(f"\n  [{args.checkpoint} + {args.z_mode}]")
    print(f"  avg non_space_rate: {avg_rate:.2%}")
    print(f"  matched (代码结构): {matched}/{args.n_samples}")

    result = {"checkpoint": args.checkpoint, "z_mode": args.z_mode,
              "n_samples": args.n_samples, "n_tokens": args.n_tokens,
              "non_space_rates": non_space_rates, "avg_rate": avg_rate,
              "samples": samples, "matched_count": matched}
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  saved: {args.output_json}")


if __name__ == "__main__":
    main()
