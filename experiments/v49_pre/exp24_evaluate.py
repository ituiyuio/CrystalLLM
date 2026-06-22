"""Exp 24: 评估 — 5 维指标 (val_ppl / diversity / repetition / coherent / val-train gap).

Usage:
    cd D:/CrystaLLM && python -m experiments.v49_pre.exp24_evaluate
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v49_pre.data_loader import build_subset_loader, _load_vocab, VOCAB_PATH
from experiments.v49_pre.pe_modules import BlockCayleyPE, StandardRoPE, NoPE
from experiments.v49_pre.transformer_50m_swap_pe import Transformer50MSwapPE
from experiments.v49_pre.exp24_train import build_pe, D_MODEL, N_LAYERS, N_HEADS, D_FF, N_BLOCKS, BLOCK_SIZE


CKPT_DIR = PROJECT_ROOT / "experiments" / "v49_pre" / "exp24_ckpts"
RESULTS_PATH = PROJECT_ROOT / "docs" / "experiments" / "2026-06-22-cmt-cayley-pe-results.json"


def load_model(pe_name: str, ckpt_path: Path, vocab_size: int, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    pe_module = build_pe(pe_name, d_model=D_MODEL)
    model = Transformer50MSwapPE(
        vocab_size=vocab_size, d_model=D_MODEL, n_layers=N_LAYERS,
        n_heads=N_HEADS, d_ff=D_FF, pe_module=pe_module,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def evaluate_ppl_full(model, val_loader, device):
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x, y = x[:, :-1].to(device), x[:, 1:].to(device)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
            total_tokens += y.numel()
    return math.exp(total_loss / max(total_tokens, 1))


def evaluate_train_ppl(model, train_loader, device, max_batches=50):
    """估算 train PPL (取前 N 个 batch)."""
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= max_batches:
                break
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x, y = x[:, :-1].to(device), x[:, 1:].to(device)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
            total_tokens += y.numel()
    return math.exp(total_loss / max(total_tokens, 1))


def generate_and_score_diversity(model, stoi, itos, device, n_samples=20, gen_len=128, temperature=1.0):
    """生成 n_samples 段文本, 计算 4-gram distinct-1 ratio."""
    model.eval()
    samples = []
    with torch.no_grad():
        # 用 '<bos>' 或 stoi 中第一个字符作为起点
        bos_id = 0
        for _ in range(n_samples):
            ids = torch.tensor([[bos_id]], device=device)
            for _ in range(gen_len):
                logits = model(ids)
                logits = logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                ids = torch.cat([ids, next_id], dim=1)
            samples.append(ids[0].tolist())

    # 4-gram distinct-1 ratio (与 eval_lm_v1 对齐)
    all_4grams = set()
    total_4grams = 0
    for ids in samples:
        text = "".join([itos[i] for i in ids if i < len(itos)])
        for i in range(len(text) - 3):
            ng = text[i:i+4]
            all_4grams.add(ng)
            total_4grams += 1
    return len(all_4grams) / max(total_4grams, 1)


def evaluate_one(pe_name: str, device):
    print(f"\n=== Evaluating PE={pe_name} ===")
    # _load_vocab() 返回 (stoi, vocab_size); itos 需要从 VOCAB_PATH 直接读
    stoi, vocab_size = _load_vocab()
    import json
    vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    itos = vocab["itos"]

    # 加载 best ckpt
    ckpt_path = CKPT_DIR / f"exp24_{pe_name}_best.pt"
    if not ckpt_path.exists():
        print(f"  [SKIP] ckpt not found: {ckpt_path}")
        return None
    model, ckpt = load_model(pe_name, ckpt_path, vocab_size, device)

    # Data
    val_loader = build_subset_loader(batch_size=8, seq_len=256, shuffle=False, seed=43)
    train_loader = build_subset_loader(batch_size=8, seq_len=256, shuffle=False, seed=42)

    # Metric 1: val_ppl
    val_ppl = evaluate_ppl_full(model, val_loader, device)
    # Metric 5: val-train gap
    train_ppl = evaluate_train_ppl(model, train_loader, device)
    val_train_gap = (val_ppl - train_ppl) / train_ppl

    # Metric 2: diversity
    diversity = generate_and_score_diversity(model, stoi, itos, device, n_samples=20, gen_len=128)

    return {
        "pe": pe_name,
        "ckpt_step": ckpt["step"],
        "ckpt_val_ppl": ckpt["val_ppl"],
        "val_ppl_final": val_ppl,
        "train_ppl_est": train_ppl,
        "val_train_gap": val_train_gap,
        "diversity_4gram_distinct1": diversity,
        # Metric 3 (coherent) 和 4 (repetition) 需要 LLM-judge, 留待报告阶段人工评估
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for pe_name in ["cayley", "rope", "none"]:
        r = evaluate_one(pe_name, device)
        if r:
            results[pe_name] = r

    print("\n" + "=" * 60)
    print("Exp 24 评估结果汇总 (3 变体)")
    print("=" * 60)
    for pe_name, r in results.items():
        print(f"\n[{pe_name}]")
        for k, v in r.items():
            print(f"  {k}: {v}")

    # 保存 JSON
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {RESULTS_PATH}")