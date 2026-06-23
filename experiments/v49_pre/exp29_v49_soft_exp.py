"""
Exp 29: V49 1.2B baseline + Soft-Exp 解码头对比
================================================
加载 v49_scale_1.2b.final.pt (1.2B baseline, val_ppl=2.42)
评估 teacher-forcing PPL / argmax PPL / soft-exp PPL
目标: 验证 Soft-Exp 优势在 1.2B 规模是否仍维持 +50-80%
"""
import math, time, json, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v49_pre.exp_runner import Transformer50M

torch.manual_seed(42)
DEVICE = "cuda"

CHECKPOINT = PROJECT_ROOT / "experiments" / "v49_pre" / "results" / "v49_scale_1.2b.final.pt"


def load_model():
    print(f"[load] {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    args = ckpt['args']
    print(f"[load] args: d_model={args['d_model']}, n_layers={args['n_layers']}, n_heads={args['n_heads']}, d_ff={args['d_ff']}")
    print(f"[load] val_ppl: {ckpt['val_ppl']:.4f}")

    model = Transformer50M(
        vocab_size=2261,  # char-level
        d_model=args['d_model'],
        n_layers=args['n_layers'],
        n_heads=args['n_heads'],
        d_ff=args['d_ff'],
        max_seq_len=args['seq_len'],
        dropout=0.0,
    )
    model.load_state_dict(ckpt['model_state'])
    model = model.to(DEVICE)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[load] params: {n_params:,} ({n_params/1e9:.2f}B)")
    return model, args


def load_val_data():
    """加载 v28 验证集 (char-level)."""
    import pandas as pd
    p = PROJECT_ROOT / "crystalllm" / "data" / "processed" / "v28_val.parquet"
    df = pd.read_parquet(p)
    text = "\n".join(df["text"].astype(str).tolist())
    return text


@torch.no_grad()
def encode_chars(text, vocab):
    """char-level 编码."""
    return [vocab.get(c, 0) for c in text]


@torch.no_grad()
def eval_teacher_forcing(model, val_ids, num_seqs=20, seq_len=128):
    model.eval()
    n = len(val_ids) - seq_len - 1
    if n <= 0:
        return None
    starts = torch.randint(0, n, (num_seqs,))
    total_loss, total_tokens = 0.0, 0
    for s in starts:
        ids = val_ids[s:s+seq_len+1].to(DEVICE).unsqueeze(0)
        logits = model(ids[:, :-1])
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), ids[:, 1:].view(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += seq_len
    return math.exp(total_loss / total_tokens)


@torch.no_grad()
def feedback_argmax(logits, model):
    return model.token_emb(logits.argmax(dim=-1))


@torch.no_grad()
def feedback_soft(logits, model):
    probs = F.softmax(logits.float(), dim=-1)
    return torch.matmul(probs, model.token_emb.weight)


@torch.no_grad()
def eval_autoregressive(model, val_ids, feedback_fn, num_seqs=20, seq_len=64, name=""):
    model.eval()
    n = len(val_ids) - seq_len - 1
    if n <= 0:
        return None
    starts = torch.randint(0, n, (num_seqs,))
    total_loss, total_tokens = 0.0, 0
    t0 = time.time()
    for seq_i, s in enumerate(starts):
        ids = val_ids[s:s+seq_len+1].to(DEVICE).unsqueeze(0)
        cur = model.token_emb(ids[:, 0]).unsqueeze(1)
        for t in range(seq_len):
            S = cur.shape[1]
            pos = torch.arange(S, device=cur.device).unsqueeze(0)
            h = cur + model.pos_emb(pos)
            mask = model.layers[0].causal_mask[:S, :S]
            for layer in model.layers:
                h_ln = layer.ln1(h)
                h_attn, _ = layer.attn(h_ln, h_ln, h_ln, attn_mask=mask, need_weights=False)
                h = h + h_attn
                h = h + layer.ffn(layer.ln2(h))
            logits = model.head(model.ln_f(h[:, -1:, :]))
            target = ids[:, t+1]
            total_loss += F.cross_entropy(logits.view(-1, logits.size(-1)), target, reduction="sum").item()
            total_tokens += 1
            fb = feedback_fn(logits[:, -1, :], model)
            if fb.dim() == 2: fb = fb.unsqueeze(1)
            cur = torch.cat([cur[:, 1:], fb], dim=1)
        if (seq_i + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"    [{name}] seq {seq_i+1}/{num_seqs}  elapsed={elapsed:.0f}s", flush=True)
    return math.exp(total_loss / total_tokens)


def main():
    print(f"[init] device={DEVICE}")
    model, args = load_model()
    print(f"\n[init] loading val data ...")
    text = load_val_data()
    print(f"[init] val text len: {len(text):,} chars")
    with open(PROJECT_ROOT / "crystalllm/data/processed/char_vocab.json", encoding="utf-8") as f:
        vocab_meta = json.load(f)
    vocab = vocab_meta["stoi"] if isinstance(vocab_meta, dict) and "stoi" in vocab_meta else vocab_meta
    val_ids = torch.tensor(encode_chars(text, vocab), dtype=torch.long)
    print(f"[init] val tokens: {len(val_ids):,}")

    # 限制 val 大小以加速
    val_ids = val_ids[:50000]  # 50k tokens

    print("\n" + "="*70)
    print(f"V49 1.2B baseline + Soft-Exp 反馈对比")
    print("="*70)

    # 1. Teacher-forcing PPL
    print("\n[1/3] Teacher-forcing PPL (模型真实能力)")
    t0 = time.time()
    tf_ppl = eval_teacher_forcing(model, val_ids, num_seqs=20, seq_len=128)
    print(f"  TF PPL: {tf_ppl:.4f}  (time={time.time()-t0:.0f}s)")

    # 2. Autoregressive argmax
    print("\n[2/3] Autoregressive ARGMAX feedback (baseline inference)")
    t0 = time.time()
    ppl_arg = eval_autoregressive(model, val_ids, feedback_argmax, num_seqs=20, seq_len=64, name="argmax")
    print(f"  Argmax PPL: {ppl_arg:.4f}  (time={time.time()-t0:.0f}s)")

    # 3. Autoregressive soft-exp
    print("\n[3/3] Autoregressive SOFT-EXP feedback (continuous expected_embed)")
    t0 = time.time()
    ppl_soft = eval_autoregressive(model, val_ids, feedback_soft, num_seqs=20, seq_len=64, name="soft")
    print(f"  Soft-Exp PPL: {ppl_soft:.4f}  (time={time.time()-t0:.0f}s)")

    # 总结
    soft_adv = (ppl_arg - ppl_soft) / ppl_arg * 100
    print("\n" + "="*70)
    print("V49 1.2B + Soft-Exp RESULT")
    print("="*70)
    print(f"  Teacher-forcing PPL:     {tf_ppl:.4f}")
    print(f"  Autoregressive argmax:  {ppl_arg:.4f}   ({ppl_arg/tf_ppl:.0f}x exposure bias)")
    print(f"  Autoregressive soft:    {ppl_soft:.4f}   ({ppl_soft/tf_ppl:.0f}x exposure bias)")
    print(f"  Soft-Exp vs Argmax:     {soft_adv:+.2f}%")
    print()
    if soft_adv > 20:
        print(f"  >>> Soft-Exp 优势 {soft_adv:.1f}% 在 1.2B 规模仍维持, 这是工程级可用的 LM 改进")
    elif soft_adv > 0:
        print(f"  >>> Soft-Exp 优势 {soft_adv:.1f}% 较小但仍正向, 需更多数据确认")
    else:
        print(f"  >>> Soft-Exp 优势消失或反向, 1.2B 规模有特殊机制 (训练数据/规模足够)")

    out = {
        "model": "v49_1.2B_baseline",
        "n_params": sum(p.numel() for p in model.parameters()),
        "val_ppl_from_train": 2.42,
        "tf_ppl_now": tf_ppl,
        "argmax_ppl": ppl_arg,
        "soft_ppl": ppl_soft,
        "soft_advantage_pct": soft_adv,
        "exposure_bias_argmax_x": ppl_arg / tf_ppl,
        "exposure_bias_soft_x": ppl_soft / tf_ppl,
    }
    out_path = PROJECT_ROOT / "experiments" / "v49_pre" / "exp29_v49_soft_exp_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[save] -> {out_path}")


if __name__ == "__main__":
    main()