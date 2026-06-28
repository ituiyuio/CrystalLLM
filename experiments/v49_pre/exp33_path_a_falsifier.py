"""
Exp 33: V49 1.2B + Path A falsifier (sinusoidal char embedding)
=================================================================
在 V49 1.2B 推理循环中,把 token_emb(char_id) 替换成连续函数
sinusoidal_char_emb(char_id)。这是路径 A 的 falsifier。

5 个条件 (相同 val_ids 切片, 相同 20 sequences, 相同 seed):
  [1] TF baseline          — 模型真实能力
  [2] argmax               — 离散 baseline (exp29 复现: 64.7)
  [3] soft (V50)           — probs ⊤ E (exp29 复现: 33.3)
  [4] sin_char (Path A)    — 连续输入 (新)
  [5] sin_char + soft      — 路径 A + V50 组合 (新)

Verdict 表:
  [4] PPL > 1000   → Path A 推理侧无用, 必须重训
  [4] PPL ≈ argmax → 连续输入是中性扰动
  [4] PPL < argmax → 重大发现 (V51 素材)
  [5] PPL < [3]    → sin + soft 协同增益
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


# ---------------------------------------------------------------------------
# 加载 (复用 exp29)
# ---------------------------------------------------------------------------
def load_model():
    print(f"[load] {CHECKPOINT.name}")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    args = ckpt['args']
    print(f"[load] args: d_model={args['d_model']}, n_layers={args['n_layers']}, "
          f"n_heads={args['n_heads']}, d_ff={args['d_ff']}")
    print(f"[load] val_ppl: {ckpt['val_ppl']:.4f}")
    model = Transformer50M(
        vocab_size=2261,
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
    import pandas as pd
    p = PROJECT_ROOT / "crystalllm" / "data" / "processed" / "v28_val.parquet"
    df = pd.read_parquet(p)
    text = "\n".join(df["text"].astype(str).tolist())
    return text


@torch.no_grad()
def encode_chars(text, vocab):
    return [vocab.get(c, 0) for c in text]


# ---------------------------------------------------------------------------
# Eval (复用 exp29, 加种子固定让 5 个条件可比)
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_teacher_forcing(model, val_ids, num_seqs=20, seq_len=128):
    model.eval()
    n = len(val_ids) - seq_len - 1
    if n <= 0:
        return None
    g = torch.Generator(device='cpu').manual_seed(42)  # 固定种子
    starts = torch.randint(0, n, (num_seqs,), generator=g)
    total_loss, total_tokens = 0.0, 0
    for s in starts:
        ids = val_ids[s:s+seq_len+1].to(DEVICE).unsqueeze(0)
        logits = model(ids[:, :-1])
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               ids[:, 1:].view(-1), reduction='sum')
        total_loss += loss.item()
        total_tokens += seq_len
    return math.exp(total_loss / total_tokens)


# ---------------------------------------------------------------------------
# Feedback functions (新增 2 个)
# ---------------------------------------------------------------------------
@torch.no_grad()
def feedback_argmax(logits, model):
    return model.token_emb(logits.argmax(dim=-1))


@torch.no_grad()
def feedback_soft(logits, model):
    probs = F.softmax(logits.float(), dim=-1)
    return torch.matmul(probs, model.token_emb.weight)


# 路径 A 核心: 连续函数 char_id → R^d
# 使用 sinusoidal encoding, 类似 positional encoding
# char_id ∈ [0, 2260], d_model = 1536 (V49 1.2B)
@torch.no_grad()
def sinusoidal_char_emb(char_ids, d_model, vocab_size=2261):
    """char_id (int tensor, any shape) → (..., d_model) 连续嵌入.
    ponytail: 固定不训练, 用 8 个倍频程 log-spaced 频率."""
    pos = char_ids.float().unsqueeze(-1) / vocab_size  # normalize to [0,1]
    half = d_model // 2
    freqs = torch.exp(torch.linspace(0, 8, half, device=char_ids.device))  # log-spaced
    angles = pos * freqs.unsqueeze(0)  # broadcast
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


@torch.no_grad()
def feedback_sin_char(logits, model):
    """路径 A: 离散 argmax + 连续 sin 嵌入."""
    char_id = logits.argmax(dim=-1)
    d_model = model.token_emb.weight.shape[1]
    return sinusoidal_char_emb(char_id, d_model)


@torch.no_grad()
def feedback_sin_char_soft(logits, model):
    """路径 A + V50: 软概率 → sin 嵌入空间凸组合."""
    probs = F.softmax(logits.float(), dim=-1)  # (B, V)
    # 期望 sin 嵌入 = Σ_v p_v · sin_char_emb(v)
    vocab_size = probs.size(-1)
    d_model = model.token_emb.weight.shape[1]
    # 构造所有 v 的 sin 嵌入, 然后加权求和
    all_ids = torch.arange(vocab_size, device=probs.device)  # (V,)
    sin_table = sinusoidal_char_emb(all_ids, d_model)  # (V, d)
    # (B, V) @ (V, d) = (B, d)
    return torch.matmul(probs, sin_table)


# ---------------------------------------------------------------------------
# Autoregressive eval (复用 exp29)
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_autoregressive(model, val_ids, feedback_fn, num_seqs=20,
                       seq_len=64, name=""):
    model.eval()
    n = len(val_ids) - seq_len - 1
    if n <= 0:
        return None
    g = torch.Generator(device='cpu').manual_seed(42)  # 固定种子
    starts = torch.randint(0, n, (num_seqs,), generator=g)
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
                h_attn, _ = layer.attn(h_ln, h_ln, h_ln,
                                       attn_mask=mask, need_weights=False)
                h = h + h_attn
                h = h + layer.ffn(layer.ln2(h))
            logits = model.head(model.ln_f(h[:, -1:, :]))
            target = ids[:, t+1]
            total_loss += F.cross_entropy(
                logits.view(-1, logits.size(-1)), target, reduction='sum'
            ).item()
            total_tokens += 1
            fb = feedback_fn(logits[:, -1, :], model)
            if fb.dim() == 2:
                fb = fb.unsqueeze(1)
            cur = torch.cat([cur[:, 1:], fb], dim=1)
        if (seq_i + 1) % 5 == 0:
            print(f"    [{name}] seq {seq_i+1}/{num_seqs}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
    return math.exp(total_loss / total_tokens)


# ---------------------------------------------------------------------------
# Main: 跑 5 个条件
# ---------------------------------------------------------------------------
def main():
    print(f"[init] device={DEVICE}")
    model, args = load_model()

    print("\n[init] loading val data ...")
    text = load_val_data()
    print(f"[init] val text len: {len(text):,} chars")
    with open(PROJECT_ROOT / "crystalllm/data/processed/char_vocab.json",
              encoding="utf-8") as f:
        vocab_meta = json.load(f)
    vocab = vocab_meta["stoi"] if isinstance(vocab_meta, dict) and "stoi" in vocab_meta else vocab_meta
    val_ids = torch.tensor(encode_chars(text, vocab), dtype=torch.long)
    val_ids = val_ids[:50000]  # 同 exp29
    print(f"[init] val tokens: {len(val_ids):,}")

    d_model = model.token_emb.weight.shape[1]
    vocab_size = model.token_emb.weight.shape[0]
    results = {"model": "v49_1.2B", "d_model": d_model,
               "vocab_size": vocab_size, "conditions": {}}

    # [1] TF
    print("\n[1/5] Teacher-forcing PPL")
    t0 = time.time()
    tf_ppl = eval_teacher_forcing(model, val_ids, num_seqs=20, seq_len=128)
    print(f"  TF PPL: {tf_ppl:.4f}  (time={time.time()-t0:.0f}s)")
    results["conditions"]["tf"] = {"ppl": tf_ppl}

    # [2] argmax
    print("\n[2/5] Autoregressive ARGMAX (baseline)")
    t0 = time.time()
    ppl_arg = eval_autoregressive(model, val_ids, feedback_argmax,
                                  num_seqs=20, seq_len=64, name="argmax")
    print(f"  Argmax PPL: {ppl_arg:.4f}  (time={time.time()-t0:.0f}s)")
    results["conditions"]["argmax"] = {"ppl": ppl_arg}

    # [3] soft (V50)
    print("\n[3/5] Autoregressive SOFT-EXP (V50)")
    t0 = time.time()
    ppl_soft = eval_autoregressive(model, val_ids, feedback_soft,
                                   num_seqs=20, seq_len=64, name="soft")
    print(f"  Soft PPL: {ppl_soft:.4f}  (time={time.time()-t0:.0f}s)")
    results["conditions"]["soft_v50"] = {"ppl": ppl_soft}

    # [4] sin_char (Path A 核心 falsifier)
    print("\n[4/5] Autoregressive SIN_CHAR (Path A — sinusoidal input)")
    t0 = time.time()
    ppl_sin = eval_autoregressive(model, val_ids, feedback_sin_char,
                                  num_seqs=20, seq_len=64, name="sin_char")
    print(f"  Sin-char PPL: {ppl_sin:.4f}  (time={time.time()-t0:.0f}s)")
    results["conditions"]["sin_char"] = {"ppl": ppl_sin}

    # [5] sin_char + soft (组合)
    print("\n[5/5] Autoregressive SIN_CHAR + SOFT (Path A + V50)")
    t0 = time.time()
    ppl_sin_soft = eval_autoregressive(model, val_ids, feedback_sin_char_soft,
                                       num_seqs=20, seq_len=64, name="sin_soft")
    print(f"  Sin+Soft PPL: {ppl_sin_soft:.4f}  (time={time.time()-t0:.0f}s)")
    results["conditions"]["sin_char_soft"] = {"ppl": ppl_sin_soft}

    # verdict
    soft_adv_v50 = (ppl_arg - ppl_soft) / ppl_arg * 100
    sin_vs_arg = (ppl_arg - ppl_sin) / ppl_arg * 100
    sinsoft_vs_soft = (ppl_soft - ppl_sin_soft) / ppl_soft * 100

    print("\n" + "="*70)
    print("VERDICT TABLE (Path A falsifier)")
    print("="*70)
    print(f"  [1] TF baseline:        PPL={tf_ppl:.4f}")
    print(f"  [2] argmax:             PPL={ppl_arg:.4f}  ({ppl_arg/tf_ppl:.0f}x exposure bias)")
    print(f"  [3] soft (V50):         PPL={ppl_soft:.4f}  ({ppl_soft/tf_ppl:.0f}x exposure bias)  V50 gain={soft_adv_v50:+.2f}%")
    print(f"  [4] sin_char (Path A):  PPL={ppl_sin:.4f}                                     Path A gain={sin_vs_arg:+.2f}%")
    print(f"  [5] sin+soft:           PPL={ppl_sin_soft:.4f}                            Combo gain={sinsoft_vs_soft:+.2f}% over V50")

    print("\n--- VERDICT ---")
    if ppl_sin > 1000:
        print("  ❌ Path A FALSIFIED: 软连续输入在 V49 1.2B 推理侧不可用 (PPL>1000)")
        print("     → 连续输入必须重训侧 accommodate, 不能纯推理侧一刀")
    elif sin_vs_arg > 5:
        print(f"  ✓ Path A CONFIRMED: 连续 sin 输入相对 argmax 增益 {sin_vs_arg:+.2f}%")
        print("     → 这是 V51 论文素材, 路径 A 可与 V50 组合")
    else:
        print(f"  ~ Path A NEUTRAL: 软连续输入增益 {sin_vs_arg:+.2f}% (不显著)")
        print("     → 软输入是中性扰动, 单独使用不构成改进")

    if sinsoft_vs_soft > 5:
        print(f"  ✓✓ Path A + V50 SYNERGY: 组合再降 {sinsoft_vs_soft:+.2f}%")
        print("       → V51 = V50 + 路径 A, 双重软反馈")

    results.update({
        "tf_ppl": tf_ppl,
        "argmax_ppl": ppl_arg,
        "soft_ppl": ppl_soft,
        "sin_char_ppl": ppl_sin,
        "sin_char_soft_ppl": ppl_sin_soft,
        "v50_advantage_pct": soft_adv_v50,
        "path_a_advantage_pct": sin_vs_arg,
        "synergy_pct_over_v50": sinsoft_vs_soft,
    })

    out_path = PROJECT_ROOT / "experiments" / "v49_pre" / "exp33_path_a_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[save] -> {out_path}")


if __name__ == "__main__":
    main()