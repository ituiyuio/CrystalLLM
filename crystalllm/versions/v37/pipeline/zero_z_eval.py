# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""zero_z_eval.py — v37 zero-z ablation 统一评估脚本

测量 decoder 在 z 强制为零向量时的 PPL 退化. 复用 v25/v36 checkpoint,
接受 --checkpoint {v25, v36} --z_mode {encoded, zero}.

zero 模式下, 在 decoder forward 入口把 z 替换为 torch.zeros(B, D_Z).
其他信号 (pos embed, KV cache, head) 不动.

预期:
  - v25 encoded: PPL 2.47 (已有)
  - v25 zero:    PPL 接近 2.49-2.55 (训练日志估算)
  - v36 encoded: PPL 2.81 (已有)
  - v36 zero:    与 v25 zero 对比, 验证 cross-attn 是否真用 z
"""
import argparse
import json
import sys
import io
import os
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); np.random.seed(42); random.seed(42)

V37_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = V37_DIR.parent.parent
DATA = PROJECT_ROOT / "data" / "processed"

# ============================================================
# v25 Decoder (BAD-DP, z as pos 0 token)
# ============================================================
class BlockCausal(nn.Module):
    def __init__(s, N_EMBD, N_HEAD):
        super().__init__()
        s.nh = N_HEAD; s.head_dim = N_EMBD // N_HEAD
        s.ln1 = nn.LayerNorm(N_EMBD); s.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        s.proj = nn.Linear(N_EMBD, N_EMBD)
        s.ln2 = nn.LayerNorm(N_EMBD)
        s.mlp = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(), nn.Linear(4 * N_EMBD, N_EMBD))
    def forward(s, x):
        B_, T_, C = x.shape
        h = s.ln1(x); qkv = s.qkv(h).reshape(B_, T_, 3, s.nh, s.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(y.transpose(1, 2).contiguous().view(B_, T_, C))
        x = x + s.mlp(s.ln2(x)); return x


class DecoderV25(nn.Module):
    """v25 BAD-DP: z as pos 0 single token."""
    def __init__(s, V, T, D_Z, DEC_LAYER, DEC_HEAD, DEC_EMBD, BOS_ID):
        super().__init__()
        s.T, s.BOS_ID = T, BOS_ID
        s.z_to_emb = nn.Linear(D_Z, DEC_EMBD)
        s.tok = nn.Embedding(V, DEC_EMBD)
        s.pos = nn.Embedding(T + 2, DEC_EMBD)
        s.blocks = nn.ModuleList([BlockCausal(DEC_EMBD, DEC_HEAD) for _ in range(DEC_LAYER)])
        s.ln_f = nn.LayerNorm(DEC_EMBD)
        s.head = nn.Linear(DEC_EMBD, V, bias=False)
        s.tok.weight = s.head.weight
    def forward(s, z, x):
        B_, T_ = x.shape
        z_emb = s.z_to_emb(z).unsqueeze(1)
        bos_emb = s.tok(torch.tensor([s.BOS_ID], device=x.device)).expand(B_, 1, -1)
        x_emb = s.tok(x)
        inp = torch.cat([z_emb, bos_emb, x_emb], dim=1)
        inp = inp + s.pos(torch.arange(T_ + 2, device=x.device))
        for b in s.blocks: inp = b(inp)
        logits = s.head(s.ln_f(inp))
        return logits[:, 1:T_ + 1]


# ============================================================
# 加载 checkpoint
# ============================================================
def load_decoder(checkpoint_name: str, device="cuda"):
    """返回 (decoder, V, D_Z, T)"""
    if checkpoint_name == "v25":
        ckpt = torch.load(V37_DIR / "v25_decoder.pt", map_location=device, weights_only=False)
        cfg = ckpt["config"]
        D_Z = cfg["D_Z"]; T = cfg["T"]
        DEC_LAYER = cfg["DEC_LAYER"]; DEC_HEAD = cfg["DEC_HEAD"]; DEC_EMBD = cfg["DEC_EMBD"]
        vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
        V = vocab["vocab_size"]; BOS_ID = vocab["stoi"].get("<bos>", 1)
        decoder = DecoderV25(V, T, D_Z, DEC_LAYER, DEC_HEAD, DEC_EMBD, BOS_ID).to(device)
        decoder.load_state_dict(ckpt["decoder"])
        return decoder, V, D_Z, T
    elif checkpoint_name == "v36":
        sys.path.insert(0, str(V37_DIR.parent / "v36" / "training"))
        from v36_model import DecoderCrossAttn
        ckpt = torch.load(V37_DIR / "v36_decoder.pt", map_location=device, weights_only=False)
        cfg = ckpt["config"]
        D_Z = cfg["D_Z"]; T = cfg["T"]
        decoder = DecoderCrossAttn(
            V=cfg["V"], T=T, DEC_LAYER=cfg["DEC_LAYER"], DEC_HEAD=cfg["DEC_HEAD"],
            DEC_EMBD=cfg["DEC_EMBD"], D_Z=D_Z, BOS_ID=cfg["BOS_ID"]
        ).to(device)
        decoder.load_state_dict(ckpt["decoder"])
        return decoder, cfg["V"], D_Z, T
    else:
        raise ValueError(f"Unknown checkpoint: {checkpoint_name}")


# ============================================================
# 数据加载
# ============================================================
def load_val_data(device="cuda"):
    df_val = pd.read_parquet(DATA / "v24_val.parquet")
    val_texts = df_val["text"].tolist()
    cache = np.load(DATA / "cached_v24_z.npz")
    val_z = torch.tensor(cache["val_z"], dtype=torch.float32, device=device)
    vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
    stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
    V = vocab["vocab_size"]
    return val_texts, val_z, stoi, itos, V


def get_val_batches(val_texts, stoi, T, B=4):
    batches = []
    for i in range(0, len(val_texts), B):
        batch = val_texts[i:i + B]
        chunks = []
        for text in batch:
            if len(text) < T: text = text + "\n" * (T - len(text))
            start = random.randint(0, max(0, len(text) - T))
            chunk = text[start:start + T]
            chunks.append([stoi.get(c, 0) for c in chunk])
        batches.append((torch.tensor(chunks, dtype=torch.long, device="cuda"), i))
    return batches


# ============================================================
# 主评估
# ============================================================
@torch.no_grad()
def eval_ppl(decoder, val_batches, val_z, D_Z, V, z_mode):
    """z_mode: 'encoded' or 'zero'"""
    total_loss = 0.0; n = 0
    for x, i in val_batches:
        B = x.size(0)
        if z_mode == "encoded":
            z = val_z[i:i + B]
        elif z_mode == "zero":
            z = torch.zeros(B, D_Z, device=x.device)
        else:
            raise ValueError(f"Unknown z_mode: {z_mode}")
        logits = decoder(z, x)
        loss = F.cross_entropy(logits.reshape(-1, V), x.reshape(-1), reduction='sum')
        total_loss += loss.item(); n += x.numel()
    avg_loss = total_loss / n
    ppl = float(np.exp(avg_loss))
    return ppl, avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=["v25", "v36"], required=True)
    parser.add_argument("--z_mode", choices=["encoded", "zero"], required=True)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    print(f"=== v37 zero-z ablation ===")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  z_mode:     {args.z_mode}")

    decoder, V, D_Z, T = load_decoder(args.checkpoint, device="cuda")
    decoder.eval()
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"  decoder params: {n_params/1e6:.2f}M, D_Z={D_Z}, T={T}")

    val_texts, val_z, stoi, itos, V_check = load_val_data()
    assert V_check == V, f"vocab mismatch: ckpt V={V}, data V={V_check}"
    val_batches = get_val_batches(val_texts, stoi, T, B=4)
    print(f"  val_batches: {len(val_batches)} (B=4, T={T})")

    ppl, avg_loss = eval_ppl(decoder, val_batches, val_z, D_Z, V, args.z_mode)
    print(f"\n  [{args.checkpoint} + {args.z_mode}] PPL = {ppl:.4f} (avg_loss {avg_loss:.4f})")

    if args.output_json:
        result = {"checkpoint": args.checkpoint, "z_mode": args.z_mode,
                  "PPL": ppl, "avg_loss": avg_loss,
                  "decoder_params_M": n_params / 1e6, "D_Z": D_Z, "T": T,
                  "n_val_samples": len(val_texts), "n_batches": len(val_batches)}
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  saved: {args.output_json}")


if __name__ == "__main__":
    main()