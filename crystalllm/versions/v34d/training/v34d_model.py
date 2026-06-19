# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
v34d_model.py — Discrete Mask Diffusion for Token-space 融合

核心思想 (用户洞察: "共享 token 一定是有潜力"):
  - DHead 和 ARHead **都输出 token logits** (同一 vocab 空间)
  - ARHead 学 P(next | prefix) — 局部续写
  - DHead 学 P(clean | noisy, z, t) — 全局窗口还原 (D3PM mask diffusion)
  - 输出空间真共享 → latent 兼容

D3PM (Simplified Mask Diffusion):
  - t=0: 全是 ground truth tokens
  - t=1: 全是 [MASK] tokens
  - 加噪: 随机选择位置 mask 掉
  - 去噪: 训练 DHead 还原被 mask 的 token
  - 推理: K 步去噪, 从全 MASK 还原到 K tokens
"""
import torch, torch.nn as nn, torch.nn.functional as F
import math


def get_time_embedding(t, dim=256):
    """Sinusoidal time embedding"""
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, device=t.device, dtype=torch.float32) *
                      (math.log(10000.0) / half))
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1) * (dim ** 0.5)


class CausalBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.nh = n_head
        self.head_dim = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.nh, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).contiguous().view(B, T, C))
        x = x + self.mlp(self.ln2(x))
        return x


class SharedBackbone(nn.Module):
    """12L Causal Transformer with z + t conditioning"""

    def __init__(self, vocab_size, n_layer=12, n_embd=1280, n_head=20,
                 z_dim=256, t_dim=256, max_seq=160):
        super().__init__()
        self.n_embd = n_embd
        self.n_layer = n_layer
        # vocab_size + 1 for [MASK]
        self.tok_emb = nn.Embedding(vocab_size + 1, n_embd)
        self.pos_emb = nn.Embedding(max_seq, n_embd)
        self.z_proj = nn.Linear(z_dim, n_embd)
        self.t_proj = nn.Linear(t_dim, n_embd)
        self.blocks = nn.ModuleList([CausalBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)

    def _common(self, x, z, t):
        B, T, C = x.shape
        x = x + self.pos_emb(torch.arange(T, device=x.device))
        x = x + self.z_proj(z).unsqueeze(1)
        if t is not None:
            t_emb = get_time_embedding(t, dim=256)
            x = x + self.t_proj(t_emb).unsqueeze(1)
        return x

    def forward(self, tokens, z, t=None):
        """tokens (B, T) 含真实 token 或 [MASK], z (B, z_dim), t (B,)"""
        x = self.tok_emb(tokens)
        x = self._common(x, z, t)
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)

    def forward_emb(self, emb, z, t=None):
        """预计算 embedding (用于连续空间实验)"""
        x = self._common(emb, z, t)
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)


class ARHead(nn.Module):
    """Tied linear head: weight = tok_emb.weight[:-1] (excluding MASK)"""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, hidden):
        # 不预测 MASK 位置 (vocab 维度 = V, 不含 MASK)
        return F.linear(hidden, self.backbone.tok_emb.weight[:-1])


class DHead(nn.Module):
    """Discrete diffusion head: predicts token logits over full vocab (含 MASK)

    关键: DHead 输出 (B, T, V+1) 含 MASK 维度, 完整 vocab 空间
    与 ARHead (B, T, V) 不含 MASK 形成对比 — 但**都是 token logits**
    """

    def __init__(self, backbone, n_embd=1280):
        super().__init__()
        self.backbone = backbone
        # 独立 head, 不 tied, 输出 V+1 维 (含 MASK)
        self.proj = nn.Linear(n_embd, backbone.tok_emb.num_embeddings)

    def forward(self, hidden):
        return self.proj(hidden)  # (B, T, V+1)


def mask_schedule(t):
    """线性 mask 调度: t=0 → 0% mask, t=1 → 100% mask"""
    return t  # 简化: mask 概率 = t


def add_mask_noise(tokens, t, mask_id):
    """按 t 概率 mask 掉 token

    tokens: (B, T) 真实 token ids
    t: (B,) 0-1
    mask_id: MASK token id
    返回: (noisy_tokens, mask_positions)
    """
    B, T = tokens.shape
    # 每个位置独立决定是否 mask
    mask_prob = t[:, None].expand(B, T)  # (B, T)
    rand = torch.rand(B, T, device=tokens.device)
    mask_positions = rand < mask_prob  # True = 需要 mask
    noisy = tokens.clone()
    noisy[mask_positions] = mask_id
    return noisy, mask_positions


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    import json
    vocab = json.load(open("data/processed/char_vocab.json", encoding="utf-8"))
    V = vocab["vocab_size"]
    MASK_ID = V  # MASK 是 vocab 最后一个

    backbone = SharedBackbone(V)
    ar = ARHead(backbone)
    dh = DHead(backbone)

    # 测试: 真实 token 路径
    tokens = torch.randint(0, V, (2, 32))
    z = torch.randn(2, 256)
    h = backbone(tokens, z, t=None)
    ar_logits = ar(h)
    print(f"AR logits: {ar_logits.shape} (expect 2, 32, {V})")
    d_logits = dh(h)
    print(f"D logits: {d_logits.shape} (expect 2, 32, {V+1})")

    # 测试: MASK 路径
    t = torch.tensor([0.5, 0.5])
    noisy, mask_pos = add_mask_noise(tokens, t, MASK_ID)
    print(f"Mask positions: {mask_pos.sum().item()}/{mask_pos.numel()}")
    h_noisy = backbone(noisy, z, t=t)
    d_logits_noisy = dh(h_noisy)
    print(f"D logits (noisy): {d_logits_noisy.shape}")

    print(f"Backbone: {count_params(backbone)/1e6:.1f}M")
    print(f"D head: {count_params(dh)/1e6:.1f}M")
    print(f"Total: {(count_params(backbone) + count_params(dh))/1e6:.1f}M")