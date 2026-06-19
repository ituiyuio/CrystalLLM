"""v36_model.py — v36 cross-attention decoder 模型定义

v36 = v25 架构 + 每 block 加 cross-attn(z) 子层
z 不再 prepended 到序列; z 作为 K/V 传给每个 block
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockCrossAttn(nn.Module):
    """self-attn + cross-attn(z as K/V) + mlp"""
    def __init__(s, N_EMBD, N_HEAD, D_Z):
        super().__init__()
        s.nh = N_HEAD
        s.head_dim = N_EMBD // N_HEAD
        # Self-attention (warm-start from v25)
        s.ln1 = nn.LayerNorm(N_EMBD)
        s.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        s.proj = nn.Linear(N_EMBD, N_EMBD)
        # Cross-attention (NEW, random init)
        s.ln_cross = nn.LayerNorm(N_EMBD)
        s.q_cross = nn.Linear(N_EMBD, N_EMBD)
        s.k_cross = nn.Linear(D_Z, N_EMBD)
        s.v_cross = nn.Linear(D_Z, N_EMBD)
        s.proj_cross = nn.Linear(N_EMBD, N_EMBD)
        # MLP (warm-start from v25)
        s.ln2 = nn.LayerNorm(N_EMBD)
        s.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
        )

    def forward(s, x, z_kv):
        B, T, C = x.shape
        # Self-attention (existing, unchanged from v25)
        h = s.ln1(x)
        qkv = s.qkv(h).reshape(B, T, 3, s.nh, s.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(attn.transpose(1, 2).contiguous().view(B, T, C))
        # Cross-attention (NEW)
        h_c = s.ln_cross(x)
        q_c = s.q_cross(h_c).reshape(B, T, s.nh, s.head_dim).permute(0, 2, 1, 3)
        # z_kv: (B, D_Z) → (B, 1, N_EMBD) → (B, 1, nh, head_dim)
        k_c = s.k_cross(z_kv).reshape(B, 1, s.nh, s.head_dim).permute(0, 2, 1, 3)
        v_c = s.v_cross(z_kv).reshape(B, 1, s.nh, s.head_dim).permute(0, 2, 1, 3)
        # 无 causal mask, full attn to z
        attn_c = F.scaled_dot_product_attention(q_c, k_c, v_c)
        x = x + s.proj_cross(attn_c.transpose(1, 2).contiguous().view(B, T, C))
        # MLP (existing)
        x = x + s.mlp(s.ln2(x))
        return x


class DecoderCrossAttn(nn.Module):
    """v36 decoder: BOS + tokens, 每 block cross-attn(z as K/V)"""
    def __init__(s, V, T, DEC_LAYER, DEC_HEAD, DEC_EMBD, D_Z, BOS_ID):
        super().__init__()
        s.V = V; s.T = T; s.BOS_ID = BOS_ID
        s.d_z = D_Z
        # 不再有 z_to_emb; z 通过 k_cross/v_cross 直接注入
        s.tok = nn.Embedding(V, DEC_EMBD)
        s.pos = nn.Embedding(T + 2, DEC_EMBD)  # T+2 保持与 v25 形状一致
        s.blocks = nn.ModuleList([
            BlockCrossAttn(DEC_EMBD, DEC_HEAD, D_Z) for _ in range(DEC_LAYER)
        ])
        s.ln_f = nn.LayerNorm(DEC_EMBD)
        s.head = nn.Linear(DEC_EMBD, V, bias=False)
        s.tok.weight = s.head.weight  # tied weights

    def forward(s, z, x):
        B, T = x.shape
        bos = s.tok(torch.tensor([s.BOS_ID], device=x.device)).expand(B, 1, -1)
        inp = torch.cat([bos, s.tok(x)], dim=1)        # (B, T+1, D) 不 prepend z
        inp = inp + s.pos(torch.arange(T + 1, device=x.device))
        for b in s.blocks:
            inp = b(inp, z)                             # z 传给每个 block
        return s.head(s.ln_f(inp))                      # (B, T+1, V)