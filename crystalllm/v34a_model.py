"""
v34a_model.py — Shared Backbone AR × 扩散联合模型

架构:
  - SharedBackbone: 12 层 × 1280 hidden × 20 heads Causal Transformer
  - ARHead: tied Linear (1280 → vocab), CE loss
  - DHead: MLP (1280 → 1280 → 8*1280), CFM loss

支持两种 forward 路径:
  - forward(tokens, z, t=None): 标准 token 输入
  - forward_emb(emb, z, t=None): 预计算 embedding 输入 (用于扩散 ODE)
"""
import torch, torch.nn as nn, torch.nn.functional as F
import math


def get_alpha(t):
    """CFM alpha schedule: alpha(t) = t"""
    return t


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
    """12 层 Causal Transformer with z + t conditioning"""

    def __init__(self, vocab_size, n_layer=12, n_embd=1280, n_head=20,
                 z_dim=256, t_dim=256, max_seq=160):
        super().__init__()
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(max_seq, n_embd)
        self.z_proj = nn.Linear(z_dim, n_embd)
        self.t_proj = nn.Linear(t_dim, n_embd)
        self.blocks = nn.ModuleList([CausalBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)

    def _common(self, x, z, t):
        """Apply position, z-bias, and (optional) t-bias."""
        B, T, C = x.shape
        x = x + self.pos_emb(torch.arange(T, device=x.device))
        x = x + self.z_proj(z).unsqueeze(1)
        if t is not None:
            t_emb = get_time_embedding(t, dim=256)
            x = x + self.t_proj(t_emb).unsqueeze(1)
        return x

    def forward(self, tokens, z, t=None):
        """
        tokens: (B, T) token ids
        z: (B, z_dim) global encoding
        t: (B,) diffusion timesteps or None for AR-only
        """
        x = self.tok_emb(tokens)
        x = self._common(x, z, t)
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)

    def forward_emb(self, emb, z, t=None):
        """
        Forward from pre-computed embeddings (用于扩散路径)

        emb: (B, T, n_embd) pre-computed token/noisy embeddings
        z: (B, z_dim)
        t: (B,) or None
        """
        x = self._common(emb, z, t)
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)


class ARHead(nn.Module):
    """Tied linear head: weight = tok_emb.weight"""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, hidden):
        return F.linear(hidden, self.backbone.tok_emb.weight)


class DHead(nn.Module):
    """Diffusion head: predicts velocity for K=8 tokens at a time"""

    def __init__(self, n_embd=1280, k_window=8):
        super().__init__()
        self.k_window = k_window
        self.fc1 = nn.Linear(n_embd, n_embd)
        self.fc2 = nn.Linear(n_embd, n_embd)
        self.out = nn.Linear(n_embd, k_window * n_embd)

    def forward(self, hidden):
        """
        hidden: (B, T, n_embd)
        returns: velocity (B, T, K, n_embd) — typically only the diagonal K=0 used
        """
        B, T, C = hidden.shape
        h = F.gelu(self.fc1(hidden))
        h = F.gelu(self.fc2(h))
        v = self.out(h)  # (B, T, K * C)
        return v.view(B, T, self.k_window, C)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    import json
    vocab = json.load(open("data/processed/char_vocab.json", encoding="utf-8"))
    V = vocab["vocab_size"]

    backbone = SharedBackbone(V)
    ar = ARHead(backbone)
    dh = DHead()

    # AR path sanity
    tokens = torch.randint(0, V, (2, 32))
    z = torch.randn(2, 256)
    t = torch.rand(2)
    h = backbone(tokens, z, t)
    print(f"backbone hidden (AR+t): {h.shape}")
    logits = ar(h)
    print(f"AR logits: {logits.shape}")

    # Diffusion path sanity
    noisy_emb = torch.randn(2, 8, 1280)
    h_diff = backbone.forward_emb(noisy_emb, z, t)
    print(f"backbone hidden (diff): {h_diff.shape}")
    v = dh(h_diff)
    print(f"D velocity: {v.shape}")
    print(f"D velocity (K=0 diagonal): {v[:, :, 0, :].shape}")

    print(f"backbone params: {count_params(backbone)/1e6:.1f}M")
    print(f"D head params: {count_params(dh)/1e6:.1f}M")
    print(f"total (excluding AR tied): {(count_params(backbone) + count_params(dh))/1e6:.1f}M")