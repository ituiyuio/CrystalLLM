"""model.py — v47 Phase 1 200M 模型

架构 (~205M active params):
  Hidden dim: 1024
  Layers:     16
  Heads:      16 (head_dim 64)
  Dense FFN:  1024 -> 4096 -> 1024    (variant A)
  MoE FFN:    8 experts x (1024->2048->1024), Top-2 routing (variants B, C)
  Sparse attn (variant C only):
    - z_emb / BOS positions: 全局可见 (attend to all causal positions)
    - x positions in block k: attend to:
        * z_emb / BOS positions in blocks 0..k
        * x in same block k
        * x in adjacent block (k-1)
    - 因果性: position i 只 attend to j <= i
  Per-block z injection (variant C only):
    Block 0:   [z_emb, BOS, x_emb_0..x_emb_{B-1}]      -> B+2 positions
    Block k>0: [z_emb + pos_block_emb[k], x_emb_{Bk}..] -> B+1 positions
  Total positions (variant C): 18 + 31*17 = 545

三个变体:
  A (baseline):   dense FFN + dense attn + AR only
  B (MoE):        MoE FFN + dense attn + AR only
  C (full):       MoE FFN + sparse attn + per-block z + 0.5 L_AR + 0.5 L_diff
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 配置常量 (与 spec.md 对齐)
# ============================================================
DEC_EMBD = 1024
DEC_LAYER = 16
DEC_HEAD = 16
HEAD_DIM = DEC_EMBD // DEC_HEAD  # 64
FFN_DIM_DENSE = 4096
FFN_DIM_EXPERT = 2048
N_EXPERTS = 8
TOP_K = 2

# Block structure (variant C)
BLOCK_SIZE = 16
T_TOKEN = 512
N_BLOCKS = T_TOKEN // BLOCK_SIZE  # 32
# Variant C total positions: 18 + 31*17 = 545
V47_BLOCK_POS_LEN = (BLOCK_SIZE + 2) + (N_BLOCKS - 1) * (BLOCK_SIZE + 1)
# Variant A/B total positions: T + 2 = 514
V47_FLAT_POS_LEN = T_TOKEN + 2

# Sparse attention: window (邻块数)
SPARSE_WINDOW = 1  # ±1 邻块


# ============================================================
# Sparse Causal Attention (variant C)
# ============================================================
def build_sparse_mask(T_total: int, n_blocks: int, block_size: int,
                       window: int = SPARSE_WINDOW, device: str = "cuda") -> torch.Tensor:
    """Build sparse causal attention mask for variant C (per-block z input).

    Returns: (T_total, T_total) bool tensor, True = can attend.

    Position types:
      Block 0: 0=z, 1=BOS, 2..(B+1)=x (18 positions, type 0=z, 1=BOS, 2=x)
      Block k>0: 0=z, 1..B=x (17 positions, type 0=z, 2=x)

    Sparse rule (with causality j <= i):
      - If i is z or BOS (type 0/1): can attend to anything (global)
      - If i is x (type 2):
        - j is z or BOS: if block[j] <= block[i] (always true causal)
        - j is x: if |block[j] - block[i]| <= window
    """
    pos_block = torch.zeros(T_total, dtype=torch.long, device=device)
    pos_type = torch.zeros(T_total, dtype=torch.long, device=device)  # 0=z, 1=BOS, 2=x

    cur = 0
    for k in range(n_blocks):
        if k == 0:
            block_len = block_size + 2  # 18
            pos_block[cur] = k; pos_type[cur] = 0; cur += 1  # z
            pos_block[cur] = k; pos_type[cur] = 1; cur += 1  # BOS
            for _ in range(block_size):
                pos_block[cur] = k; pos_type[cur] = 2; cur += 1  # x
        else:
            block_len = block_size + 1  # 17
            pos_block[cur] = k; pos_type[cur] = 0; cur += 1  # z
            for _ in range(block_size):
                pos_block[cur] = k; pos_type[cur] = 2; cur += 1  # x

    i_block = pos_block.unsqueeze(1)  # (T, 1)
    j_block = pos_block.unsqueeze(0)  # (1, T)
    i_type = pos_type.unsqueeze(1)    # (T, 1)
    j_type = pos_type.unsqueeze(0)    # (1, T)

    i_is_x = (i_type == 2)  # (T, 1)
    j_is_z = (j_type == 0) | (j_type == 1)  # z or BOS

    # x-to-x adjacency
    x_to_x = (~j_is_z) & ((j_block - i_block).abs() <= window)  # (T, T)

    # x-to-z (any z/BOS with block <= i_block, causal handled below)
    x_to_z = j_is_z

    # z/BOS can attend to anything
    z_to_all = ~i_is_x

    # Combine
    causal = torch.tril(torch.ones(T_total, T_total, dtype=torch.bool, device=device))
    sparse = z_to_all | (i_is_x & (x_to_x | x_to_z))
    mask = causal & sparse
    return mask


class SparseAttention(nn.Module):
    """Sparse causal multi-head attention.

    For variant C: uses precomputed sparse mask (per-block z + sliding window).
    For variants A/B: uses standard dense causal attention (mask=None).
    """

    def __init__(self, n_embd: int = DEC_EMBD, n_head: int = DEC_HEAD,
                 use_sparse: bool = False, max_T: int = V47_BLOCK_POS_LEN):
        super().__init__()
        self.nh = n_head
        self.head_dim = n_embd // n_head
        self.use_sparse = use_sparse
        self.max_T = max_T
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        # Precompute sparse mask if needed
        if use_sparse:
            mask = build_sparse_mask(max_T, N_BLOCKS, BLOCK_SIZE,
                                      window=SPARSE_WINDOW, device="cpu")
            # attn_mask in PyTorch: True means MASK (don't attend)
            self.register_buffer('sparse_mask', ~mask)  # invert: True=mask
        else:
            self.sparse_mask = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, T_, C = x.shape
        qkv = self.qkv(x).reshape(B_, T_, 3, self.nh, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.use_sparse and T_ == self.sparse_mask.shape[0]:
            # Use sparse mask (True = mask out)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=self.sparse_mask)
        else:
            # Dense causal attention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        x = self.proj(y.transpose(1, 2).contiguous().view(B_, T_, C))
        return x

    def sparse_ratio(self) -> float:
        """Fraction of positions that are masked out (1 = full sparse, 0 = dense)."""
        if self.sparse_mask is None:
            return 0.0
        return float(self.sparse_mask.float().mean().item())


# ============================================================
# Transformer Block
# ============================================================
class BlockCausalV47(nn.Module):
    """v47 Transformer block: pre-norm + (sparse) attention + FFN."""

    def __init__(self, n_embd: int = DEC_EMBD, n_head: int = DEC_HEAD,
                 use_sparse_attn: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = SparseAttention(n_embd, n_head, use_sparse=use_sparse_attn)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = None  # set externally

    def set_ffn(self, ffn: nn.Module):
        self.ffn = ffn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ============================================================
# Dense FFN (variant A)
# ============================================================
class DenseFFNV47(nn.Module):
    def __init__(self, n_embd: int = DEC_EMBD, ffn_dim: int = FFN_DIM_DENSE):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, n_embd)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

    def aux_loss(self):
        return torch.tensor(0.0, device=next(self.parameters()).device)


# ============================================================
# MoE FFN (variants B, C)
# ============================================================
class MoEFFNV47(nn.Module):
    """Mixture-of-Experts FFN with Top-2 routing + load balance loss (Switch Transformer style)."""

    def __init__(self, n_embd: int = DEC_EMBD, ffn_dim: int = FFN_DIM_EXPERT,
                 n_experts: int = N_EXPERTS, top_k: int = TOP_K):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(n_embd, n_experts, bias=False)
        self.experts = nn.ModuleList([
            DenseFFNV47(n_embd, ffn_dim) for _ in range(n_experts)
        ])
        self.register_buffer('importance_acc', torch.zeros(n_experts))
        self.register_buffer('importance_count', torch.tensor(0.0))
        self.aux_loss_coef = 0.01

    def forward(self, x):
        B, T, D = x.shape
        flat_x = x.reshape(-1, D)

        router_logits = self.router(flat_x)
        router_probs = F.softmax(router_logits, dim=-1)

        topk_probs, topk_idx = router_probs.topk(self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        out = torch.zeros_like(flat_x)
        for e_idx in range(self.n_experts):
            expert_mask = (topk_idx == e_idx).any(dim=-1)
            if not expert_mask.any():
                continue
            token_ids = expert_mask.nonzero(as_tuple=True)[0]
            weight = torch.zeros(token_ids.shape[0], device=x.device, dtype=x.dtype)
            for slot in range(self.top_k):
                slot_mask = (topk_idx[token_ids, slot] == e_idx)
                weight = torch.where(slot_mask, topk_probs[token_ids, slot], weight)

            expert_out = self.experts[e_idx](flat_x[token_ids])
            out[token_ids] += expert_out * weight.unsqueeze(-1)

        # Store router_logits for differentiable aux loss
        self._last_router_logits = router_logits

        # Update importance stats (no grad)
        with torch.no_grad():
            importance = router_probs.mean(dim=0)
            self.importance_acc += importance.detach()
            self.importance_count += 1.0

        return out.reshape(B, T, D)

    def aux_loss(self, router_logits=None):
        if router_logits is None:
            router_logits = getattr(self, '_last_router_logits', None)
            if router_logits is None:
                if self.importance_count < 1:
                    return torch.tensor(0.0, device=next(self.parameters()).device)
                importance = self.importance_acc / self.importance_count.clamp(min=1.0)
                return self.aux_loss_coef * self.n_experts * (importance * importance).sum()

        router_probs = F.softmax(router_logits, dim=-1)
        with torch.no_grad():
            topk_idx = router_probs.topk(self.top_k, dim=-1)[1]
            f = torch.zeros(self.n_experts, device=router_logits.device)
            for e in range(self.n_experts):
                f[e] = (topk_idx == e).any(dim=-1).float().mean()
        P = router_probs.mean(dim=0)
        return self.aux_loss_coef * self.n_experts * (f * P).sum()

    def reset_stats(self):
        self.importance_acc.zero_()
        self.importance_count.zero_()

    def importance_variance(self) -> float:
        if self.importance_count < 1:
            return float('nan')
        importance = self.importance_acc / self.importance_count
        return float(importance.var().item())


# ============================================================
# V47 Decoder (3 variants)
# ============================================================
class V47Decoder(nn.Module):
    """Decoder for v47 Phase 1.

    Variants:
      A (dense, dense attn):    ffn_type="dense", use_per_block_z=False, use_sparse_attn=False
      B (MoE, dense attn):     ffn_type="moe",   use_per_block_z=False, use_sparse_attn=False
      C (MoE + per-block z + sparse attn):
                                ffn_type="moe",   use_per_block_z=True,  use_sparse_attn=True
    """

    def __init__(
        self,
        V: int,
        D_Z: int = 256,
        ffn_type: str = "dense",
        use_per_block_z: bool = False,
        use_sparse_attn: bool = False,
        bos_id: int = 1,
        mask_id: int = 0,
        n_layer: int = DEC_LAYER,
        n_head: int = DEC_HEAD,
        n_embd: int = DEC_EMBD,
        n_experts: int = N_EXPERTS,
        top_k: int = TOP_K,
        ffn_dim_dense: int = FFN_DIM_DENSE,
        ffn_dim_expert: int = FFN_DIM_EXPERT,
        n_blocks: int = N_BLOCKS,
        block_size: int = BLOCK_SIZE,
    ):
        super().__init__()
        self.V = V
        self.D_Z = D_Z
        self.ffn_type = ffn_type
        self.use_per_block_z = use_per_block_z
        self.use_sparse_attn = use_sparse_attn
        self.bos_id = bos_id
        self.mask_id = mask_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_blocks = n_blocks
        self.block_size = block_size

        self.total_pos = (
            V47_BLOCK_POS_LEN if use_per_block_z else V47_FLAT_POS_LEN
        )

        self.z_to_emb = nn.Linear(D_Z, n_embd)
        self.tok = nn.Embedding(V, n_embd)
        self.pos = nn.Embedding(self.total_pos, n_embd)

        if use_per_block_z:
            self.pos_block_emb = nn.Embedding(n_blocks, n_embd)
        else:
            self.pos_block_emb = None

        self.blocks = nn.ModuleList([
            BlockCausalV47(n_embd, n_head, use_sparse_attn=use_sparse_attn)
            for _ in range(n_layer)
        ])
        for blk in self.blocks:
            if ffn_type == "dense":
                blk.set_ffn(DenseFFNV47(n_embd, ffn_dim_dense))
            elif ffn_type == "moe":
                blk.set_ffn(MoEFFNV47(n_embd, ffn_dim_expert, n_experts, top_k))
            else:
                raise ValueError(f"Unknown ffn_type: {ffn_type}")

        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, V, bias=False)
        self.tok.weight = self.head.weight

    def _forward_flat(self, z, x, mask_input):
        """Variants A/B: dense attention on flat input [z, BOS, x_0..x_{T-1}]."""
        B, T = x.shape
        if mask_input is not None:
            x = x.clone()
            x[mask_input] = self.mask_id

        z_emb = self.z_to_emb(z).unsqueeze(1)
        bos_emb = self.tok(torch.tensor([self.bos_id], device=x.device)).expand(B, 1, -1)
        x_emb = self.tok(x)
        inp = torch.cat([z_emb, bos_emb, x_emb], dim=1)
        inp = inp + self.pos(torch.arange(T + 2, device=x.device))

        for blk in self.blocks:
            inp = blk(inp)
        logits = self.head(self.ln_f(inp))
        return logits[:, 1:T + 1, :]

    def _forward_per_block_z(self, z, x, mask_input):
        """Variant C: per-block z injection + sparse attention."""
        B, T = x.shape
        K = self.n_blocks
        Bsz = self.block_size

        if mask_input is not None:
            x = x.clone()
            x[mask_input] = self.mask_id

        z_emb_base = self.z_to_emb(z)
        bos_emb = self.tok(torch.tensor([self.bos_id], device=x.device)).expand(B, 1, -1)
        x_emb = self.tok(x)

        blocks = []
        for k in range(K):
            pos_emb_k = self.pos_block_emb(torch.tensor(k, device=z.device))
            z_block_k = z_emb_base + pos_emb_k.unsqueeze(0)
            z_block_k = z_block_k.unsqueeze(1)
            if k == 0:
                x_block = x_emb[:, 0:Bsz, :]
                block = torch.cat([z_block_k, bos_emb, x_block], dim=1)
            else:
                x_block = x_emb[:, k*Bsz:(k+1)*Bsz, :]
                block = torch.cat([z_block_k, x_block], dim=1)
            blocks.append(block)

        inp = torch.cat(blocks, dim=1)
        inp = inp + self.pos(torch.arange(self.total_pos, device=x.device))

        for blk in self.blocks:
            inp = blk(inp)
        logits = self.head(self.ln_f(inp))

        # Extract x logits from each block
        x_logits = []
        for k in range(K):
            if k == 0:
                x_logits.append(logits[:, 2:2 + Bsz, :])
            else:
                start = (Bsz + 2) + (k - 1) * (Bsz + 1)
                x_logits.append(logits[:, start + 1:start + 1 + Bsz, :])
        return torch.cat(x_logits, dim=1)

    def forward(self, z, x, mask_input=None):
        if self.use_per_block_z:
            logits = self._forward_per_block_z(z, x, mask_input)
        else:
            logits = self._forward_flat(z, x, mask_input)

        aux = torch.tensor(0.0, device=z.device)
        if self.ffn_type == "moe":
            aux = sum(blk.ffn.aux_loss() for blk in self.blocks)
        return logits, aux

    def reset_moe_stats(self):
        if self.ffn_type == "moe":
            for blk in self.blocks:
                if hasattr(blk.ffn, 'reset_stats'):
                    blk.ffn.reset_stats()

    def moe_importance_variance(self) -> float:
        if self.ffn_type != "moe":
            return float('nan')
        variances = []
        for blk in self.blocks:
            if hasattr(blk.ffn, 'importance_variance'):
                v = blk.ffn.importance_variance()
                if not math.isnan(v):
                    variances.append(v)
        return sum(variances) / len(variances) if variances else float('nan')

    def pos_block_emb_norm(self) -> float:
        if self.pos_block_emb is None:
            return 0.0
        return float(self.pos_block_emb.weight.norm().item())

    def num_active_params(self) -> int:
        """Count active params (dense-equivalent for MoE).

        For MoE with N experts Top-K: per-token active = K/N fraction of expert weights.
        """
        from model import TOP_K
        n = 0
        expert_total = 0
        for name, p in self.named_parameters():
            if 'experts' in name and self.ffn_type == "moe":
                expert_total += p.numel()
            else:
                n += p.numel()
        if self.ffn_type == "moe":
            n_experts = sum(1 for _ in self.blocks[0].ffn.experts)
            n += int(expert_total * TOP_K / max(n_experts, 1))
        return n

    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def sparse_attn_ratio(self) -> float:
        """Average sparsity ratio across layers (1 = full sparse, 0 = dense)."""
        if not self.use_sparse_attn:
            return 0.0
        ratios = [blk.attn.sparse_ratio() for blk in self.blocks]
        return sum(ratios) / len(ratios) if ratios else 0.0


# ============================================================
# Loss functions (复用 v46 模式)
# ============================================================
def make_block_mask(batch_size, seq_len, block_size=BLOCK_SIZE,
                    mask_rate_range=(0.1, 0.5), device="cuda"):
    n_blocks = seq_len // block_size
    assert seq_len % block_size == 0, f"T={seq_len} must be divisible by block_size={block_size}"
    rates = torch.empty(batch_size, n_blocks, device=device).uniform_(*mask_rate_range)
    rand = torch.rand(batch_size, n_blocks, block_size, device=device)
    block_mask = rand < rates.unsqueeze(-1)
    return block_mask.view(batch_size, seq_len)


def ar_loss(decoder: V47Decoder, z, x, mask_input=None):
    logits, _ = decoder(z, x, mask_input=mask_input)
    return F.cross_entropy(logits.reshape(-1, decoder.V), x.reshape(-1))


def block_diffusion_loss(decoder: V47Decoder, z, x, mask_rate_range=(0.1, 0.5)):
    mask = make_block_mask(x.size(0), x.size(1), block_size=decoder.block_size,
                            mask_rate_range=mask_rate_range, device=x.device)
    logits, aux = decoder(z, x, mask_input=mask)
    loss_per_tok = F.cross_entropy(
        logits.reshape(-1, decoder.V), x.reshape(-1), reduction='none'
    ).reshape(x.shape)
    n_masked = mask.float().sum().clamp(min=1)
    loss = (loss_per_tok * mask.float()).sum() / n_masked
    return loss, aux