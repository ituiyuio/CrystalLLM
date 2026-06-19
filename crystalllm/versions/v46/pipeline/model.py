"""model.py — v46 Phase 0 从零训练模型 (50M active params)

架构 (~33M active params):
  Hidden dim: 512
  Layers:     8
  Heads:      8 (head_dim 64)
  Dense FFN:  512 → 3072 → 512       (variant A)
  MoE FFN:    4 experts × (512→1536→512), Top-2 routing (variants B, C)
  Per-block z injection (variant C):
    Block 0:   [z_emb, BOS, x_emb_0..x_emb_{B-1}]      → B+2 positions
    Block k>0: [z_emb + pos_block_emb[k], x_emb_{Bk}..] → B+1 positions
  Total positions (variant C): 18 + 31*17 = 545

三个变体共享同一架构, 通过 use_dense / use_per_block_z 切换:
  A: dense FFN + AR only, no per-block z
  B: MoE FFN + AR only, no per-block z
  C: MoE FFN + per-block z + 0.5 L_AR + 0.5 L_diff
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 配置常量 (与 spec.md 对齐)
# ============================================================
DEC_EMBD = 512
DEC_LAYER = 8
DEC_HEAD = 8
HEAD_DIM = DEC_EMBD // DEC_HEAD  # 64
FFN_DIM_DENSE = 3072
FFN_DIM_EXPERT = 1536
N_EXPERTS = 4
TOP_K = 2

# Block structure (variant C)
BLOCK_SIZE = 16
T_TOKEN = 512
N_BLOCKS = T_TOKEN // BLOCK_SIZE  # 32
# Variant C total positions: 18 + 31*17 = 545
V46_BLOCK_POS_LEN = (BLOCK_SIZE + 2) + (N_BLOCKS - 1) * (BLOCK_SIZE + 1)
# Variant A/B total positions: T + 2 = 514 (z + BOS + T)
V46_FLAT_POS_LEN = T_TOKEN + 2


# ============================================================
# Attention block (causal)
# ============================================================
class BlockCausal(nn.Module):
    """8-layer causal Transformer block with pre-norm."""

    def __init__(self, n_embd: int = DEC_EMBD, n_head: int = DEC_HEAD):
        super().__init__()
        self.nh = n_head
        self.head_dim = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # FFN is replaced externally by set_ffn()
        self.ffn = None  # placeholder, must be set

    def set_ffn(self, ffn: nn.Module):
        self.ffn = ffn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, T_, C = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B_, T_, 3, self.nh, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).contiguous().view(B_, T_, C))
        x = x + self.ffn(self.ln2(x))
        return x


# ============================================================
# Dense FFN (variant A)
# ============================================================
class DenseFFN(nn.Module):
    """Standard 2-layer MLP FFN."""

    def __init__(self, n_embd: int = DEC_EMBD, ffn_dim: int = FFN_DIM_DENSE):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))

    def aux_loss(self) -> torch.Tensor:
        """Dense FFN has no auxiliary loss."""
        return torch.tensor(0.0, device=next(self.parameters()).device)


# ============================================================
# MoE FFN (variants B, C)
# ============================================================
class MoEFFN(nn.Module):
    """Mixture-of-Experts FFN with Top-2 routing + load balance loss.

    Architecture:
      router: Linear(n_embd, n_experts) → logits
      experts: n_experts parallel DenseFFN(ffn_dim_expert)
      Top-K routing with softmax weights

    Returns weighted sum of expert outputs.
    """

    def __init__(self, n_embd: int = DEC_EMBD, ffn_dim: int = FFN_DIM_EXPERT,
                 n_experts: int = N_EXPERTS, top_k: int = TOP_K):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(n_embd, n_experts, bias=False)
        # Each expert is a small DenseFFN
        self.experts = nn.ModuleList([
            DenseFFN(n_embd, ffn_dim) for _ in range(n_experts)
        ])
        # Running stats for load balance monitoring (no grad)
        self.register_buffer('importance_acc', torch.zeros(n_experts))
        self.register_buffer('importance_count', torch.tensor(0.0))
        # Aux loss weight (Switch Transformer style)
        self.aux_loss_coef = 0.01

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        flat_x = x.reshape(-1, D)  # (B*T, D)

        # Router logits
        router_logits = self.router(flat_x)  # (B*T, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)  # (B*T, n_experts)

        # Top-K selection
        topk_probs, topk_idx = router_probs.topk(self.top_k, dim=-1)  # (B*T, top_k)
        # Normalize top-k weights
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        # Compute expert outputs
        out = torch.zeros_like(flat_x)
        for e_idx in range(self.n_experts):
            # Mask: which tokens selected this expert (in any of top_k slots)
            expert_mask = (topk_idx == e_idx).any(dim=-1)  # (B*T,)
            if not expert_mask.any():
                continue
            # For each selected token, find the weight for this expert
            # (a token may have selected this expert in slot 0 or slot 1)
            token_ids = expert_mask.nonzero(as_tuple=True)[0]
            # Get the weight for expert e_idx for each token
            # Use the first matching slot's weight (or zero if not selected)
            weight = torch.zeros(token_ids.shape[0], device=x.device, dtype=x.dtype)
            for slot in range(self.top_k):
                slot_mask = (topk_idx[token_ids, slot] == e_idx)
                weight = torch.where(slot_mask, topk_probs[token_ids, slot], weight)

            # Expert output for these tokens
            expert_out = self.experts[e_idx](flat_x[token_ids])  # (n_tokens, D)
            out[token_ids] += expert_out * weight.unsqueeze(-1)

        # Store router_logits for differentiable aux_loss
        self._last_router_logits = router_logits

        # Update importance stats (for monitoring, no grad)
        with torch.no_grad():
            # Importance = mean probability assigned to each expert across all tokens
            importance = router_probs.mean(dim=0)  # (n_experts,)
            self.importance_acc += importance.detach()
            self.importance_count += 1.0

        return out.reshape(B, T, D)

    def aux_loss(self, router_logits: torch.Tensor | None = None) -> torch.Tensor:
        """Switch Transformer load balance loss.

        L_aux = n_experts * sum_e (f_e * P_e)
        where:
          f_e = fraction of tokens routed to expert e (one-hot mask, non-diff)
          P_e = mean router probability for expert e (DIFFERENTIABLE through router)

        Encourages uniform routing.

        Args:
            router_logits: Optional precomputed router logits (B*T, n_experts).
                          If None, falls back to most recent stored logits from forward().
        """
        if router_logits is None:
            router_logits = getattr(self, '_last_router_logits', None)
            if router_logits is None:
                if self.importance_count < 1:
                    return torch.tensor(0.0, device=next(self.parameters()).device)
                importance = self.importance_acc / self.importance_count.clamp(min=1.0)
                return self.aux_loss_coef * self.n_experts * (importance * importance).sum()

        # Compute differentiable aux loss from current batch router logits
        router_probs = F.softmax(router_logits, dim=-1)  # (B*T, n_experts)
        # Fraction of tokens per expert (one-hot, non-diff but used for weighting)
        with torch.no_grad():
            topk_idx = router_probs.topk(self.top_k, dim=-1)[1]  # (B*T, top_k)
            # For each expert, count fraction of tokens that selected it
            f = torch.zeros(self.n_experts, device=router_logits.device)
            for e in range(self.n_experts):
                f[e] = (topk_idx == e).any(dim=-1).float().mean()
        # Mean router probability per expert (diff)
        P = router_probs.mean(dim=0)  # (n_experts,)
        # Switch Transformer auxiliary loss
        return self.aux_loss_coef * self.n_experts * (f * P).sum()

    def reset_stats(self):
        """Reset running importance stats (call between eval/train or periodically)."""
        self.importance_acc.zero_()
        self.importance_count.zero_()

    def importance_variance(self) -> float:
        """Return variance of expert importance (sanity check: should be small)."""
        if self.importance_count < 1:
            return float('nan')
        importance = self.importance_acc / self.importance_count
        return float(importance.var().item())


# ============================================================
# V46 Decoder (supports 3 variants)
# ============================================================
class V46Decoder(nn.Module):
    """Decoder for v46 Phase 0.

    Variants (controlled by config):
      A (dense, no per-block z):   ffn_type="dense", use_per_block_z=False
      B (MoE, no per-block z):    ffn_type="moe",   use_per_block_z=False
      C (MoE + per-block z):      ffn_type="moe",   use_per_block_z=True

    Forward signature:
      logits, aux_loss = decoder(z, x, mask_input=None)
        z:          (B, D_Z)
        x:          (B, T) token ids (T=512)
        mask_input: (B, T) bool, True → replace with MASK_ID. Only used for variant C
                                          during L_diff computation.

    Output:
      logits: (B, T, V)
      aux_loss: scalar (MoE load balance loss, or 0.0 for variant A)
    """

    def __init__(
        self,
        V: int,
        D_Z: int = 256,
        ffn_type: str = "dense",        # "dense" | "moe"
        use_per_block_z: bool = False,  # True only for variant C
        bos_id: int = 1,
        mask_id: int = 0,               # 0 means no <mask> token
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
        self.bos_id = bos_id
        self.mask_id = mask_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_blocks = n_blocks
        self.block_size = block_size

        # Total positions depends on variant
        self.total_pos = (
            V46_BLOCK_POS_LEN if use_per_block_z else V46_FLAT_POS_LEN
        )

        # Embeddings
        self.z_to_emb = nn.Linear(D_Z, n_embd)
        self.tok = nn.Embedding(V, n_embd)
        self.pos = nn.Embedding(self.total_pos, n_embd)

        # Per-block z position embeddings (variant C only)
        if use_per_block_z:
            # pos_block_emb[k] for k = 0..n_blocks-1 (block index)
            self.pos_block_emb = nn.Embedding(n_blocks, n_embd)
        else:
            self.pos_block_emb = None

        # Transformer blocks
        self.blocks = nn.ModuleList([BlockCausal(n_embd, n_head) for _ in range(n_layer)])
        for blk in self.blocks:
            if ffn_type == "dense":
                blk.set_ffn(DenseFFN(n_embd, ffn_dim_dense))
            elif ffn_type == "moe":
                blk.set_ffn(MoEFFN(n_embd, ffn_dim_expert, n_experts, top_k))
            else:
                raise ValueError(f"Unknown ffn_type: {ffn_type}")

        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, V, bias=False)
        self.tok.weight = self.head.weight  # tied embedding

    # ============================================================
    # Variant A/B: flat input [z_emb, BOS, x_emb_0..x_emb_{T-1}]
    # ============================================================
    def _forward_flat(self, z: torch.Tensor, x: torch.Tensor,
                       mask_input: torch.Tensor | None) -> torch.Tensor:
        """Forward for variant A/B (no per-block z).

        Total positions: 514 (z + BOS + T tokens).
        mask_input: (B, T) bool, True → replace x with mask_id.
        """
        B, T = x.shape

        # Optional masking (for L_diff, though variants A/B only use L_AR)
        if mask_input is not None:
            x = x.clone()
            x[mask_input] = self.mask_id

        # Embeddings
        z_emb = self.z_to_emb(z).unsqueeze(1)  # (B, 1, D)
        bos_emb = self.tok(torch.tensor([self.bos_id], device=x.device)).expand(B, 1, -1)
        x_emb = self.tok(x)  # (B, T, D)

        inp = torch.cat([z_emb, bos_emb, x_emb], dim=1)  # (B, T+2, D)
        inp = inp + self.pos(torch.arange(T + 2, device=x.device))

        for blk in self.blocks:
            inp = blk(inp)

        logits = self.head(self.ln_f(inp))  # (B, T+2, V)
        # Skip z and BOS positions, return only x logits
        return logits[:, 1:T + 1, :]  # (B, T, V)

    # ============================================================
    # Variant C: per-block z injection (block-structured input)
    # ============================================================
    def _forward_per_block_z(self, z: torch.Tensor, x: torch.Tensor,
                              mask_input: torch.Tensor | None) -> torch.Tensor:
        """Forward for variant C (per-block z injection).

        Layout:
          Block 0:   [z_emb_base, BOS, x_emb_0..x_emb_{B-1}]                      → B+2 positions
          Block k>0: [z_emb_base + pos_block_emb[k], x_emb_{Bk}..x_emb_{Bk+B-1}] → B+1 positions

        Total positions: (B+2) + (K-1)*(B+1) = 18 + 31*17 = 545 (for B=16, K=32).

        mask_input: (B, T) bool, True → replace x with mask_id. The mask is in x-token
                    coordinates; we map it to input positions when masking.
        """
        B, T = x.shape
        K = self.n_blocks
        Bsz = self.block_size

        # Optional masking (in x-token space)
        if mask_input is not None:
            x = x.clone()
            x[mask_input] = self.mask_id

        # Base z_emb (shared across blocks)
        z_emb_base = self.z_to_emb(z)  # (B, D)
        bos_emb = self.tok(torch.tensor([self.bos_id], device=x.device)).expand(B, 1, -1)
        x_emb = self.tok(x)  # (B, T, D)

        # Build block-structured input
        blocks = []
        for k in range(K):
            # z_block_k = z_emb_base + pos_block_emb[k]
            # pos_block_emb is a single embedding per block (broadcast over batch)
            pos_emb_k = self.pos_block_emb(torch.tensor(k, device=z.device))  # (D,)
            z_block_k = z_emb_base + pos_emb_k.unsqueeze(0)  # (B, D)
            z_block_k = z_block_k.unsqueeze(1)  # (B, 1, D)

            if k == 0:
                x_block = x_emb[:, 0:Bsz, :]
                block = torch.cat([z_block_k, bos_emb, x_block], dim=1)  # (B, Bsz+2, D)
            else:
                x_block = x_emb[:, k*Bsz:(k+1)*Bsz, :]
                block = torch.cat([z_block_k, x_block], dim=1)  # (B, Bsz+1, D)
            blocks.append(block)

        inp = torch.cat(blocks, dim=1)  # (B, total_pos=545, D)

        # Position IDs (sequential, since each position is unique)
        inp = inp + self.pos(torch.arange(self.total_pos, device=x.device))

        for blk in self.blocks:
            inp = blk(inp)

        logits = self.head(self.ln_f(inp))  # (B, total_pos, V)

        # Extract x logits from each block
        x_logits = []
        for k in range(K):
            if k == 0:
                # x positions are 2..(Bsz+1) within block 0 (skip z at 0, BOS at 1)
                x_logits.append(logits[:, 2:2 + Bsz, :])
            else:
                # Block k starts at position (Bsz+2) + (k-1)*(Bsz+1) = 18 + (k-1)*17
                start = (Bsz + 2) + (k - 1) * (Bsz + 1)
                # x positions are start+1 .. start+Bsz (skip z at start)
                x_logits.append(logits[:, start + 1:start + 1 + Bsz, :])
        return torch.cat(x_logits, dim=1)  # (B, T, V)

    # ============================================================
    # Forward dispatch
    # ============================================================
    def forward(self, z: torch.Tensor, x: torch.Tensor,
                mask_input: torch.Tensor | None = None):
        if self.use_per_block_z:
            logits = self._forward_per_block_z(z, x, mask_input)
        else:
            logits = self._forward_flat(z, x, mask_input)

        # Aggregate MoE auxiliary loss across all blocks
        aux = torch.tensor(0.0, device=z.device)
        if self.ffn_type == "moe":
            aux = sum(blk.ffn.aux_loss() for blk in self.blocks)
        return logits, aux

    def reset_moe_stats(self):
        """Reset MoE running stats (call between train/eval or at eval start)."""
        if self.ffn_type == "moe":
            for blk in self.blocks:
                if hasattr(blk.ffn, 'reset_stats'):
                    blk.ffn.reset_stats()

    def moe_importance_variance(self) -> float:
        """Average importance variance across MoE blocks (sanity metric)."""
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
        """Norm of per-block z position embeddings (sanity: should be > 0 if learning)."""
        if self.pos_block_emb is None:
            return 0.0
        return float(self.pos_block_emb.weight.norm().item())

    def num_active_params(self) -> int:
        """Count of active parameters (all params in dense, MoE has 4x storage)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_params(self) -> int:
        """Count of all parameters including MoE storage."""
        return sum(p.numel() for p in self.parameters())


# ============================================================
# Loss functions
# ============================================================
def make_block_mask(batch_size: int, seq_len: int, block_size: int = BLOCK_SIZE,
                    mask_rate_range: tuple = (0.1, 0.5),
                    device: str = "cuda") -> torch.Tensor:
    """Generate per-block mask: each block samples its own mask rate, tokens are
    independently Bernoulli sampled within each block.

    Args:
        batch_size: B
        seq_len: T (must be divisible by block_size)
        block_size: B
        mask_rate_range: (min, max) for Uniform mask rate per block
        device: torch device

    Returns:
        (B, T) bool tensor, True = masked
    """
    n_blocks = seq_len // block_size
    assert seq_len % block_size == 0, f"T={seq_len} must be divisible by block_size={block_size}"

    # Per-block mask rate (B, n_blocks)
    rates = torch.empty(batch_size, n_blocks, device=device).uniform_(*mask_rate_range)
    # Per-token Bernoulli within each block (B, n_blocks, block_size)
    rand = torch.rand(batch_size, n_blocks, block_size, device=device)
    block_mask = rand < rates.unsqueeze(-1)
    return block_mask.view(batch_size, seq_len)


def ar_loss(decoder: V46Decoder, z: torch.Tensor, x: torch.Tensor,
            mask_input: torch.Tensor | None = None) -> torch.Tensor:
    """L_AR: standard next-token CE (no masking).

    Optionally accepts mask_input for the diffusion case (so we can reuse
    the same decoder forward signature).
    """
    logits, _ = decoder(z, x, mask_input=mask_input)
    return F.cross_entropy(logits.reshape(-1, decoder.V), x.reshape(-1))


def block_diffusion_loss(decoder: V46Decoder, z: torch.Tensor, x: torch.Tensor,
                          mask_rate_range: tuple = (0.1, 0.5)) -> tuple:
    """L_diff: MDLM-style block-diffusion loss.

    Generates a per-block mask (each block gets its own mask rate ~ Uniform(low, high)),
    applies mask_input to decoder, computes CE only on masked positions, averages
    over masked positions (not all positions).

    Returns:
        (loss, aux_loss): both are scalar tensors with grad
    """
    mask = make_block_mask(x.size(0), x.size(1), block_size=decoder.block_size,
                            mask_rate_range=mask_rate_range, device=x.device)
    logits, aux = decoder(z, x, mask_input=mask)
    # CE on masked positions only (against original x, not masked)
    loss_per_tok = F.cross_entropy(
        logits[..., :decoder.V - 1].reshape(-1, decoder.V - 1) if False else logits.reshape(-1, decoder.V),
        x.reshape(-1), reduction='none'
    ).reshape(x.shape)  # (B, T)
    n_masked = mask.float().sum().clamp(min=1)
    loss = (loss_per_tok * mask.float()).sum() / n_masked
    return loss, aux