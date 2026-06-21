"""Wave Function Transformer (Option A) — 严格量子启发实现.

数学基础:
  - 态空间: ψ ∈ ℂ^d, ‖ψ‖² = 1 (归一化)
  - 演化: ψ ← exp(-iH dt) ψ via Trotter 分解
  - 测量: P(v) = |⟨v|ψ⟩|² (Born rule)

关键设计:
  1. UnitaryLinear: Cayley transform U = (I - A)(I + A)^{-1}, A skew-Hermitian
  2. Born rule attention: scores = |⟨q, k⟩|² 而非 |⟨q, k⟩|
  3. modReLU 激活: 保留相位, 只修改 magnitude
  4. Born rule readout: 概率 = |project(ψ)|² / sum

参考: notes/2026-06-21-wave-function-scalpel.md §📐 M1-M5 失败机制
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# 基础数学: Cayley transform
# ===========================================================================
def cayley_transform(A: torch.Tensor) -> torch.Tensor:
    """Cayley transform: skew-Hermitian A → unitary U.

    U = (I - A)(I + A)^{-1}

    性质:
      - A skew-Hermitian ⟹ U unitary
      - U = I when A = 0 (identity)
      - 谱范数 ‖U‖ = 1 (preserves norms)
      - O(d^3) 矩阵求逆

    数值稳定: 需 A 谱范数 < 1, 保证 (I + A) 可逆.
    """
    I = torch.eye(A.size(-1), device=A.device, dtype=A.dtype)
    return torch.linalg.solve(I + A, I - A)


def skew_hermitian(*shape, std: float = 0.01, dtype=torch.complex64) -> torch.Tensor:
    """随机 skew-Hermitian 矩阵: A = (X - X^H) / 2."""
    X = torch.randn(*shape, dtype=dtype) * std
    return (X - X.conj().transpose(-2, -1)) / 2


# ===========================================================================
# Unitary linear (Cayley)
# ===========================================================================
class UnitaryLinear(nn.Module):
    """Unitary 线性层 via Cayley.

    Y = X @ U,  U ∈ U(d) 由 skew-Hermitian A 参数化.
    训练时 A 随机, 前向时计算 U = cayley(A).
    """

    def __init__(self, dim_in: int, dim_out: int, std: float = 0.01):
        super().__init__()
        assert dim_in == dim_out, "UnitaryLinear 当前仅支持 dim_in == dim_out (square)"
        self.dim = dim_in
        self.A = nn.Parameter(skew_hermitian(dim_in, dim_in, std=std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., dim) complex → (..., dim) complex."""
        U = cayley_transform(self.A)
        return x @ U


# ===========================================================================
# Complex LayerNorm
# ===========================================================================
class ComplexLayerNorm(nn.Module):
    """对复数 tensor 分别 LN real / imag."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.ln_real = nn.LayerNorm(dim, eps=eps)
        self.ln_imag = nn.LayerNorm(dim, eps=eps)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.complex(self.ln_real(z.real), self.ln_imag(z.imag))


class WaveFunctionNorm(nn.Module):
    """总 norm 归一化: ‖ψ‖² = 1 (quantum state semantics).

    保留每个维度的相对 magnitude, 但总 norm 固定为 1.
    与 per-element 归一化不同, 这允许 wave function 在不同维度上有不同的 |ψ|.

    z' = z / ‖z‖  where ‖z‖ = sqrt(Σ|zi|²)
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Total norm per token: ‖ψ‖
        norm_sq = (z.real ** 2 + z.imag ** 2).sum(dim=-1, keepdim=True) + self.eps
        norm = torch.sqrt(norm_sq)
        return z / norm


# ===========================================================================
# RoPE for complex wave function
# ===========================================================================
def apply_rope_2d(psi: torch.Tensor, base_freq: float = 10000.0) -> torch.Tensor:
    """对 complex ψ 施加 RoPE-style 2D 旋转.

    对每个复数维度 k, 旋转角 = pos * freq[k]:
      new_ψ[k] = ψ[k] * exp(i * pos * freq[k])
              = ψ[k] * (cos(θ) + i sin(θ))
              = (real[k] * cos(θ) - imag[k] * sin(θ)) + i * (real[k] * sin(θ) + imag[k] * cos(θ))

    这是 z · e^{iθ} — 在复平面上的 2D 旋转, 等价于 RoPE 在 (real, imag) 配对上的应用.
    """
    B, T, D = psi.shape
    # Half frequencies (per complex dimension)
    freqs = 1.0 / (base_freq ** (torch.arange(D, device=psi.device).float() / D))
    pos = torch.arange(T, device=psi.device).float()
    angles = torch.einsum('t,k->tk', pos, freqs)  # (T, D)
    cos_a = torch.cos(angles)  # (T, D) real
    sin_a = torch.sin(angles)  # (T, D) real

    real = psi.real  # (B, T, D) real
    imag = psi.imag  # (B, T, D) real
    new_real = real * cos_a.unsqueeze(0) - imag * sin_a.unsqueeze(0)  # (B, T, D)
    new_imag = real * sin_a.unsqueeze(0) + imag * cos_a.unsqueeze(0)  # (B, T, D)
    return torch.complex(new_real, new_imag)


# ===========================================================================
# modReLU: phase-preserving non-linearity
# ===========================================================================
class ModReLU(nn.Module):
    """modReLU: z' = ReLU(|z| + b) * (z / |z|).

    保留相位, 只修改 magnitude. 不可逆 (会丢相位为 0 的信息),
    但比 complex GELU 更波函数友好.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(dim))  # 偏置

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = torch.sqrt(z.real ** 2 + z.imag ** 2 + 1e-8)
        phase = z / mag  # 单位相位
        new_mag = F.relu(mag + self.b)
        return new_mag * phase


# ===========================================================================
# Wave Function Attention (Born rule)
# ===========================================================================
class WaveFunctionAttention(nn.Module):
    """Born rule attention with unitary Q, K, V.

    Q, K, V 都是 unitary 旋转 (Cayley parameterization).
    Score: P(attend to j | query i) = |⟨q_i, k_j⟩|² / Σ_j |⟨q_i, k_j⟩|²
    Output: out_i = Σ_j attn_w[i,j] * v_j
    """

    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # Unitary Q, K, V, Out
        self.A_q = nn.Parameter(skew_hermitian(dim, dim, std=0.01))
        self.A_k = nn.Parameter(skew_hermitian(dim, dim, std=0.01))
        self.A_v = nn.Parameter(skew_hermitian(dim, dim, std=0.01))
        self.A_out = nn.Parameter(skew_hermitian(dim, dim, std=0.01))

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """psi: (B, T, D) complex → (B, T, D) complex."""
        B, T, D = psi.shape

        U_q = cayley_transform(self.A_q)
        U_k = cayley_transform(self.A_k)
        U_v = cayley_transform(self.A_v)
        U_out = cayley_transform(self.A_out)

        # Unitary projections
        q = psi @ U_q  # (B, T, D)
        k = psi @ U_k
        v = psi @ U_v

        # Multi-head reshape
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)

        # Inner product: <q_i, k_j> = sum_d q_i[d] * conj(k_j[d])
        # scores: (B, H, T_q, T_k) complex
        scores = torch.einsum('bihd,bjhd->bhij', q, k.conj())

        # Born rule: P(i→j) ∝ |<q_i, k_j>|²
        prob = scores.real ** 2 + scores.imag ** 2  # (B, H, T, T), real ≥ 0

        # Causal mask
        causal = torch.triu(torch.ones(T, T, device=psi.device, dtype=torch.bool), diagonal=1)
        prob = prob.masked_fill(causal.unsqueeze(0).unsqueeze(0), 0.0)

        # Softmax over keys (with temperature 1/sqrt(d) for stability)
        attn_w = F.softmax(prob / math.sqrt(self.head_dim), dim=-1)
        # Cast to complex for einsum (PyTorch doesn't broadcast real * complex in einsum)
        attn_w_c = attn_w.to(v.dtype)

        # Apply attention (complex weights, complex v)
        out = torch.einsum('bhij,bjhd->bihd', attn_w_c, v)  # (B, T, H, dh) complex
        out = out.reshape(B, T, D)

        # Output projection (unitary)
        return out @ U_out


# ===========================================================================
# Wave Function FFN
# ===========================================================================
class WaveFunctionFFN(nn.Module):
    """FFN with complex linear layers and modReLU activation.

    ψ ← (modReLU(W_1 ψ)) W_2

    备注: W_1, W_2 是非酉的 complex linear (modReLU 已经非酉, 酉 FFN 收益有限).
    注意力层保持严格酉.
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1_real = nn.Linear(dim, hidden_dim, bias=False)
        self.fc1_imag = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2_real = nn.Linear(hidden_dim, dim, bias=False)
        self.fc2_imag = nn.Linear(hidden_dim, dim, bias=False)
        # Initialize: small weights
        for layer in [self.fc1_real, self.fc1_imag, self.fc2_real, self.fc2_imag]:
            nn.init.xavier_uniform_(layer.weight)
        self.act = ModReLU(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _complex_linear(self, x, wr, wi):
        """(a + ib) @ (W_r + iW_i) = (a W_r - b W_i) + i(a W_i + b W_r)."""
        out_r = self._linear(x.real, wr) - self._linear(x.imag, wi)
        out_i = self._linear(x.real, wi) + self._linear(x.imag, wr)
        return torch.complex(out_r, out_i)

    def _linear(self, x, w):
        return F.linear(x, w)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        # Complex linear: ψ @ (W_r + i W_i)
        h = self._complex_linear(psi, self.fc1_real.weight, self.fc1_imag.weight)
        h = self.act(h)
        h = self.dropout(h)
        out = self._complex_linear(h, self.fc2_real.weight, self.fc2_imag.weight)
        return out


# ===========================================================================
# Wave Function Block (Trotter step)
# ===========================================================================
class WaveFunctionBlock(nn.Module):
    """One block: Trotter step with pre-norm + residual.

    ψ ← ψ + U_pos(ψ)         # RoPE (position)
    ψ ← ψ + α · U_attn(LN(ψ))  # Born rule attention (scaled)
    ψ ← ψ + α · U_ffn(LN(ψ))   # FFN with modReLU (scaled)

    变体:
      - use_wfnorm=True: 严格 total norm=1 (物理 wave function)
      - use_wfnorm=False: 纯 complex net (无 norm 约束, 让 magnitude 自由)
    """

    def __init__(self, dim: int, n_heads: int = 8, mlp_ratio: int = 4, dropout: float = 0.0,
                 residual_scale: float = 0.1, use_wfnorm: bool = False):
        super().__init__()
        self.attn = WaveFunctionAttention(dim, n_heads)
        self.ffn = WaveFunctionFFN(dim, dim * mlp_ratio, dropout)
        self.ln1 = ComplexLayerNorm(dim)
        self.ln2 = ComplexLayerNorm(dim)
        if use_wfnorm:
            self.norm = WaveFunctionNorm()  # total ‖ψ‖² = 1
        else:
            self.norm = nn.Identity()
        self.residual_scale = residual_scale

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        # Position encoding (RoPE) + residual (full scale, RoPE is unitary)
        psi = psi + apply_rope_2d(psi)
        # Attention with scaling
        psi = psi + self.residual_scale * self.attn(self.ln1(psi))
        # FFN with scaling
        psi = psi + self.residual_scale * self.ffn(self.ln2(psi))
        # Optional norm
        psi = self.norm(psi)
        return psi


# ===========================================================================
# Wave Function Embedding
# ===========================================================================
class WaveFunctionEmbedding(nn.Module):
    """Token → complex vector embedding."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.emb_real = nn.Embedding(vocab_size, dim)
        self.emb_imag = nn.Embedding(vocab_size, dim)
        # Initialize imag small (start near real)
        nn.init.normal_(self.emb_real.weight, std=0.02)
        nn.init.normal_(self.emb_imag.weight, std=0.01)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.complex(self.emb_real(ids), self.emb_imag(ids))


# ===========================================================================
# Born Rule Readout
# ===========================================================================
class BornRuleHead(nn.Module):
    """Output: P(v) = |⟨v|ψ⟩|² / Σ_v' |⟨v'|ψ⟩|².

    W: (V, D) complex, 是 vocab basis 在 hidden space 中的表示.
    P(v) = |<v|ψ>|² 是该 basis vector 与 ψ 的 overlap 的 magnitude 平方.
    """

    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        # Initialize small (near uniform distribution)
        std = 1.0 / math.sqrt(dim)
        self.W = nn.Parameter(torch.randn(vocab_size, dim, dtype=torch.complex64) * std)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """psi: (B, T, D) complex → (B, T, V) probabilities (real, normalized)."""
        # amp[v] = sum_i W[v, i] * conj(psi[b, t, i])? Or psi * W?
        # Convention: amp[v] = <v|ψ> = sum_i v_i* ψ_i = sum_i W[v, i]* conj(psi[b, t, i])
        # Or: amp[v] = sum_i W[v, i] * psi[b, t, i] (depending on convention)
        # Use: amp = psi @ W^T (closer to standard linear)
        # Standard linear: out = W @ psi = sum_i W[v, i] * psi[b, t, i]
        # For complex: out = W @ psi, no conj on psi
        amp = torch.einsum('btd,vd->btv', psi, self.W)  # (B, T, V) complex
        prob = amp.real ** 2 + amp.imag ** 2  # Born rule
        # Normalize to probability distribution
        prob = prob / prob.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return prob


# ===========================================================================
# Full Wave Function Transformer
# ===========================================================================
class WaveFunctionTransformer(nn.Module):
    """严格波函数 Transformer.

    架构:
      1. ψ_0 = embedding(token_ids)        # 初始 wave function
      2. for l in layers: ψ_l = block(ψ_{l-1})  # Trotter evolution
      3. P(v) = |⟨v|ψ_L⟩|² / Σ_v' |⟨v'|ψ_L⟩|²  # Born rule readout

    关键性质:
      - Phase preserved: complex dtype throughout
      - Unitarity: linear layers are unitary via Cayley
      - Born rule: attention uses |⟨q,k⟩|², output uses |⟨v|ψ⟩|²
      - Non-linearity: modReLU (phase-preserving)
    """

    def __init__(self, vocab_size: int, dim: int = 256, n_layers: int = 6,
                 n_heads: int = 8, max_seq_len: int = 2048,
                 mlp_ratio: int = 4, dropout: float = 0.0,
                 use_wfnorm: bool = False):
        super().__init__()
        self.config = type('Config', (), {
            'vocab_size': vocab_size, 'dim': dim, 'n_layers': n_layers,
            'n_heads': n_heads, 'max_seq_len': max_seq_len, 'dropout': dropout,
        })()
        self.token_emb = WaveFunctionEmbedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            WaveFunctionBlock(dim, n_heads, mlp_ratio, dropout, use_wfnorm=use_wfnorm)
            for _ in range(n_layers)
        ])
        self.ln_f = ComplexLayerNorm(dim)
        self.head = BornRuleHead(dim, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, T) int → (B, T, V) probabilities."""
        psi = self.token_emb(ids)  # (B, T, D) complex
        for layer in self.layers:
            psi = layer(psi)
        psi = self.ln_f(psi)
        return self.head(psi)


# ===========================================================================
# 训练辅助: count params
# ===========================================================================
def count_params(model: nn.Module) -> int:
    """参数计数 (complex 参数按 2x 算)."""
    n = 0
    for p in model.parameters():
        if p.is_complex():
            n += p.numel() * 2
        else:
            n += p.numel()
    return n


# ===========================================================================
# Smoke test
# ===========================================================================
if __name__ == "__main__":
    """Smoke test: forward + backward + 数值 sanity."""
    print("=== Wave Function Transformer smoke test ===\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    vocab_size = 2261
    dim = 64  # 小尺寸测试
    n_layers = 4
    n_heads = 4

    model = WaveFunctionTransformer(
        vocab_size=vocab_size, dim=dim, n_layers=n_layers, n_heads=n_heads,
    ).to(device)
    n_params = count_params(model)
    print(f"params: {n_params:,} (small test model)")

    # Forward
    B, T = 2, 16
    ids = torch.randint(0, vocab_size, (B, T), device=device)
    print(f"\n1. Forward:")
    prob = model(ids)
    print(f"   prob shape: {prob.shape}, dtype: {prob.dtype}")
    print(f"   prob[0, 0, :5]: {prob[0, 0, :5].detach().cpu().tolist()}")
    print(f"   prob sum (should be 1): {prob.sum(dim=-1)[0, 0].item():.4f}")

    # Backward
    print(f"\n2. Backward:")
    target = torch.randint(0, vocab_size, (B, T), device=device)
    loss = F.cross_entropy(prob.reshape(-1, vocab_size), target.reshape(-1))
    loss.backward()
    print(f"   loss: {loss.item():.4f}")

    # 数值 sanity: check |ψ| 变化
    print(f"\n3. Wave function magnitude:")
    psi = model.token_emb(ids)
    mag = torch.sqrt(psi.real**2 + psi.imag**2 + 1e-8)
    print(f"   input  ψ |·| mean: {mag.mean().item():.4f}, std: {mag.std().item():.4f}")
    psi = model.ln_f(psi)
    for i, layer in enumerate(model.layers):
        psi = layer(psi)
    mag = torch.sqrt(psi.real**2 + psi.imag**2 + 1e-8)
    print(f"   output ψ |·| mean: {mag.mean().item():.4f}, std: {mag.std().item():.4f}")

    # 单元测试: Cayley
    print(f"\n4. Cayley transform unit test:")
    A = skew_hermitian(8, 8, std=0.1, dtype=torch.complex64).to(device)
    U = cayley_transform(A)
    # U 应该 unitary: U^H U = I
    UH = U.conj().T
    I_check = UH @ U
    err = (I_check - torch.eye(8, device=device, dtype=torch.complex64)).abs().max().item()
    print(f"   ‖U^H U - I‖_max = {err:.6e} (should be < 1e-5)")

    print(f"\n=== All smoke tests passed ===")
