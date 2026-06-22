"""PE 模块集合 — Exp 24.

三个 PE 变体:
  - BlockCayleyPE: 真 Cayley 变换, block-diagonal 16×16 块
  - StandardRoPE: 标准 RoPE (冻结)
  - NoPE: 无 PE (identity)

所有 PE 接受 (B, T, d_model) 输入, 输出 (B, T, d_model).
"""
import math

import torch
import torch.nn as nn


class BlockCayleyPE(nn.Module):
    """Block-diagonal Cayley PE (静态, 无 context_net).

    设计:
      - d 维空间分成 n_blocks 个块, 每块 size = d // n_blocks
      - 每个块有独立的可学习 skew-symmetric 参数 A_block ∈ R^{s×s}
      - 位置 m 处: A(m) = m * A_block (线性缩放)
      - Cayley: R(m) = (I - A(m))^{-1} (I + A(m))
      - 总旋转 = block_diag(R_1(m), R_2(m), ..., R_n(m))
      - 应用: z' = block_diag_R @ z (按块独立旋转, 等价于 einsum)

    参数数: n_blocks * (s * (s-1) // 2) = n_blocks * 120 (s=16)
    """

    def __init__(self, d_model: int, n_blocks: int = 16, block_size: int = 16,
                 max_position: int = 2048):
        super().__init__()
        assert d_model == n_blocks * block_size, \
            f"d_model={d_model} must equal n_blocks={n_blocks} * block_size={block_size}"
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.max_position = max_position

        # 每个块一个静态 skew-symmetric 参数 (用上三角参数化, size * (size-1) // 2 个)
        n_skew_per_block = block_size * (block_size - 1) // 2  # = 120 for block_size=16
        # shape: (n_blocks, n_skew_per_block)
        self.A_params = nn.Parameter(torch.randn(n_blocks, n_skew_per_block) * 0.05)

        # 上三角索引 cache (在 device 上重建)
        triu_idx = torch.triu_indices(block_size, block_size, offset=1)  # (2, n_skew)
        self.register_buffer("triu_i", triu_idx[0], persistent=False)
        self.register_buffer("triu_j", triu_idx[1], persistent=False)

        # Identity cache
        self.register_buffer("I_block", torch.eye(block_size), persistent=False)

    def _build_block_rotation(self, A_params_block: torch.Tensor, position: int) -> torch.Tensor:
        """对一个块, 给定 A 参数和位置 m, 返回 R(m) = (I - mA)^{-1} (I + mA)."""
        s = self.block_size
        # 构造 skew-symmetric A: 16x16
        A = torch.zeros(s, s, device=A_params_block.device, dtype=A_params_block.dtype)
        A[self.triu_i, self.triu_j] = A_params_block
        A[self.triu_j, self.triu_i] = -A_params_block
        # 缩放到位置 m
        A = A * float(position)
        # Cayley 变换
        I = self.I_block.to(dtype=A.dtype)
        IA = I - A
        IB = I + A
        try:
            R = torch.linalg.solve(IA, IB)
        except RuntimeError:
            R = torch.linalg.pinv(IA) @ IB
        return R

    def _build_rotation_stack_vectorized(self, T: int) -> torch.Tensor:
        """向量化构造 (T, n_blocks, s, s) 旋转矩阵 stack.

        公式: A_stack[t, b] = t * skew(A_params[b])
              R_stack[t, b] = (I - A_stack[t, b])^{-1} (I + A_stack[t, b])
        """
        s = self.block_size
        n = self.n_blocks
        device = self.A_params.device
        dtype = self.A_params.dtype

        # 1. 构造 (n, s, s) skew-symmetric 基矩阵
        A_base = torch.zeros(n, s, s, device=device, dtype=dtype)
        A_base[:, self.triu_i, self.triu_j] = self.A_params
        A_base[:, self.triu_j, self.triu_i] = -self.A_params

        # 2. 扩展到 (T, n, s, s), 乘以位置 t
        positions = torch.arange(T, device=device, dtype=dtype).view(T, 1, 1, 1)
        A_stack = A_base.unsqueeze(0) * positions  # (T, n, s, s)

        # 3. Cayley 变换: R = (I - A)^{-1} (I + A), 用 batched solve
        I = self.I_block.to(dtype=dtype)  # (s, s)
        IA = I.unsqueeze(0).unsqueeze(0) - A_stack  # (T, n, s, s)
        IB = I.unsqueeze(0).unsqueeze(0) + A_stack
        # batched solve: solve(IA, IB) → IA^{-1} @ IB
        try:
            R_stack = torch.linalg.solve(IA, IB)
        except RuntimeError:
            R_stack = torch.linalg.pinv(IA) @ IB
        return R_stack

    def get_rotation_matrix(self, position: int) -> torch.Tensor:
        """返回位置 m 处的 (d_model, d_model) 总旋转矩阵 (用于 T3/T4 测试)."""
        R_stack = self._build_rotation_stack_vectorized(position + 1)
        # 取 position 处的旋转
        R_blocks = R_stack[position]  # (n, s, s)
        # 拼成 (d, d) block-diagonal
        R_full = torch.zeros(self.d_model, self.d_model,
                             device=R_blocks.device, dtype=R_blocks.dtype)
        s = self.block_size
        for b in range(self.n_blocks):
            start = b * s
            R_full[start:start+s, start:start+s] = R_blocks[b]
        return R_full

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, T, d_model) → (B, T, d_model) 旋转后的特征."""
        B, T, D = z.shape
        assert D == self.d_model, f"got d_model={D}, expected {self.d_model}"
        s = self.block_size
        n = self.n_blocks
        # reshape: (B, T, n, s)
        z_blocks = z.view(B, T, n, s)
        # 向量化构造 (T, n, s, s) 旋转矩阵 stack
        R_stack = self._build_rotation_stack_vectorized(T)  # (T, n, s, s)
        # 应用: einsum 't n s k, b t n k -> b t n s'
        out_blocks = torch.einsum('tnsk,btnk->btns', R_stack, z_blocks)
        return out_blocks.reshape(B, T, D)


class StandardRoPE(nn.Module):
    """标准 RoPE (冻结, 无学习参数). 用于直接对照."""

    def __init__(self, d_model: int, base_freq: float = 10000.0, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % 2 == 0, f"d_model={d_model} must be even"
        self.d_model = d_model
        self.base_freq = base_freq
        half = d_model // 2
        freqs = 1.0 / (base_freq ** (torch.arange(0, half).float() / half))
        self.register_buffer("freqs", freqs, persistent=False)
        # 预计算 cos/sin (T, half)
        pos = torch.arange(max_seq_len).float()
        angles = pos.unsqueeze(-1) * freqs.unsqueeze(0)
        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, T, D = z.shape
        assert T <= self.max_seq_len, f"T={T} > max_seq_len={self.max_seq_len}"
        cos_a = self.cos_cache[:T].to(z.dtype)  # (T, half)
        sin_a = self.sin_cache[:T].to(z.dtype)
        # 相邻维度配对旋转
        z_pairs = z.view(B, T, D // 2, 2)
        z_even = z_pairs[..., 0]
        z_odd = z_pairs[..., 1]
        new_even = z_even * cos_a - z_odd * sin_a
        new_odd = z_even * sin_a + z_odd * cos_a
        out_pairs = torch.stack([new_even, new_odd], dim=-1)
        return out_pairs.view(B, T, D)


class NoPE(nn.Module):
    """无 PE, identity. 用于 ablation 下界."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z