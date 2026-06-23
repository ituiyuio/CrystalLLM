"""
Lorenz Data Generator: RK4 Trajectories for Training
====================================================

Day 1: 写 Lorenz 数据生成器 + ground truth 验证 + 计算真 PSD

设计:
- 5000 个训练轨迹, 500 个验证轨迹
- 每个轨迹 T=2000 步, dt=0.01 (总时长 20 Lyapunov times)
- 初始条件从 attractor basin 内随机采样 (避免初始瞬态)
- 训练前丢弃前 100 步瞬态
"""
from __future__ import annotations

import torch
import numpy as np
from pathlib import Path

from lorenz_oracle import LorenzOracle, lorenz_ode


def generate_lorenz_trajectories(
    n_trajectories: int,
    seq_len: int = 2000,
    dt: float = 0.01,
    burn_in: int = 100,
    init_range: tuple = (-15.0, 15.0),
    seed: int = 42,
    device: str = "cuda",
) -> torch.Tensor:
    """生成 Lorenz 轨迹数据集 (GPU 加速).

    Args:
        n_trajectories: 轨迹数
        seq_len: 每条轨迹长度 (含 burn_in 后)
        dt: 积分步长
        burn_in: 丢弃的初始瞬态步数
        init_range: 初始条件均匀分布范围
        seed: 随机种子
        device: 'cpu' or 'cuda'
    Returns:
        trajectories: (n_trajectories, seq_len, 3) tensor
    """
    if device == "cuda":
        # GPU 模式: 直接 tensor 操作, 不用 torch.Generator (太慢)
        torch.manual_seed(seed)
        oracle = LorenzOracle(dt=dt).to(device)
        total_steps = seq_len + burn_in
        init_state = (torch.rand(n_trajectories, 3, device=device)
                      * (init_range[1] - init_range[0]) + init_range[0])
        trajectories = oracle.rollout(init_state, steps=total_steps)
        return trajectories[:, burn_in:, :]
    else:
        # CPU 模式 (原实现, 兼容)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        oracle = LorenzOracle(dt=dt)
        total_steps = seq_len + burn_in
        init_state = torch.empty(n_trajectories, 3).uniform_(
            init_range[0], init_range[1], generator=gen
        )
        trajectories = oracle.rollout(init_state, steps=total_steps)
        return trajectories[:, burn_in:, :] if trajectories.shape[1] > burn_in else trajectories


def compute_dataset_lyapunov(trajectories: torch.Tensor, dt: float = 0.01,
                              window: int = 1000) -> float:
    """从轨迹数据估计最大 Lyapunov 指数.

    用经典 Wolf 算法简化版:
    - 取一段轨迹, 找相邻点 (距离 < ε), 追踪它们距离随时间的演化
    - 拟合 ln(distance) vs time 的斜率 = Lyapunov

    Args:
        trajectories: (N, T, 3)
        dt: 时间步长
        window: 用于估计的窗口长度
    Returns:
        lambda_max: 估计的 Lyapunov 指数 (理论值 ~0.906)
    """
    # 取一条轨迹用于估计
    traj = trajectories[0].cpu().numpy()  # (T, 3)
    T = traj.shape[0]

    # 简化算法: 计算相邻点距离增长率
    eps = 0.01  # 初始小扰动
    growth_rates = []

    for i in range(0, T - window, window // 2):
        # 在点 traj[i] 加扰动
        perturbed = traj[i] + eps * np.random.randn(3)
        perturbed = perturbed / np.linalg.norm(perturbed) * eps

        # 计算演化
        original_traj = traj[i:i + window]
        oracle = LorenzOracle(dt=dt)
        perturbed_traj = oracle.rollout(
            torch.tensor(perturbed[None], dtype=torch.float32), steps=window - 1
        )[0].cpu().numpy()

        # 距离随时间演化
        distances = np.linalg.norm(original_traj - perturbed_traj, axis=1)
        distances = np.maximum(distances, 1e-12)  # 避免 log(0)

        # 拟合 ln(d) vs t 的斜率 (忽略前几个点)
        skip = 5
        t_arr = np.arange(skip, window) * dt
        log_d = np.log(distances[skip:])
        if len(t_arr) > 2:
            slope = np.polyfit(t_arr, log_d, 1)[0]
            growth_rates.append(slope)

    if not growth_rates:
        return 0.0
    return float(np.median(growth_rates))


def compute_dataset_correlation_dim(trajectories: torch.Tensor,
                                     max_lag: int = 50) -> float:
    """估计相关维数 D_2 (Grassberger-Procaccia 简化版).

    D_2 = lim_{r->0} log(C(r)) / log(r)
    其中 C(r) = (1/N^2) Σ_{i,j} I(||x_i - x_j|| < r) 是相关积分.

    Args:
        trajectories: (N, T, 3)
        max_lag: 时间延迟嵌入的最大滞后
    Returns:
        D_2: 相关维数估计 (Lorenz 理论值 ~2.05)
    """
    # 用第一条轨迹的 Takens 嵌入
    traj = trajectories[0].cpu().numpy()  # (T, 3)

    # 简化: 直接用原始 3 维点云, 不做延迟嵌入
    points = traj[::10]  # 降采样
    N = len(points)

    # 计算所有点对的距离 (O(N^2), 限制 N <= 5000)
    if N > 2000:
        idx = np.random.choice(N, 2000, replace=False)
        points = points[idx]
        N = 2000

    # 用 sklearn 或手算 pairwise distance
    from scipy.spatial.distance import pdist, squareform
    dists = pdist(points)  # (N*(N-1)/2,)

    # 计算 C(r) 对若干 r 值
    r_values = np.logspace(-2, 1, 20)
    C_r = []
    for r in r_values:
        C = (dists < r).sum() / len(dists)
        C_r.append(max(C, 1e-12))

    # 拟合 log(C) vs log(r) 的斜率 (中间线性段)
    log_r = np.log10(r_values)
    log_C = np.log10(C_r)
    # 找最线性段
    mid = len(r_values) // 2
    coeffs = np.polyfit(log_r[mid - 3:mid + 3], log_C[mid - 3:mid + 3], 1)
    return float(coeffs[0])


if __name__ == "__main__":
    print("=" * 70)
    print("Day 1: Lorenz Dataset Generation + Validation")
    print("=" * 70)

    # 生成小样本验证
    print("\n[1] Generating sample trajectories (10 train + 5 val)...")
    train_data = generate_lorenz_trajectories(n_trajectories=10, seq_len=2000, seed=42)
    val_data = generate_lorenz_trajectories(n_trajectories=5, seq_len=2000, seed=99)
    print(f"  Train: {train_data.shape}")
    print(f"  Val:   {val_data.shape}")
    print(f"  Range x: [{train_data[..., 0].min():.2f}, {train_data[..., 0].max():.2f}]")
    print(f"  Range y: [{train_data[..., 1].min():.2f}, {train_data[..., 1].max():.2f}]")
    print(f"  Range z: [{train_data[..., 2].min():.2f}, {train_data[..., 2].max():.2f}]")

    # 估计 Lyapunov 指数
    print("\n[2] Estimating Lyapunov exponent...")
    lam = compute_dataset_lyapunov(train_data, dt=0.01)
    print(f"  Estimated lambda_max: {lam:.4f} (theoretical: 0.9056)")
    print(f"  Error: {abs(lam - 0.9056):.4f}")

    # 估计相关维数
    print("\n[3] Estimating correlation dimension D_2...")
    D2 = compute_dataset_correlation_dim(train_data)
    print(f"  Estimated D_2: {D2:.4f} (theoretical: 2.05)")
    print(f"  Error: {abs(D2 - 2.05):.4f}")

    # 计算真 PSD (作为 ground truth 用于后续对比)
    print("\n[4] Computing ground-truth PSD...")
    from psd_utils import compute_psd_fft
    fs = 100.0
    traj_x = val_data[0, :, 0].cpu().numpy()
    freqs, psd = compute_psd_fft(traj_x, fs=fs)
    print(f"  Frequencies: {freqs[0]:.2f} to {freqs[-1]:.2f} Hz")
    print(f"  PSD shape: {psd.shape}, peak freq: {freqs[psd.argmax()]:.2f} Hz")

    # 保存样本数据
    print("\n[5] Saving sample data...")
    out_path = Path(__file__).parent / "sample_data.pt"
    torch.save({
        "train": train_data,
        "val": val_data,
        "lyapunov_estimated": lam,
        "D2_estimated": D2,
    }, out_path)
    print(f"  Saved -> {out_path}")

    print("\n[OK] Day 1 components verified.")
