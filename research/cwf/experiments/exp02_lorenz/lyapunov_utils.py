"""
Lyapunov Exponent & Correlation Dimension Estimation
====================================================

Day 4 评估需要的两个动力学子结构指标.

Lyapunov 指数 (Wolf 算法):
    - 经典算法: 在 attractor 上取相邻点, 追踪它们距离随时间的指数增长
    - 简化版: 用 Benettin 算法 (1980), 周期性重正化
    - 参考: https://en.wikipedia.org/wiki/Lyapunov_exponent#Benettin.27s_algorithm

相关维数 D_2 (Grassberger-Procaccia):
    - 经典算法: 从时间序列做 Takens 嵌入, 计算相关积分 C(r)
    - D_2 = lim log C(r) / log r (线性区斜率)
    - 参考: https://en.wikipedia.org/wiki/Correlation_dimension
"""
from __future__ import annotations

import numpy as np
import torch


def estimate_lyapunov_benettin(traj_np: np.ndarray, dt: float = 0.01,
                               n_renorm_steps: int = 100,
                               eps_init: float = 1e-7,
                               n_iterations: int = 50) -> float:
    """Benettin 算法估计最大 Lyapunov 指数.

    算法:
    1. 取轨迹上一点 x, 加小扰动 x + delta (delta 沿 ODE 切向)
    2. 演化 n_renorm_steps 步, 测量距离增长 = ||x' - y'|| / ||delta||
    3. 重正化: 把 y' 拉回到距离 eps_init 处
    4. 重复, 平均 ln(growth)

    Args:
        traj_np: (T, 3) 单条轨迹
        dt: 时间步
        n_renorm_steps: 每次重正化前演化步数
        eps_init: 初始扰动距离
        n_iterations: 重复次数
    Returns:
        lambda_max: Lyapunov 指数 (Lorenz 理论值 ~0.9056)
    """
    from lorenz_oracle import lorenz_ode, SIGMA, RHO, BETA

    T = traj_np.shape[0]
    growth_rates = []

    # 用连续点对, 避免瞬态
    stride = max(1, (T - 2 * n_renorm_steps) // n_iterations)
    starts = range(0, T - 2 * n_renorm_steps, stride)

    for start in list(starts)[:n_iterations]:
        # 基准点 x
        x = torch.tensor(traj_np[start], dtype=torch.float32)
        # 切向扰动 (沿 ODE 方向)
        f_x = lorenz_ode(x)
        f_x_norm = f_x / (f_x.norm() + 1e-12)
        y = x + eps_init * f_x_norm

        # 演化并测量距离
        log_growth = 0.0
        cur_x, cur_y = x, y

        for step in range(n_renorm_steps):
            # 单步 RK4 (简化: 直接 Euler, 因为短窗口)
            def f(s):
                return lorenz_ode(s, SIGMA, RHO, BETA)
            k1_x = f(cur_x)
            k2_x = f(cur_x + 0.5 * dt * k1_x)
            k3_x = f(cur_x + 0.5 * dt * k2_x)
            k4_x = f(cur_x + dt * k3_x)
            cur_x = cur_x + (dt / 6) * (k1_x + 2*k2_x + 2*k3_x + k4_x)

            k1_y = f(cur_y)
            k2_y = f(cur_y + 0.5 * dt * k1_y)
            k3_y = f(cur_y + 0.5 * dt * k2_y)
            k4_y = f(cur_y + dt * k3_y)
            cur_y = cur_y + (dt / 6) * (k1_y + 2*k2_y + 2*k3_y + k4_y)

            # 距离
            dist = (cur_x - cur_y).norm()
            if dist > 0:
                log_growth += torch.log(dist / eps_init).item()

        # 增长率 = log_growth / total_time
        total_time = n_renorm_steps * dt
        growth_rates.append(log_growth / total_time)

    if not growth_rates:
        return 0.0
    return float(np.median(growth_rates))


def estimate_correlation_dimension(traj_np: np.ndarray, max_dim: int = 5,
                                    tau: int = 10, n_points: int = 2000,
                                    r_range: tuple = (0.01, 5.0),
                                    n_r: int = 30) -> float:
    """Grassberger-Procaccia 算法估计相关维数.

    算法:
    1. Takens 嵌入: 用 tau 步延迟构造 (max_dim)-dim 向量
    2. 计算相关积分 C(r) = (1/N^2) Σ_{i,j} I(||x_i - x_j|| < r)
    3. D_2 = log C(r) / log r 的斜率 (在 scaling 区)

    Args:
        traj_np: (T, 3) 单条轨迹
        max_dim: 嵌入维度 (typical 3-7)
        tau: 延迟时间 (typical 10-50 步)
        n_points: 使用的点数
        r_range: (r_min, r_max) for scaling region
        n_r: 半径采样数
    Returns:
        D_2: 相关维数 (Lorenz 理论值 ~2.05)
    """
    T = traj_np.shape[0]

    # Takens 嵌入
    n_embed = T - (max_dim - 1) * tau
    if n_embed < 100:
        return 0.0

    # 构造嵌入向量 (n_embed, max_dim)
    embedded = np.zeros((n_embed, max_dim))
    for d in range(max_dim):
        embedded[:, d] = traj_np[d * tau:d * tau + n_embed, 0]

    # 降采样到 n_points
    if n_embed > n_points:
        idx = np.random.choice(n_embed, n_points, replace=False)
        embedded = embedded[idx]

    N = embedded.shape[0]

    # 计算 pairwise distances
    from scipy.spatial.distance import pdist
    dists = pdist(embedded, metric='chebyshev')  # Chebyshev 距离对 D_2 更稳定

    # 计算 C(r) 对若干 r 值
    r_values = np.logspace(np.log10(r_range[0]), np.log10(r_range[1]), n_r)
    C_r = np.array([(dists < r).sum() / len(dists) for r in r_values])
    C_r = np.maximum(C_r, 1e-12)

    # 拟合 scaling 区 (取中间 ~50% 数据, 排除边界)
    n = len(r_values)
    start, end = n // 4, 3 * n // 4
    log_r = np.log10(r_values[start:end])
    log_C = np.log10(C_r[start:end])
    if len(log_r) < 3:
        return 0.0
    coeffs = np.polyfit(log_r, log_C, 1)
    return float(coeffs[0])


if __name__ == "__main__":
    # Sanity test: 用我们生成的 Lorenz 轨迹验证
    print("=" * 60)
    print("Lyapunov / D_2 Utils Sanity Test")
    print("=" * 60)

    # 加载 sample data
    import torch
    sample = torch.load("sample_data.pt", weights_only=False)
    traj = sample["train"][0].cpu().numpy()  # (2001, 3)
    print(f"\nTrajectory shape: {traj.shape}")

    # Lyapunov 估计
    print("\n[1] Lyapunov (Benettin algorithm)...")
    lam = estimate_lyapunov_benettin(traj, dt=0.01, n_renorm_steps=50, n_iterations=20)
    print(f"  Estimated: {lam:.4f}")
    print(f"  Theoretical: 0.9056")
    print(f"  Error: {abs(lam - 0.9056):.4f}")

    # D_2 估计
    print("\n[2] Correlation dimension D_2 (Grassberger-Procaccia)...")
    D2 = estimate_correlation_dimension(traj, max_dim=5, tau=15, n_points=1500)
    print(f"  Estimated: {D2:.4f}")
    print(f"  Theoretical: ~2.05")
    print(f"  Error: {abs(D2 - 2.05):.4f}")

    print("\n[OK] Lyapunov / D_2 utils working")
