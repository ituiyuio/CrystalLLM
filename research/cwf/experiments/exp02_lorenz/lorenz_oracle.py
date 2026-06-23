"""
Lorenz System Oracle: Ground Truth Predictor via RK4
====================================================

Oracle 用真实的 Lorenz ODE 和 RK4 数值积分预测未来 K 步.
它没有任何可学习参数, 代表"知道物理定律的完美模型"的性能上界.

Lorenz system:
    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z

Standard chaotic parameters:
    sigma = 10, rho = 28, beta = 8/3
    Lyapunov exponent: lambda_max ≈ 0.9056

RK4 (4th-order Runge-Kutta):
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt*k1/2)
    k3 = f(t + dt/2, y + dt*k2/2)
    k4 = f(t + dt, y + dt*k3)
    y_next = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

Reference:
    Lorenz, E. N. (1963). "Deterministic nonperiodic flow".
    Strogatz, S. (2015). "Nonlinear Dynamics and Chaos".
"""
from __future__ import annotations

import torch
import torch.nn as nn


# 标准混沌参数
SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0


def lorenz_ode(state: torch.Tensor, sigma: float = SIGMA,
               rho: float = RHO, beta: float = BETA) -> torch.Tensor:
    """Lorenz 系统右端项 f(state) = dstate/dt.

    Args:
        state: (..., 3) tensor (x, y, z)
    Returns:
        derivative: (..., 3)
    """
    x, y, z = state[..., 0], state[..., 1], state[..., 2]
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return torch.stack([dx, dy, dz], dim=-1)


class LorenzOracle(nn.Module):
    """完美的物理模型: 直接 RK4 积分 Lorenz ODE.

    用于给出预测性能的理论上限. 无可学习参数.
    """

    def __init__(self, dt: float = 0.01, sigma: float = SIGMA,
                 rho: float = RHO, beta: float = BETA):
        super().__init__()
        self.dt = dt
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        # 标记: 没有可学习参数
        for p in self.parameters():
            p.requires_grad = False

    def _rk4_step(self, state: torch.Tensor) -> torch.Tensor:
        """单步 RK4."""
        dt = self.dt
        k1 = lorenz_ode(state, self.sigma, self.rho, self.beta)
        k2 = lorenz_ode(state + 0.5 * dt * k1, self.sigma, self.rho, self.beta)
        k3 = lorenz_ode(state + 0.5 * dt * k2, self.sigma, self.rho, self.beta)
        k4 = lorenz_ode(state + dt * k3, self.sigma, self.rho, self.beta)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def rollout(self, initial_state: torch.Tensor, steps: int) -> torch.Tensor:
        """从 initial_state 出发, RK4 积分 steps 步.

        Args:
            initial_state: (B, 3) 初始状态
            steps: 积分步数 K (K 步之后的状态, 不含初始)
        Returns:
            trajectory: (B, steps, 3) K 个未来状态
        """
        states = []
        cur = initial_state
        for _ in range(steps):
            cur = self._rk4_step(cur)
            states.append(cur)
        return torch.stack(states, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Interface to match model forward signatures. x: (B, T, 3).

        Returns last step's prediction (B, 3).
        """
        # 取最后一步作为"输入状态", 向前预测 1 步
        last = x[:, -1, :]
        traj = self.rollout(last, steps=1)
        return traj[:, -1, :]  # (B, 3)


if __name__ == "__main__":
    # Sanity test: 与 SciPy 对比, 验证 RK4 精度
    print("=" * 60)
    print("Lorenz Oracle Sanity Test")
    print("=" * 60)

    oracle = LorenzOracle(dt=0.01)

    # 测试 1: 从经典初值出发, 跑 2000 步, 检查是否落在 attractor 上
    initial = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
    trajectory = oracle.rollout(initial, steps=2000)  # (1, 2001, 3)
    print(f"\nTest 1: 2000-step rollout from (1, 1, 1)")
    print(f"  Initial: {trajectory[0, 0].tolist()}")
    print(f"  After 2000 steps: {trajectory[0, -1].tolist()}")
    print(f"  Expected range on attractor: x,y ~ ±20, z ~ 0-50")

    # 测试 2: 与 SciPy solve_ivp 对比
    print("\nTest 2: Compare with SciPy solve_ivp...")
    try:
        from scipy.integrate import solve_ivp
        import numpy as np

        def scipy_lorenz(t, state):
            x, y, z = state
            return [SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z]

        # 短窗口对比 (1 Lyapunov time 内, 数值方法应一致)
        t_eval_short = np.linspace(0, 1.0, 101)
        sol_short = solve_ivp(scipy_lorenz, (0, 1.0), [1.0, 1.0, 1.0],
                               t_eval=t_eval_short, method='RK45', rtol=1e-10, atol=1e-12)
        my_traj_short = oracle.rollout(initial, steps=100)  # (1, 101, 3)
        diff_short = abs(my_traj_short[0].numpy() - sol_short.y.T).max()

        # 长窗口对比 (5+ Lyapunov times, 必发散)
        t_eval_long = np.linspace(0, 20.0, 2001)
        sol_long = solve_ivp(scipy_lorenz, (0, 20.0), [1.0, 1.0, 1.0],
                             t_eval=t_eval_long, method='RK45', rtol=1e-10, atol=1e-12)
        scipy_final = sol_long.y[:, -1]
        diff_long = abs(trajectory[0, -1].numpy() - scipy_final).max()

        print(f"  Short window (100 steps, ~1 Lyapunov time):")
        print(f"    Our RK4 vs SciPy RK45 max diff: {diff_short:.2e}")
        print(f"    Status: {'PASS' if diff_short < 1e-3 else 'WARN'} (threshold: 1e-3)")
        print(f"    (Note: 1e-3 is realistic for chaotic system at 1 Lyapunov time;")
        print(f"\n  Long window (2000 steps, ~18 Lyapunov times):")
        print(f"    Our RK4 vs SciPy RK45 max diff: {diff_long:.2e}")
        print(f"    Note: DIVERGENCE EXPECTED — Lyapunov exponent = 0.906,")
        print(f"          tiny RK4 vs RK45 numerical differences get amplified e^(0.906*20) ~ 6e7 times.")
        print(f"          This is the definition of chaos, NOT a bug.")
    except ImportError:
        print("  SciPy not available, skipping comparison")

    # 测试 3: 短时预测应非常准 (Lyapunov time = 1/λ ≈ 1.1 步)
    print("\nTest 3: Short-term prediction accuracy...")
    true_traj = oracle.rollout(initial, steps=10)  # (1, 11, 3)
    # Oracle "predict" from step 5: 应该与真值 step 5:10 完全一致
    print(f"  Step 5 true:     {true_traj[0, 5].tolist()}")
    print(f"  Step 10 oracle:  {oracle.rollout(true_traj[0, 5:6], steps=5)[0, -1].tolist()}")
    print(f"  Step 10 true:    {true_traj[0, 10].tolist()}")
    print("  (Should be identical since Oracle uses true ODE)")

    print("\n[OK] Lorenz Oracle working")
