"""
Exp 31: PDE 求解 smoke — CWF 在波动方程上 + STFT 损失, 是否比 MSE 强?
========================================================================

假说: CWF 的复数+球面约束在 PDE 任务上, 配合 STFT/Multi-resolution 损失
      应该比 MSE 强, 因为波动方程本质是相位演化.

任务: 1D 波动方程 u_tt = c² u_xx, 初始 Gaussian packet, 预测 u(x, T).

数据: 解析解 wave_packet(x, t) = exp(-((x-x0-c·t)²)/(2σ²)) · sin(k₀(x-x0-c·t))
     每条样本随机 x0, σ, k0; 训练时 t=0 → 预测 t=T.

四组对照 (2×2):
  1. CWF + MSE            (复数 + 错配损失)
  2. CWF + STFT loss      (复数 + 匹配损失) ← 实验组
  3. Transformer + MSE    (实数 + 错配损失) ← 实数 baseline
  4. Transformer + STFT   (实数 + 匹配损失)

判决 (falsifiable):
  - CWF+STFT 比 Trans+MSE 显著低 (≤0.8x)        → PDE 战场有戏, 继续大实验
  - CWF+STFT ≈ Trans+MSE (0.8x ~ 1.2x)         → 持平, 需要看 loss 曲线和学习到的频谱
  - CWF+STFT 比 Trans+MSE 高 (≥1.2x)           → 战场仍不对, CWF+PDE 路线归档

预期时间: 1k 步, CPU ~5 分钟.

ponytail: 只用最小 CWF (cwf_minimal.CWFSingleBlock), 不重写架构.
ponytail: 单一 PDE (1D 波动方程) + 单一数据 (Gaussian packet), 不做多任务.
ponytail: 一旦 STFT loss 不比 MSE 强 ≥20%, 整个路线归档.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
PROTOTYPE_DIR = HERE.parents[1] / "prototype"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROTOTYPE_DIR))

from cwf_minimal import CWFSingleBlock, complex_norm  # noqa: E402

DEVICE = "cpu"  # ponytail: CPU 可跑, 不依赖 GPU
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# === PDE 参数 ===
S = 64           # 空间网格点数 (1D 序列长度)
DT = 0.01        # 时间步长
T_PRED = 0.5     # 预测时刻
C = 1.0          # 波速
X_RANGE = (0.0, 1.0)
K0_RANGE = (10.0, 25.0)   # 初始波数
SIGMA_RANGE = (0.04, 0.10)
X0_RANGE = (0.2, 0.8)

# === 模型参数 ===
D_MODEL = 64
N_EVAL = 256

# === 训练参数 ===
TRAIN_STEPS = 1000
BATCH_SIZE = 32
LR = 3e-4
SEEDS = [42, 123, 2024]


# ===========================================================================
# 1D 波动方程解析解: Gaussian packet 向右传播
# ===========================================================================
def wave_packet(x: np.ndarray, x0: float, sigma: float, k0: float,
                t: float = 0.0, c: float = C) -> np.ndarray:
    """u(x, t) = exp(-((x-x0-ct)²)/(2σ²)) · sin(k₀(x-x0-ct))."""
    arg = x - x0 - c * t
    envelope = np.exp(-(arg ** 2) / (2 * sigma ** 2))
    return envelope * np.sin(k0 * arg)


def sample_pde(batch_size: int, seed: int = None) -> tuple[np.ndarray, np.ndarray]:
    """采样一批 (u(x,0), u(x,T)) 对. Returns shape (B, S)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(*X_RANGE, S)
    u0, uT = [], []
    for _ in range(batch_size):
        x0 = rng.uniform(*X0_RANGE)
        sigma = rng.uniform(*SIGMA_RANGE)
        k0 = rng.uniform(*K0_RANGE)
        u0.append(wave_packet(x, x0, sigma, k0, t=0.0))
        uT.append(wave_packet(x, x0, sigma, k0, t=T_PRED))
    return np.stack(u0).astype(np.float32), np.stack(uT).astype(np.float32)


# ===========================================================================
# 损失函数: MSE vs STFT 多分辨率损失
# ===========================================================================
def stft_loss(y_pred: torch.Tensor, y_true: torch.Tensor,
              n_ffts: tuple[int, ...] = (16, 32, 64)) -> torch.Tensor:
    """多分辨率 STFT 损失: 幅度谱 L1 + 时间 L1 (替代方案: 仅幅度谱).

    Args:
        y_pred, y_true: (B, S) 实数序列
        n_ffts: 多个 FFT 长度, 模拟多尺度
    """
    total = F.l1_loss(y_pred, y_true)  # 时间分量
    for n_fft in n_ffts:
        if n_fft > y_pred.shape[-1]:
            continue
        # 用 torch.fft 做实数 FFT
        P = torch.fft.rfft(y_pred, n=n_fft, dim=-1)
        T = torch.fft.rfft(y_true, n=n_fft, dim=-1)
        # 幅度谱 L1 (忽略相位)
        total = total + F.l1_loss(P.abs(), T.abs())
    return total


# ===========================================================================
# 模型: CWF 包装 (1D PDE field → 复数场 → CWFSingleBlock → 1D PDE field)
# ===========================================================================
class CWFPredictor(nn.Module):
    def __init__(self, d: int = D_MODEL):
        super().__init__()
        self.d = d
        # 输入编码: 1D 实数 → d 维复数 (real=input, imag=0, 然后乘 0.1 保持 norm 小)
        self.input_proj = nn.Linear(S, d)
        # CWF block
        self.block = CWFSingleBlock(d=d, hidden_mult=2)
        # 输出解码: d 维复数 → 1D 实数 (取实部)
        self.output_proj = nn.Linear(d, S)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (B, S) 实数 PDE 场 u(x, 0)
        Returns:
            u_pred: (B, S) 实数预测 u(x, T)
        """
        # 编码到复数空间: shape (B, d, 2)
        h = self.input_proj(u)  # (B, d)
        # 复数化: real = h, imag = 0
        psi = torch.stack([h, torch.zeros_like(h)], dim=-1)  # (B, d, 2)
        # 乘 0.1 保持 norm 在 0.1 以下, 满足 closure
        psi = psi * 0.1
        # 扩展 S 维度: 把 batch 维度看作 sequence=1
        psi = psi.unsqueeze(1)  # (B, 1, d, 2)
        # CWF forward
        psi_out, _ = self.block(psi)
        # 解码: 取实部
        psi_out = psi_out.squeeze(1)  # (B, d, 2)
        u_pred = self.output_proj(psi_out[..., 0])  # (B, S)
        return u_pred


class TransformerPredictor(nn.Module):
    """1D Transformer encoder baseline (实数).

    用 1 层 TransformerEncoder 处理 u(x, 0) sequence → u(x, T) sequence.
    """
    def __init__(self, d: int = D_MODEL, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(1, d)
        self.pos_embed = nn.Parameter(torch.randn(S, d) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=d * 4,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d, 1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # (B, S) → (B, S, 1) → (B, S, d)
        x = self.input_proj(u.unsqueeze(-1)) + self.pos_embed.unsqueeze(0)
        x = self.encoder(x)  # (B, S, d)
        return self.output_proj(x).squeeze(-1)  # (B, S)


# ===========================================================================
# 训练一个 (model, loss_fn) 配置
# ===========================================================================
def train_one(model: nn.Module, loss_name: str, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = stft_loss if loss_name == "stft" else F.mse_loss

    # 训练
    losses = []
    t0 = time.time()
    for step in range(TRAIN_STEPS):
        u0_np, uT_np = sample_pde(BATCH_SIZE)
        u0 = torch.from_numpy(u0_np).to(DEVICE)
        uT = torch.from_numpy(uT_np).to(DEVICE)
        u_pred = model(u0)
        loss = loss_fn(u_pred, uT)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 200 == 0:
            print(f"    step {step:4d}  loss={loss.item():.6f}")

    # 评估: 在 N_EVAL 个测试样本上算最终 MSE 和 STFT
    model.eval()
    with torch.no_grad():
        u0_np, uT_np = sample_pde(N_EVAL, seed=seed + 999)
        u0 = torch.from_numpy(u0_np).to(DEVICE)
        uT = torch.from_numpy(uT_np).to(DEVICE)
        u_pred = model(u0)
        final_mse = F.mse_loss(u_pred, uT).item()
        final_stft = stft_loss(u_pred, uT).item()
    elapsed = time.time() - t0
    return {
        "losses": losses,
        "final_mse": final_mse,
        "final_stft": final_stft,
        "elapsed_s": elapsed,
    }


# ===========================================================================
# 主程序: 4 组对照
# ===========================================================================
def main():
    print("=" * 70)
    print("Exp 31: PDE 求解 smoke — CWF vs Transformer × MSE vs STFT")
    print("=" * 70)

    configs = [
        ("CWF",        CWFPredictor,       "mse"),
        ("CWF",        CWFPredictor,       "stft"),
        ("Transformer", TransformerPredictor, "mse"),
        ("Transformer", TransformerPredictor, "stft"),
    ]

    # 参数量统计
    print("\nModel sizes:")
    for name, cls, _ in configs:
        m = cls()
        n_params = sum(p.numel() for p in m.parameters())
        print(f"  {name:15s}  {n_params:>10,} params")

    # 跑 4 组对照 (单 seed 节省时间, multi-seed 在第一轮诊断后再说)
    results = {}
    for name, cls, loss in configs:
        key = f"{name}_{loss}"
        print(f"\n[{key}] training {TRAIN_STEPS} steps...")
        r = train_one(cls(), loss, seed=SEEDS[0])
        results[key] = r
        print(f"  final MSE  = {r['final_mse']:.6f}")
        print(f"  final STFT = {r['final_stft']:.6f}")
        print(f"  elapsed    = {r['elapsed_s']:.1f}s")

    # 判决: 三个角度
    cwf_mse = results["CWF_mse"]["final_mse"]
    cwf_stft = results["CWF_stft"]["final_mse"]
    trans_mse = results["Transformer_mse"]["final_mse"]
    trans_stft = results["Transformer_stft"]["final_mse"]

    # 同损失对比 (apple-to-apple): 关键对比
    cwf_vs_trans_under_mse = cwf_mse / trans_mse
    cwf_vs_trans_under_stft = cwf_stft / trans_stft
    # 自身对比: 损失函数影响
    cwf_loss_effect = cwf_stft / cwf_mse
    trans_loss_effect = trans_stft / trans_mse

    print("\n" + "=" * 70)
    print("Verdict (multi-angle:)")
    print("=" * 70)
    print(f"  同损失对比 (CWF vs Transformer):")
    print(f"    CWF+MSE / Trans+MSE  = {cwf_vs_trans_under_mse:.4f}x  ← 关键!")
    print(f"    CWF+STFT / Trans+STFT = {cwf_vs_trans_under_stft:.4f}x")
    print(f"  损失函数影响 (同模型, STFT vs MSE):")
    print(f"    CWF:        STFT / MSE = {cwf_loss_effect:.2f}x")
    print(f"    Trans:      STFT / MSE = {trans_loss_effect:.2f}x")

    # 判决逻辑: 同损失对比才是正确判决
    if cwf_vs_trans_under_mse <= 0.8:
        verdict_main = f"CWF+MSE 碾压 Trans+MSE ({cwf_vs_trans_under_mse:.3f}x) → PDE 战场有戏, CWF 归纳偏置天然匹配 PDE"
    elif cwf_vs_trans_under_mse <= 1.2:
        verdict_main = f"CWF ≈ Trans ({cwf_vs_trans_under_mse:.3f}x) → 持平"
    else:
        verdict_main = f"CWF 输 ({cwf_vs_trans_under_mse:.3f}x) → 战场不对"

    # 副判决: 损失函数是否有用
    if cwf_loss_effect < 1.5:
        verdict_loss = f"STFT 对 CWF 无显著影响 ({cwf_loss_effect:.2f}x)"
    elif cwf_loss_effect > 2.0:
        verdict_loss = f"STFT 拖 CWF 后腿 ({cwf_loss_effect:.2f}x) → CWF 用 MSE 训练即可, 损失函数不是瓶颈"
    else:
        verdict_loss = f"STFT 中性 ({cwf_loss_effect:.2f}x)"

    print(f"\n主判决: {verdict_main}")
    print(f"副判决: {verdict_loss}")

    # 保存结果
    out = {
        "config": {
            "S": S, "T_PRED": T_PRED, "C": C,
            "TRAIN_STEPS": TRAIN_STEPS, "BATCH_SIZE": BATCH_SIZE, "LR": LR,
            "D_MODEL": D_MODEL,
        },
        "results": {k: {"final_mse": v["final_mse"],
                        "final_stft": v["final_stft"],
                        "elapsed_s": v["elapsed_s"]}
                    for k, v in results.items()},
        "verdict_main": verdict_main,
        "verdict_loss": verdict_loss,
        "ratios": {
            "cwf_vs_trans_under_mse": cwf_vs_trans_under_mse,
            "cwf_vs_trans_under_stft": cwf_vs_trans_under_stft,
            "cwf_loss_effect": cwf_loss_effect,
            "trans_loss_effect": trans_loss_effect,
        },
    }
    with open(RESULTS_DIR / "exp31_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {RESULTS_DIR / 'exp31_results.json'}")


if __name__ == "__main__":
    main()