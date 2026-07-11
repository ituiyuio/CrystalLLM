"""
Exp 29: 复数动力系统探针 - 相位在已知因果数据中能否被学到?
===============================================================

exp28 证明: 文本波场的相位不携带可预测的时间信息. 但这可能是数据问题
(文本不是物理波) 或架构问题 (modReLU 等变约束), 两者叠加无法区分.

本实验用**已知有相位因果的合成动力系统**做数据, 一次性拆开两个变量:

数据 (2 种, 修复数据层):
  linear:    dz/dt = Az, A 是复数矩阵. 相位有因果但演化简单.
  stuart:    dz/dt = (σ+iω)z - (1+ic)|z|²z, Stuart-Landau 方程.
             相位有因果且非线性耦合 (极限环).

架构 (3 种, 测试架构层):
  real:      cat[Re,Im] -> Linear -> GELU -> Linear (非等变, 可自由混合 Re/Im)
  modrelu:   ComplexLinear -> modReLU -> ComplexLinear (等变, exp28 的架构)
  siren:     ComplexLinear -> sin(z) -> ComplexLinear (全纯, 非等变, 相位敏感)

6 条件 x 3 seeds x 5000 steps.

判决矩阵:
  Siren 赢 -> 相位有用 + 非等变是关键 -> 波场世界模型有活路
  Real 赢 -> 即使数据有相位因果, 实数拆分也能覆盖 -> 相位无用是表示论层面的
  modReLU 输但 Siren 赢 -> 确认 exp23 三难困境是架构问题, 不是原理问题
  全输 baseline -> 合成 ODE 轨迹太难学, 探针无效
"""
from __future__ import annotations

import argparse
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
EXP05_DIR = HERE.parent / "exp05_wave_autoencoder"
PROJECT_ROOT = HERE.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXP05_DIR))

from wave_autoencoder import grad_norm  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

D = 32              # 复数状态维度
TRAJ_LEN = 64       # 轨迹长度 (时间步)
TRAIN_STEPS = 5000
BATCH_SIZE = 64
LR = 1e-3
WD = 0.01
WARMUP_STEPS = 100
EVAL_STEPS = [200, 500, 1000, 2000, 3000, 5000]
SEEDS_DEFAULT = [42, 123, 2024]
PRED_FRAC = 0.5    # 前 50% 做输入, 后 50% 做预测目标

# Siren omega_0
SIREN_W0 = 1.0


# ============================================================================
# 合成动力系统数据生成
# ============================================================================
def gen_linear_traj(batch_size, d=D, traj_len=TRAJ_LEN, seed=None):
    """线性复数 ODE: dz/dt = Az.
    A = 随机复数矩阵, 特征值实部接近 0 (纯旋转, 不爆不衰减).
    相位有因果: A 的特征值的虚部决定旋转速度, 相位直接由 A 决定.
    """
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
    else:
        gen = None

    # 纯虚数矩阵: A = i*B, B 实数. e^{iBt} 是酉旋转, 不爆不衰减.
    B = torch.randn(d, d, generator=gen) * 0.3 / math.sqrt(d)
    A = torch.complex(torch.zeros_like(B), B)  # 纯虚

    # 随机初始状态 (单位范数)
    z0_real = torch.randn(batch_size, d, generator=gen) * 0.3
    z0_imag = torch.randn(batch_size, d, generator=gen) * 0.3
    z0 = torch.complex(z0_real, z0_imag)

    # 解析解: z(t) = e^{At} z0, 用矩阵指数 (torch.linalg.matrix_exp)
    dt = 0.2
    expAdt = torch.linalg.matrix_exp(A * dt)  # (d, d) cfloat
    traj = [z0]
    z = z0
    for t in range(traj_len - 1):
        z = z @ expAdt.t()  # batch matmul
        traj.append(z)
    traj = torch.stack(traj, dim=1)  # (B, T, d) cfloat
    return traj


def gen_stuart_landau(batch_size, d=D, traj_len=TRAJ_LEN, seed=None):
    """Stuart-Landau 方程: dz_i/dt = (σ+iω)z_i - (1+ic)|z_i|²z_i.

    每个维度独立演化 (解耦), 但非线性相位耦合:
    - σ 控制幅度增长/衰减 (趋向极限环 |z|=sqrt(σ))
    - ω 控制线性旋转
    - c 控制非线性相位耦合 (幅度影响相位速度)

    这是极限环振荡器, 相位有真实的因果作用.
    """
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
    else:
        gen = None

    # 参数: 每个维度不同的 σ, ω, c
    sigma = torch.rand(d, generator=gen) * 0.5 + 0.5   # [0.5, 1.0]
    omega = torch.rand(d, generator=gen) * 3.0 + 0.5   # [0.5, 3.5]
    c_param = torch.rand(d, generator=gen) * 2.0 - 1.0 # [-1, 1]

    # 初始状态: 接近极限环但有扰动
    r0 = torch.sqrt(sigma + 1e-6)
    z0 = torch.complex(
        r0.unsqueeze(0).expand(batch_size, -1) * (1 + 0.1 * torch.randn(batch_size, d, generator=gen)),
        0.1 * torch.randn(batch_size, d, generator=gen)
    )

    dt = 0.05
    traj = [z0]
    z = z0
    for t in range(traj_len - 1):
        # dz/dt = (σ+iω)z - (1+ic)|z|²z
        z_abs_sq = z.abs().pow(2)
        linear_term = torch.complex(sigma, omega) * z  # broadcast
        nonlinear_term = (1 + 1j * c_param) * z_abs_sq * z
        dz = linear_term - nonlinear_term
        z = z + dz * dt  # Euler step
        traj.append(z)
    traj = torch.stack(traj, dim=1)  # (B, T, d) cfloat
    return traj


def generate_dataset(system="linear", n_samples=4096, seed=42):
    """生成训练和验证数据."""
    gen = torch.Generator().manual_seed(seed)
    n_train = int(n_samples * 0.8)
    n_val = n_samples - n_train

    if system == "linear":
        gen_fn = gen_linear_traj
    elif system == "stuart":
        gen_fn = gen_stuart_landau
    else:
        raise ValueError(f"unknown system: {system}")

    # 用不同 seed 生成不同 batch, 保证多样性
    train_trajs = []
    for i in range(n_train // BATCH_SIZE):
        t = gen_fn(BATCH_SIZE, seed=seed + i)
        train_trajs.append(t)
    val_trajs = []
    for i in range(n_val // BATCH_SIZE):
        t = gen_fn(BATCH_SIZE, seed=seed + 10000 + i)
        val_trajs.append(t)

    train_data = torch.cat(train_trajs, dim=0)  # (n_train, T, d)
    val_data = torch.cat(val_trajs, dim=0)
    return train_data.to(DEVICE), val_data.to(DEVICE)


def get_batch(trajs, bs=BATCH_SIZE):
    """随机采样 batch."""
    n = trajs.shape[0]
    idx = torch.randint(0, n, (bs,))
    return trajs[idx]


# ============================================================================
# 演化器架构
# ============================================================================
class RealEvolver(nn.Module):
    """实数 MLP: cat[Re, Im] -> Linear -> GELU -> Linear -> split.
    非等变, 可自由混合 Re/Im."""
    def __init__(self, d=D):
        super().__init__()
        self.fc1 = nn.Linear(2 * d, 4 * d)
        self.fc2 = nn.Linear(4 * d, 2 * d)
        self.d = d

    def forward(self, z):
        B, T, D = z.shape
        x = torch.cat([z.real, z.imag], dim=-1)  # (B, T, 2d)
        h = F.gelu(self.fc1(x))
        out = self.fc2(h)  # (B, T, 2d)
        re, im = out.chunk(2, dim=-1)
        return torch.complex(re, im)


class ComplexModReLUEvolver(nn.Module):
    """复数 MLP (modReLU): ComplexLinear -> modReLU -> ComplexLinear.
    等变, 相位是自由乘客 (exp28 的架构)."""
    def __init__(self, d=D):
        super().__init__()
        scale1 = 1.0 / math.sqrt(d)
        scale2 = 1.0 / math.sqrt(4 * d)
        self.W1 = nn.Parameter(torch.randn(4 * d, d, dtype=torch.complex64) * scale1)
        self.W2 = nn.Parameter(torch.randn(d, 4 * d, dtype=torch.complex64) * scale2)
        self.b1 = nn.Parameter(torch.zeros(4 * d, dtype=torch.complex64))
        self.b2 = nn.Parameter(torch.zeros(d, dtype=torch.complex64))

    def forward(self, z):
        # z: (B, T, d) cfloat
        h = z @ self.W1.t() + self.b1  # (B, T, 4d)
        # modReLU: tanh(|z|) * z / |z|
        mag = h.abs()
        phase = h / torch.clamp(mag, min=1e-8)
        h = torch.tanh(mag) * phase
        out = h @ self.W2.t() + self.b2  # (B, T, d)
        return out


class ComplexSirenEvolver(nn.Module):
    """复数 MLP (Siren): ComplexLinear -> sin(z) -> ComplexLinear.
    全纯, 非等变, 相位敏感."""
    def __init__(self, d=D):
        super().__init__()
        # Siren init: uniform(±sqrt(6/fan_in)/omega_0)
        fan_in = d
        bound1 = math.sqrt(6.0 / fan_in) / SIREN_W0
        bound2 = math.sqrt(6.0 / (4 * d))
        W1_real = (torch.rand(4 * d, d) * 2 - 1) * bound1
        W1_imag = (torch.rand(4 * d, d) * 2 - 1) * bound1
        W2_real = (torch.rand(d, 4 * d) * 2 - 1) * bound2
        W2_imag = (torch.rand(d, 4 * d) * 2 - 1) * bound2
        self.W1 = nn.Parameter(torch.complex(W1_real, W1_imag))
        self.W2 = nn.Parameter(torch.complex(W2_real, W2_imag))
        self.b1 = nn.Parameter(torch.zeros(4 * d, dtype=torch.complex64))
        self.b2 = nn.Parameter(torch.zeros(d, dtype=torch.complex64))

    def forward(self, z):
        # z: (B, T, d) cfloat
        h = z @ self.W1.t() + self.b1  # (B, T, 4d)
        # 裁剪虚部防止 cosh/sinh 爆炸
        h = torch.complex(h.real, torch.clamp(h.imag, -10.0, 10.0))
        # Siren: sin(z) = sin(re)cosh(im) + i*cos(re)sinh(im)
        h = torch.sin(h)
        out = h @ self.W2.t() + self.b2  # (B, T, d)
        return out


# ============================================================================
# CONFIGS: 2 data x 3 arch = 6 conditions
# ============================================================================
ARCHS = {
    "real":    RealEvolver,
    "modrelu": ComplexModReLUEvolver,
    "siren":   ComplexSirenEvolver,
}

SYSTEMS = ["linear", "stuart"]

def list_configs():
    configs = []
    for sys_name in SYSTEMS:
        for arch_name in ARCHS:
            configs.append(f"{sys_name}_{arch_name}")
    return configs

CONFIGS = {c: c for c in list_configs()}


def build_model(config_name):
    parts = config_name.split("_", 1)
    sys_name, arch_name = parts[0], parts[1]
    return ARCHS[arch_name](d=D), sys_name


# ============================================================================
# 诊断
# ============================================================================
@torch.no_grad()
def compute_pred_mse(pred, target):
    """复数 L2: |pred - target|^2."""
    diff = pred - target
    return (diff.real.pow(2) + diff.imag.pow(2)).mean().item()


@torch.no_grad()
def compute_phase_alignment(pred, target):
    """|cos(Δφ)| = |<pred,target>| / (|pred||target|). 0=随机, 1=对齐."""
    inner = (pred.conj() * target).sum().abs()
    norm_prod = pred.abs().sum() * target.abs().sum() + 1e-8
    return (inner / norm_prod).item()


@torch.no_grad()
def compute_mag_ratio(pred, target):
    """|pred|/|target|."""
    return (pred.abs().mean() / (target.abs().mean() + 1e-8)).item()


@torch.no_grad()
def eval_model(model, val_data, pred_len=None):
    """评估: 单步预测 z_t -> z_{t+1} 的平均误差 (非自回归, 避免 error accumulation)."""
    model.eval()

    n_batches = min(10, val_data.shape[0] // BATCH_SIZE)
    mses, aligns, ratios = [], [], []

    for i in range(n_batches):
        batch = val_data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        # 单步预测: z_t -> pred z_{t+1}, 对所有 t
        z_in = batch[:, :-1]    # (B, T-1, d)
        z_tgt = batch[:, 1:]    # (B, T-1, d)
        pred = model(z_in)      # (B, T-1, d)

        mses.append(compute_pred_mse(pred, z_tgt))
        aligns.append(compute_phase_alignment(pred, z_tgt))
        ratios.append(compute_mag_ratio(pred, z_tgt))

    model.train()
    return {
        "pred_mse": round(sum(mses) / len(mses), 6),
        "phase_alignment": round(sum(aligns) / len(aligns), 6),
        "mag_ratio": round(sum(ratios) / len(ratios), 6),
    }


# ============================================================================
# run_one
# ============================================================================
def run_one(config_name, seed, steps=TRAIN_STEPS):
    tag = f"{config_name}_s{seed}"
    out_json = RESULTS_DIR / f"{tag}.json"
    if out_json.exists():
        d = json.load(open(out_json))
        if d.get("train_steps") == steps and not d.get("diverged"):
            print(f"[skip] {tag} already done")
            return d

    model, sys_name = build_model(config_name)
    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*70}\n[{tag}] system={sys_name}  seed={seed}  steps={steps}\n{'='*70}")
    print(f"[model] {config_name}  params: {n_params:,}")

    # Siren 需要更低的 lr 防止发散. peak_lr 同时驱动 opt 初值和 warmup 终点.
    peak_lr = LR * 0.1 if "siren" in config_name else LR
    opt = torch.optim.AdamW(
        list(p for p in model.parameters() if p.requires_grad),
        lr=peak_lr, weight_decay=WD, betas=(0.9, 0.95))

    train_data, val_data = generate_dataset(system=sys_name, seed=42)

    results = {
        "config": config_name, "seed": seed, "params": n_params,
        "system": sys_name,
        "train_steps": steps, "trace": [],
        "nan_step": None, "diverged": False,
    }
    t0 = time.time()
    input_len = TRAJ_LEN - int(TRAJ_LEN * PRED_FRAC)

    for step in range(1, steps + 1):
        if step <= WARMUP_STEPS:
            lr = peak_lr * step / WARMUP_STEPS
            for g in opt.param_groups:
                g["lr"] = lr

        # 采样 batch, 训练: 给定 z[0..T/2], 预测 z[T/2..T]
        # 单步训练: z_t -> z_{t+1} for t in [0, T/2)
        batch = get_batch(train_data)
        z_in = batch[:, :input_len]   # (B, T_in, d)
        z_tgt = batch[:, 1:input_len + 1]  # shifted by 1

        # 单步预测损失: z_t -> pred z_{t+1}
        pred = model(z_in)  # (B, T_in, d)
        diff = pred - z_tgt
        loss = (diff.real.pow(2) + diff.imag.pow(2)).mean()

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  !!! NaN at step {step}")
            results["nan_step"] = step
            results["diverged"] = True
            break
        opt.zero_grad()
        loss.backward()
        gn = grad_norm(model)
        if math.isnan(gn) or math.isinf(gn):
            print(f"  !!! grad NaN at step {step}")
            results["nan_step"] = step
            results["diverged"] = True
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 250 == 0 or step == 100:
            print(f"  step {step:>4}/{steps}  loss={loss.item():.6f}  "
                  f"|g|={gn:.2e}  t={time.time()-t0:.0f}s", flush=True)
        if step in EVAL_STEPS:
            em = eval_model(model, val_data)
            results["trace"].append({
                "step": step,
                "train_loss": round(loss.item(), 6),
                "pred_mse": em["pred_mse"],
                "phase_alignment": em["phase_alignment"],
                "mag_ratio": em["mag_ratio"],
                "grad_norm": round(gn, 4),
            })
            print(f"  >>> pred_mse={em['pred_mse']:.6f}  "
                  f"align={em['phase_alignment']:.4f}  "
                  f"mag_ratio={em['mag_ratio']:.4f}", flush=True)

    results["total_time_s"] = round(time.time() - t0, 2)
    final = results["trace"][-1] if results["trace"] else {}
    results["final_eval"] = final
    results["best_pred_mse"] = min(
        (t["pred_mse"] for t in results["trace"]), default=None)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[saved] -> {out_json}  (best_pred_mse={results['best_pred_mse']})")
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return results


# ============================================================================
# compute_verdict
# ============================================================================
def compute_verdict(seeds=tuple(SEEDS_DEFAULT)):
    print("\n" + "=" * 70)
    print("VERDICT: 复数动力系统探针 - 相位在已知因果数据中能否被学到?")
    print("=" * 70)

    configs = list(CONFIGS.keys())
    mses = {c: [] for c in configs}
    aligns = {c: [] for c in configs}
    ratios = {c: [] for c in configs}

    for c in configs:
        for seed in seeds:
            p = RESULTS_DIR / f"{c}_s{seed}.json"
            if not p.exists():
                mses[c].append(None)
                aligns[c].append(None)
                ratios[c].append(None)
                continue
            d = json.load(open(p))
            mses[c].append(d.get("best_pred_mse"))
            fe = d.get("final_eval", {})
            if isinstance(fe, dict):
                aligns[c].append(fe.get("phase_alignment"))
                ratios[c].append(fe.get("mag_ratio"))
            else:
                aligns[c].append(None)
                ratios[c].append(None)

    print("\n--- 预测 MSE (lower=better) ---")
    means_mse = {}
    for c in configs:
        vals = [v for v in mses[c] if v is not None]
        if vals:
            m = sum(vals) / len(vals)
            means_mse[c] = m
            print(f"  {c:20s}: mean={m:.6f}  (seeds: {[round(v,6) if v else None for v in mses[c]]})")

    print("\n--- phase_alignment (0=随机, 1=对齐) ---")
    for c in configs:
        vals = [v for v in aligns[c] if v is not None]
        if vals:
            m = sum(vals) / len(vals)
            print(f"  {c:20s}: mean={m:.6f}")

    print("\n--- mag_ratio (1=匹配, <<1=压幅度, >>1=放幅度) ---")
    for c in configs:
        vals = [v for v in ratios[c] if v is not None]
        if vals:
            m = sum(vals) / len(vals)
            print(f"  {c:20s}: mean={m:.6f}")

    print("\n--- 判决矩阵 ---")
    per_condition = {}

    for sys_name in SYSTEMS:
        real_key = f"{sys_name}_real"
        modrelu_key = f"{sys_name}_modrelu"
        siren_key = f"{sys_name}_siren"

        r_mse = means_mse.get(real_key)
        m_mse = means_mse.get(modrelu_key)
        s_mse = means_mse.get(siren_key)

        print(f"\n  [{sys_name}]")

        # Siren vs Real: 相位有用吗?
        if s_mse and r_mse:
            ratio = s_mse / r_mse
            if ratio < 0.9:
                per_condition[f"{sys_name}_siren_vs_real"] = "SIREN_BETTER"
                print(f"    Siren vs Real: {s_mse:.6f} / {r_mse:.6f} = {ratio:.3f}  -> 复数(Siren)赢! 相位有用!")
            elif ratio > 1.1:
                per_condition[f"{sys_name}_siren_vs_real"] = "REAL_BETTER"
                print(f"    Siren vs Real: {s_mse:.6f} / {r_mse:.6f} = {ratio:.3f}  -> 实数赢, 相位无用")
            else:
                per_condition[f"{sys_name}_siren_vs_real"] = "NEUTRAL"
                print(f"    Siren vs Real: {s_mse:.6f} / {r_mse:.6f} = {ratio:.3f}  -> 中性")

        # modReLU vs Real: 等变约束有害吗?
        if m_mse and r_mse:
            ratio = m_mse / r_mse
            if ratio > 1.1:
                per_condition[f"{sys_name}_modrelu_vs_real"] = "MODRELU_WORSE"
                print(f"    modReLU vs Real: {m_mse:.6f} / {r_mse:.6f} = {ratio:.3f}  -> 等变约束有害 (三难困境确认)")
            elif ratio < 0.9:
                per_condition[f"{sys_name}_modrelu_vs_real"] = "MODRELU_BETTER"
                print(f"    modReLU vs Real: {m_mse:.6f} / {r_mse:.6f} = {ratio:.3f}  -> modReLU赢?!")
            else:
                per_condition[f"{sys_name}_modrelu_vs_real"] = "NEUTRAL"
                print(f"    modReLU vs Real: {m_mse:.6f} / {r_mse:.6f} = {ratio:.3f}  -> 中性")

        # Siren vs modReLU: 非等变是关键吗?
        if s_mse and m_mse:
            ratio = s_mse / m_mse
            if ratio < 0.9:
                per_condition[f"{sys_name}_siren_vs_modrelu"] = "SIREN_UNLOCKS"
                print(f"    Siren vs modReLU: {s_mse:.6f} / {m_mse:.6f} = {ratio:.3f}  -> 非等变解锁相位!")
            elif ratio > 1.1:
                per_condition[f"{sys_name}_siren_vs_modrelu"] = "SIREN_WORSE"
                print(f"    Siren vs modReLU: {s_mse:.6f} / {m_mse:.6f} = {ratio:.3f}  -> Siren更差")
            else:
                per_condition[f"{sys_name}_siren_vs_modrelu"] = "NEUTRAL"
                print(f"    Siren vs modReLU: {s_mse:.6f} / {m_mse:.6f} = {ratio:.3f}  -> 中性")

    # 总判决
    any_siren_better = any(
        v == "SIREN_BETTER" for k, v in per_condition.items()
        if "siren_vs_real" in k
    )
    any_siren_unlocks = any(
        v == "SIREN_UNLOCKS" for k, v in per_condition.items()
        if "siren_vs_modrelu" in k
    )

    if any_siren_better:
        verdict = "WAVE_DYNAMICS_VIABLE"
        print(f"\n  *** 总判决: {verdict} ***")
        print("  Siren 在已知相位因果的动力系统中击败实数!")
        print("  -> 相位有用, 非等变全纯是关键 -> 波场世界模型有活路!")
    elif any_siren_unlocks:
        verdict = "ARCHITECTURE_MATTERS"
        print(f"\n  *** 总判决: {verdict} ***")
        print("  Siren 虽未击败实数, 但显著优于 modReLU")
        print("  -> 等变约束是瓶颈, 换非等变可能解锁")
    else:
        verdict = "PHASE_UNLEARNABLE"
        print(f"\n  *** 总判决: {verdict} ***")
        print("  即使在已知相位因果的合成动力系统中, 复数也不优于实数")
        print("  -> 相位无用是表示论层面的, 不是数据或架构问题")

    summary = {
        "means_mse": means_mse,
        "per_condition": per_condition,
        "overall_verdict": verdict,
    }
    (RESULTS_DIR / "exp29_verdict_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


# ============================================================================
# main
# ============================================================================
def main():
    p = argparse.ArgumentParser(description="Exp29: Complex Dynamics Probe")
    p.add_argument("--config", choices=list(CONFIGS.keys()) + ["all"], default="all")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    p.add_argument("--steps", type=int, default=TRAIN_STEPS)
    p.add_argument("--verdict_only", action="store_true")
    args = p.parse_args()
    if args.verdict_only:
        compute_verdict(tuple(args.seeds))
        return
    configs = list(CONFIGS.keys()) if args.config == "all" else [args.config]
    for c in configs:
        for seed in args.seeds:
            run_one(c, seed, args.steps)
    if args.config == "all":
        compute_verdict(tuple(args.seeds))


if __name__ == "__main__":
    main()
