"""
Exp 19: 稳定性机制分解 — Cayley vs 强制投影 vs 复数表示
=========================================================

动机 (cwf-manifesto, 紧接 exp18 PASS 之后):
  exp18 确认 CWF-full 在 Lorenz rollout 上 5-11× 胜连续 AR baseline (排除 VQ 混淆).
  A 的 EPT@0.9 = 50.0 (s2024), D 的 EPT = 2.0. 用户假说: Hamiltonian 结构 (Cayley
  等距) 是稳定性来源.

  **但读 CWFSingleBlock 代码后发现稳定性不止来自 Cayley**:
  - LieRotation (Cayley, 等距) — 结构性保 ‖ψ‖ (H1)
  - ComplexAttention + max(norm,1) — 强制投影 (H2)
  - ComplexSirenFFN + max(norm,1) — 强制投影 (H2)
  - BornStableNorm (硬投影 ≤1-ε) — 强制投影 (H2)

  3 种竞争假说:
  - H1: Cayley 等距是稳定性来源 (辛映射保能量)
  - H2: 3 个强制投影防止漂移 (工程技巧, 非物理结构)
  - H3: 复数表示内在性质 (exp18 C vs D 部分测试)

  exp19 通过消融分离这 3 种机制:
    A. cwf_full (Cayley + 3 投影) — exp18 baseline
    B. cwf_linear (通用线性 + 3 投影) — 测 H1: 换 Cayley 为非等距
    C. cwf_no_proj (Cayley, 无投影) — 测 H2: 移除 3 个投影
    D. cwf_cayley_only (仅 Cayley) — 测 Cayley 单独是否够
    E. real_full — exp18 连续 AR baseline (复用)

核心诊断: energy_drift = |‖ψ(t)‖² - ‖ψ(0)‖²| 在 rollout K=200 中的轨迹
  - Cayley (等距): drift ≈ 0
  - 强制投影: drift 被截断但非 0
  - 无约束: drift 发散

判决:
  H1_CONFIRMED: A.drift≈0, B.drift>>0, EPT(A)>EPT(B) → Cayley 等距是稳定性来源.
  H1_REFUTED: A≈B → Cayley 不关键.
  H2_CONFIRMED: C 发散但 A 稳定 → 投影必要.
  H2_REFUTED: C 也稳定 → 投影不关键, Cayley 单独够.
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
EXP02_DIR = HERE.parent / "exp02_lorenz"
EXP18_DIR = HERE.parent / "exp18_lorenz_attribution"
PROJECT_ROOT = HERE.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXP02_DIR))
sys.path.insert(0, str(EXP18_DIR))

# 复用 exp02 的数据 + Oracle + CWF 组件
from lorenz_data import generate_lorenz_trajectories  # noqa: E402
from lorenz_oracle import LorenzOracle  # noqa: E402
from cwf_lorenz import _FFTChannelEncoder, _BornChannelDecoder  # noqa: E402
from research.cwf.prototype.cwf_minimal import (  # noqa: E402
    CWFSingleBlock, LieRotation, ComplexAttention, ComplexSirenFFN, BornStableNorm,
    complex_norm, cayley_rotation, skew_to_params,
)
from exp18_lorenz_attribution import (  # noqa: E402
    generate_data, get_batch, grad_norm, MLPEncoder, MLPDynamics, MLPDecoder,
    ConfigD_RealFull, D_PER_CHANNEL, N_CHANNELS, SEQ_LEN, BATCH_SIZE, LR, WD,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_STEPS = 600
EVAL_STEPS = [400]  # only final eval (CWF slow)
ROLLOUT_K = 100  # 减到 100 (从 200, 平衡长程与速度)
HORIZONS = [1, 5, 10, 25, 50, 100]
ENERGY_LOG_INTERVAL = 10  # 每 10 步记录 ‖ψ‖²


# ============================================================================
# 消融模块
# ============================================================================
class GenericComplexLinear(nn.Module):
    """通用复数线性层 (非等距), 替换 LieRotation [config B].

    (W_re + i·W_im) · ψ, 不保范. 参数量匹配 LieRotation (d*2 → d*(d-1)/2 → d×d).
    为公平, 输出维度 d, 参数量约 d² (与 Cayley 的 skew_params d*(d-1)/2 + 构造 d×d 同量级).
    """
    def __init__(self, d):
        super().__init__()
        self.d = d
        # 直接 d×d 复数权重 (比 Cayley 的 d*(d-1)/2 略多, 但同量级)
        self.W_re = nn.Parameter(torch.randn(d, d) * (1.0 / math.sqrt(d)))
        self.W_im = nn.Parameter(torch.randn(d, d) * (1.0 / math.sqrt(d)))

    def forward(self, psi):
        """psi: (B, S, d, 2) → (B, S, d, 2), 不保范."""
        B_shape = psi.shape[:-2]
        d = self.d
        psi_flat = psi.reshape(-1, d, 2)
        re = psi_flat[..., 0]  # (N, d)
        im = psi_flat[..., 1]
        # (W_re + i·W_im)(re + i·im) = (W_re·re - W_im·im) + i(W_re·im + W_im·re)
        out_re = re @ self.W_re.T - im @ self.W_im.T
        out_im = re @ self.W_im.T + im @ self.W_re.T
        out = torch.stack([out_re, out_im], dim=-1)
        return out.reshape(*B_shape, d, 2)


class CWFSingleBlockNoProj(nn.Module):
    """Cayley + attention/FFN, 但移除所有 max(norm,1) 投影 + BornNorm [config C].

    测 H2: 强制投影是否必要. 如果移除后仍稳定, 投影不关键.
    S=1 时 attention 退化为 v 自身 (单 token 自注意力 = identity), 所以简化处理.
    """
    def __init__(self, d, hidden_mult=1):
        super().__init__()
        self.d = d
        self.lie = LieRotation(d)
        self.attn = ComplexAttention(d)  # 保留参数, 但 S=1 时 attention 退化
        self.ffn = ComplexSirenFFN(d, hidden_mult=hidden_mult)
        # 无 BornStableNorm

    def forward(self, psi):
        # Cayley (保范)
        psi = self.lie(psi)
        # S=1 时 attention: Q^H·K 是标量, unit-phase, V 是自身 → attention ≈ V
        # 直接调用 attn 但其内部 max(norm,1) 投影会被触发. 为干净消融, 我们跳过 attn,
        # 只做 Cayley + FFN (无投影).
        # FFN (移除 max(norm,1) 投影):
        h = ComplexAttention._complex_matmul(psi, self.ffn.W1) + self.ffn.b1
        re, im = h[..., 0], h[..., 1]
        sin_re = torch.sin(re) * torch.cosh(im)
        sin_im = torch.cos(re) * torch.sinh(im)
        h = torch.stack([sin_re, sin_im], dim=-1)
        psi = ComplexAttention._complex_matmul(h, self.ffn.W2) + self.ffn.b2
        # 无 BornStableNorm, 无 max(norm,1)
        return psi, []


def complex_mul_simple(a, b):
    """复数乘法 (a, b: (..., 2)) → (..., 2)."""
    re = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
    im = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
    return torch.stack([re, im], dim=-1)


class CayleyOnlyBlock(nn.Module):
    """仅 Cayley, 无 attention/FFN/projection [config D].

    测 Cayley 单独是否够维持稳定性.
    """
    def __init__(self, d, hidden_mult=1):
        super().__init__()
        self.d = d
        self.lie = LieRotation(d)

    def forward(self, psi):
        psi = self.lie(psi)
        return psi, []


# ============================================================================
# 模型包装 (复用 exp18 的 encoder/decoder, 换 block)
# ============================================================================
class LorenzModelBase(nn.Module):
    """Base: 3× FFTChannelEncoder + block + 3× BornChannelDecoder.

    子类只需指定 self.block. 提供 encode/decode + energy 监控.
    """
    def __init__(self, d, seq_len):
        super().__init__()
        self.d = d
        self.complex_d = N_CHANNELS * d
        self.encoders = nn.ModuleList([_FFTChannelEncoder(seq_len, d) for _ in range(N_CHANNELS)])
        self.decoders = nn.ModuleList([_BornChannelDecoder(d) for _ in range(N_CHANNELS)])

    def encode(self, x):
        """x: (B, T, 3) → psi: (B, 1, 3d, 2)."""
        B, T, _ = x.shape
        psi_list = [self.encoders[ch](x[:, :, ch]) for ch in range(N_CHANNELS)]
        psi = torch.cat(psi_list, dim=1)  # (B, 3d, 2)
        return psi.unsqueeze(1)  # (B, 1, 3d, 2)

    def decode(self, psi):
        """psi: (B, 1, 3d, 2) → y: (B, 3)."""
        psi = psi.squeeze(1)  # (B, 3d, 2)
        outputs = []
        for ch in range(N_CHANNELS):
            psi_ch = psi[:, ch * self.d:(ch + 1) * self.d, :]
            outputs.append(self.decoders[ch](psi_ch))
        return torch.cat(outputs, dim=-1)


class ConfigA_CWFFull(LorenzModelBase):
    """A: Cayley + 3 投影 (exp18 baseline, 完整 CWFSingleBlock)."""
    def __init__(self, d=D_PER_CHANNEL, seq_len=SEQ_LEN):
        super().__init__(d, seq_len)
        self.block = CWFSingleBlock(d=self.complex_d, hidden_mult=1)

    def forward(self, x):
        psi = self.encode(x)
        psi, _ = self.block(psi)
        y = self.decode(psi)
        return y, {"psi_norm": complex_norm(psi.squeeze(1)).mean().item()}


class ConfigB_CWFLinear(LorenzModelBase):
    """B: 通用复数线性 (非等距) + 3 投影. 测 H1."""
    def __init__(self, d=D_PER_CHANNEL, seq_len=SEQ_LEN):
        super().__init__(d, seq_len)
        self.linear = GenericComplexLinear(self.complex_d)
        self.attn = ComplexAttention(self.complex_d)
        self.ffn = ComplexSirenFFN(self.complex_d, hidden_mult=1)
        self.norm = BornStableNorm(eps=1e-3)

    def forward(self, x):
        psi = self.encode(x)
        # 通用线性 (非等距) 替换 Cayley
        psi = self.linear(psi)
        # 其余同 CWFSingleBlock (含投影)
        psi = self.attn(psi)
        psi = self.ffn(psi)
        psi = self.norm(psi)
        y = self.decode(psi)
        return y, {"psi_norm": complex_norm(psi.squeeze(1)).mean().item()}


class ConfigC_CWFNoProj(LorenzModelBase):
    """C: Cayley + attention/FFN, 无投影. 测 H2."""
    def __init__(self, d=D_PER_CHANNEL, seq_len=SEQ_LEN):
        super().__init__(d, seq_len)
        self.block = CWFSingleBlockNoProj(self.complex_d, hidden_mult=1)

    def forward(self, x):
        psi = self.encode(x)
        psi, _ = self.block(psi)
        y = self.decode(psi)
        return y, {"psi_norm": complex_norm(psi.squeeze(1)).mean().item()}


class ConfigD_CayleyOnly(LorenzModelBase):
    """D: 仅 Cayley, 无 attention/FFN/projection. 测 Cayley 单独是否够."""
    def __init__(self, d=D_PER_CHANNEL, seq_len=SEQ_LEN):
        super().__init__(d, seq_len)
        self.block = CayleyOnlyBlock(self.complex_d)

    def forward(self, x):
        psi = self.encode(x)
        psi, _ = self.block(psi)
        y = self.decode(psi)
        return y, {"psi_norm": complex_norm(psi.squeeze(1)).mean().item()}


CONFIGS = {
    "a_cwf_full": ConfigA_CWFFull,
    "b_cwf_linear": ConfigB_CWFLinear,
    "c_cwf_no_proj": ConfigC_CWFNoProj,
    "d_cayley_only": ConfigD_CayleyOnly,
}


def build_model(config_name):
    return CONFIGS[config_name](d=D_PER_CHANNEL, seq_len=SEQ_LEN)


# ============================================================================
# 能量监控 rollout
# ============================================================================
@torch.no_grad()
def rollout_with_energy(model, val_traj, k=ROLLOUT_K, n_samples=2):
    """Free rollout K 步, 记录 ‖ψ‖² 轨迹 + MSE @ 各 horizon."""
    model.eval()
    horizons = [h for h in HORIZONS if h <= k]
    max_h = max(horizons)

    all_mse = {h: [] for h in horizons}
    all_energy = []  # 每个 sample 的 ‖ψ‖² 轨迹

    for i in range(min(n_samples, val_traj.shape[0])):
        T = val_traj.shape[1]
        start = torch.randint(100, T - max_h - 10, (1,)).item()
        true_traj = val_traj[i, start:start + max_h + 1]
        hist_start = max(0, start - SEQ_LEN)
        history = val_traj[i, hist_start:start + 1].unsqueeze(0)
        if history.shape[1] < SEQ_LEN:
            pad = history[:, :1].expand(-1, SEQ_LEN - history.shape[1], -1)
            history = torch.cat([pad, history], dim=1)
        history = history[:, -SEQ_LEN:, :]

        cur_input = history
        preds = []
        energy_traj = []
        for step in range(max_h):
            # 记录 encoder 输出的 ‖ψ‖² (block 之前)
            if hasattr(model, 'encode'):
                psi = model.encode(cur_input)
                psi_norm_sq = (psi ** 2).sum().item() / psi.shape[0]  # per-sample mean
                energy_traj.append(psi_norm_sq)
            y_hat, _ = model(cur_input)
            preds.append(y_hat)
            cur_input = torch.cat([cur_input[:, 1:, :], y_hat.unsqueeze(1)], dim=1)

        preds = torch.stack(preds, dim=1).squeeze(0)
        for h in horizons:
            mse = F.mse_loss(preds[h - 1], true_traj[h]).item()
            all_mse[h].append(mse)
        all_energy.append(energy_traj)

    mean_mse = {h: sum(v) / len(v) for h, v in all_mse.items()}
    # energy drift: 最后 10 步 mean ‖ψ‖² - 前 10 步 mean ‖ψ‖²
    energy_drifts = []
    for et in all_energy:
        if len(et) >= 20:
            early = sum(et[:10]) / 10
            late = sum(et[-10:]) / 10
            energy_drifts.append(abs(late - early))
    mean_drift = sum(energy_drifts) / len(energy_drifts) if energy_drifts else float('nan')
    model.train()
    return mean_mse, mean_drift, all_energy


@torch.no_grad()
def compute_ept_metric(model, val_traj, n_samples=2, k=ROLLOUT_K):
    model.eval()
    epts = []
    for i in range(min(n_samples, val_traj.shape[0])):
        T = val_traj.shape[1]
        start = torch.randint(100, T - k - 10, (1,)).item()
        true_traj = val_traj[i, start:start + k + 1].cpu().numpy()
        hist_start = max(0, start - SEQ_LEN)
        history = val_traj[i, hist_start:start + 1].unsqueeze(0)
        if history.shape[1] < SEQ_LEN:
            pad = history[:, :1].expand(-1, SEQ_LEN - history.shape[1], -1)
            history = torch.cat([pad, history], dim=1)
        history = history[:, -SEQ_LEN:, :]
        cur_input = history
        preds = []
        for step in range(k):
            y_hat, _ = model(cur_input)
            preds.append(y_hat)
            cur_input = torch.cat([cur_input[:, 1:, :], y_hat.unsqueeze(1)], dim=1)
        preds = torch.stack(preds, dim=1).squeeze(0).cpu().numpy()
        epts.append(_compute_ept(preds, true_traj[1:], 0.9))
    model.train()
    return sum(epts) / len(epts) if epts else 0.0


def _compute_ept(pred_traj, true_traj, threshold=0.9):
    T = pred_traj.shape[0]
    for t in range(1, T):
        p = pred_traj[:t + 1].mean(axis=0)
        g = true_traj[:t + 1].mean(axis=0)
        num = ((pred_traj[:t + 1] - p) * (true_traj[:t + 1] - g)).sum(axis=0)
        denom = np.sqrt(((pred_traj[:t + 1] - p) ** 2).sum(axis=0) *
                        ((true_traj[:t + 1] - g) ** 2).sum(axis=0) + 1e-12)
        r = (num / (denom + 1e-12)).mean()
        if r < threshold:
            return t + 1
    return T


# ============================================================================
# 训练循环
# ============================================================================
def run_one(config_name, seed, steps=TRAIN_STEPS):
    tag = f"{config_name}_s{seed}"
    print(f"\n{'='*70}\n[{tag}] config={config_name}  seed={seed}  steps={steps}\n{'='*70}")
    torch.manual_seed(seed)
    model = build_model(config_name).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {config_name}  params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.95))
    train_traj, val_traj, _, _ = generate_data()

    results = {
        "config": config_name, "seed": seed, "params": n_params,
        "train_steps": steps, "trace": [], "rollout_eval": [],
        "ept": None, "nan_step": None, "diverged": False,
    }
    t0 = time.time()

    for step in range(1, steps + 1):
        x, y = get_batch(train_traj, BATCH_SIZE, SEQ_LEN)
        y_hat, info = model(x)
        loss = F.mse_loss(y_hat, y)
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
        if step % 100 == 0:
            print(f"  step {step:>4}/{steps}  loss={loss.item():.4f}  |g|={gn:.2e}  "
                  f"‖ψ‖={info.get('psi_norm', 0):.3f}  t={time.time()-t0:.0f}s", flush=True)
        if step in EVAL_STEPS:
            mse_dict, drift, energy = rollout_with_energy(model, val_traj)
            results["rollout_eval"].append({
                "step": step, "mse": mse_dict, "energy_drift": drift,
                "energy_trajectory": energy[0] if energy else [],
            })
            print(f"  >>> MSE@10={mse_dict.get(10, 0):.4f}  drift={drift:.6f}  "
                  f"EPT=...", flush=True)

    if not results["diverged"]:
        ept = compute_ept_metric(model, val_traj)
        results["ept"] = round(ept, 2)
        print(f"  >>> EPT@0.9: {ept:.2f}", flush=True)

    results["total_time_s"] = round(time.time() - t0, 2)
    best_mse10 = min((r["mse"].get(10, float('inf')) for r in results["rollout_eval"]), default=None)
    results["best_mse10"] = best_mse10
    # 最终 energy drift (最后一个 eval 的)
    results["final_energy_drift"] = results["rollout_eval"][-1]["energy_drift"] if results["rollout_eval"] else None

    out_json = RESULTS_DIR / f"{tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[saved] -> {out_json}")
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return results


# ============================================================================
# Verdict
# ============================================================================
def _load(config, seed):
    p = RESULTS_DIR / f"{config}_s{seed}.json"
    if not p.exists():
        return None
    return json.load(open(p))


def compute_verdict(seeds=(42, 2024)):
    print("\n" + "=" * 70)
    print("VERDICT: 稳定性机制分解 (Cayley vs 强制投影 vs 复数)")
    print("=" * 70)

    configs = ["a_cwf_full", "b_cwf_linear", "c_cwf_no_proj", "d_cayley_only"]
    data = {c: [] for c in configs}
    for c in configs:
        for seed in seeds:
            d = _load(c, seed)
            data[c].append(d)

    # 1. Best MSE@10 + EPT + energy drift
    print("\n--- 1. 各 config 指标 (lower MSE/drift = better, higher EPT = better) ---")
    print(f"  {'config':18s} {'MSE@10':>10} {'EPT':>6} {'drift':>10}  seeds")
    means = {}
    for c in configs:
        mses = [d["best_mse10"] for d in data[c] if d and d.get("best_mse10") is not None]
        epts = [d["ept"] for d in data[c] if d and d.get("ept") is not None]
        drifts = [d["final_energy_drift"] for d in data[c] if d and d.get("final_energy_drift") is not None]
        mse_mean = sum(mses) / len(mses) if mses else None
        ept_mean = sum(epts) / len(epts) if epts else None
        drift_mean = sum(drifts) / len(drifts) if drifts else None
        means[c] = {"mse10": mse_mean, "ept": ept_mean, "drift": drift_mean}
        print(f"  {c:18s} {mse_mean or 0:>10.4f} {ept_mean or 0:>6.2f} {drift_mean or 0:>10.6f}  "
              f"{[round(d['best_mse10'], 4) if d and d.get('best_mse10') else None for d in data[c]]}")

    # 2. H1 判决 (Cayley vs 通用线性)
    print("\n--- 2. H1: Cayley 等距 vs 通用线性 ---")
    a_drift = means["a_cwf_full"]["drift"]
    b_drift = means["b_cwf_linear"]["drift"]
    a_ept = means["a_cwf_full"]["ept"]
    b_ept = means["b_cwf_linear"]["ept"]
    print(f"  A (Cayley):    drift={a_drift or 0:.6f}  EPT={a_ept or 0:.2f}")
    print(f"  B (线性):      drift={b_drift or 0:.6f}  EPT={b_ept or 0:.2f}")
    h1_confirmed = (a_drift is not None and b_drift is not None and
                    a_drift < 0.01 and b_drift > 0.1 and (a_ept or 0) > (b_ept or 0))
    h1_refuted = (a_drift is not None and b_drift is not None and
                  abs(a_drift - b_drift) < 0.05)
    if h1_confirmed:
        print("  *** H1_CONFIRMED: Cayley 等距是稳定性来源 (A.drift≈0, B.drift>>0) ***")
    elif h1_refuted:
        print("  *** H1_REFUTED: Cayley 不关键 (A≈B) ***")
    else:
        print(f"  *** H1_INCONCLUSIVE: drift 差异不够显著 (A={a_drift}, B={b_drift}) ***")

    # 3. H2 判决 (强制投影贡献)
    print("\n--- 3. H2: 强制投影贡献 ---")
    c_drift = means["c_cwf_no_proj"]["drift"]
    c_ept = means["c_cwf_no_proj"]["ept"]
    print(f"  A (Cayley+投影): drift={a_drift or 0:.6f}  EPT={a_ept or 0:.2f}")
    print(f"  C (Cayley无投影): drift={c_drift or 0:.6f}  EPT={c_ept or 0:.2f}")
    h2_confirmed = (c_drift is not None and c_drift > 0.1 and
                    a_drift is not None and a_drift < 0.01)
    h2_refuted = (c_drift is not None and c_drift < 0.01)
    if h2_confirmed:
        print("  *** H2_CONFIRMED: 投影必要 (C 无投影后 drift 爆炸) ***")
    elif h2_refuted:
        print("  *** H2_REFUTED: 投影不关键 (C 无投影也稳定) ***")
    else:
        print(f"  *** H2_INCONCLUSIVE: C drift={c_drift} ***")

    # 4. D (仅 Cayley)
    print("\n--- 4. D: 仅 Cayley (无 attention/FFN) ---")
    d_ept = means["d_cayley_only"]["ept"]
    d_drift = means["d_cayley_only"]["drift"]
    print(f"  D (仅 Cayley): drift={d_drift or 0:.6f}  EPT={d_ept or 0:.2f}")
    if d_ept and a_ept and d_ept >= a_ept * 0.8:
        print("  *** Cayley 单独足够维持稳定性 (D.EPT ≈ A.EPT) ***")
    else:
        print("  *** Cayley 单独不够 (D.EPT << A.EPT), 需其他组件 ***")

    # 5. 总结
    print("\n--- 5. 总结 ---")
    if h1_confirmed and not h2_confirmed:
        verdict = "CAYLEY_IS_KEY"
        print("  Cayley 等距是稳定性来源, 投影不关键. 波演化有物理必然性.")
    elif h1_confirmed and h2_confirmed:
        verdict = "BOTH_MECHANISM"
        print("  Cayley + 投影都贡献稳定性. 结构 + 工程双重保障.")
    elif not h1_confirmed and h2_confirmed:
        verdict = "PROJECTION_IS_KEY"
        print("  投影是稳定性来源, Cayley 不关键. 稳定性是工程技巧非物理必然.")
    elif not h1_confirmed and not h2_refuted:
        verdict = "NEITHER"
        print("  Cayley 和投影都不关键. 稳定性来自复数表示本身 (H3).")
    else:
        verdict = "INCONCLUSIVE"
        print("  需更多数据.")

    summary = {"means": means, "h1_confirmed": h1_confirmed, "h2_confirmed": h2_confirmed,
               "verdict": verdict}
    (RESULTS_DIR / "exp19_verdict_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[saved] {RESULTS_DIR / 'exp19_verdict_summary.json'}")
    return summary


def main():
    p = argparse.ArgumentParser(description="Exp19: Stability mechanism")
    p.add_argument("--config", choices=list(CONFIGS.keys()) + ["all"], default="all")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 2024])
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
