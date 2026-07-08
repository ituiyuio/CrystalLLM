"""
Exp 28: 波场内部动态演化探针
================================

核心假设: 冻结 Stage A 编码器的波场状态 z_t 随位置(时间片)的演化有规律,
复数演化器能比实数演化器更准确地学习这个转移函数, 因为相位编码了
字节级模式之间的结构关系 (n-gram 边界、节奏、重复模式).

与 exp24 (JEPA) 的区别:
  - exp24: 在线编码器, 可把相位推成任意值 (逃逸路径)
  - exp28: 冻结编码器, 相位结构固定, 演化器必须在此基础上预测

与 next-token prediction 的区别:
  - next-token: 预测内容 (冗余)
  - 波场演化: 学习字节级模式的转移函数 (模拟动态)

设计:
  - 256-byte 窗口分成 4 个 64-byte 时间片
  - 冻结编码器对每个时间片输出 z_t (B, 16, 32) cfloat
  - 演化器: z_t -> ẑ_{t+1}, 损失 |ẑ - z_{t+1}|^2 (直接读相位)

3 条件:
  baseline:  恒等映射 ẑ_{t+1} = z_t (波场是否已足够平滑?)
  R_evolve:  实数 MLP 演化器 (cat[Re,Im] -> Linear -> GELU -> Linear)
  C_evolve:  复数 MLP 演化器 (ComplexLinear -> modReLU -> ComplexLinear)

裁决: C_evolve 的预测误差显著 < R_evolve -> 相位携带时间预测信息.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
EXP05_DIR = HERE.parent / "exp05_wave_autoencoder"
PROJECT_ROOT = HERE.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXP05_DIR))

from wave_autoencoder import (  # noqa: E402
    WaveTokenizerComplex, load_data, VOCAB_SIZE, UNIFORM_LOSS, grad_norm,
    complex_modrelu,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 损失类型: "l2" (默认复数L2), "l2_norm" (幅度归一化L2), "cos" (复余弦非Born)
LOSS_TYPE = "l2"

SEQ_LEN = 256
N_SLICES = 4
SLICE_LEN = SEQ_LEN // N_SLICES  # 64
BATCH_SIZE = 32
TRAIN_STEPS = 5000
LR = 1e-3
WD = 0.01
WARMUP_STEPS = 100
EVAL_STEPS = [200, 500, 1000, 2000, 3000, 5000]
SEEDS_DEFAULT = [42, 123, 2024]

TOKENIZER_CKPT = EXP05_DIR / "results" / "tokenizer_complex.pt"
WAVE_D = 32
# 64-byte slice -> stride-4 conv -> M=16 positions
SLICE_M = SLICE_LEN // 4  # 16
WAVE_DIM = SLICE_M * WAVE_D  # 512, flattened


# ============================================================================
# 数据
# ============================================================================
def get_batch_slices(ids, bs=BATCH_SIZE, device=DEVICE):
    """采样 256-byte 窗口, 切成 4 个 64-byte 时间片.
    返回 slices: (B, 4, 64) long.
    """
    n = len(ids) - SEQ_LEN - 1
    starts = torch.randint(0, n, (bs,))
    window = torch.stack([ids[s:s + SEQ_LEN] for s in starts])  # (B, 256)
    slices = window.view(bs, N_SLICES, SLICE_LEN)  # (B, 4, 64)
    return slices.to(device)


# ============================================================================
# 冻结波编码器 (复用 exp25/27 设计, 但对 64-byte 时间片编码)
# ============================================================================
class FrozenSliceEncoder(nn.Module):
    """冻结 Stage A 编码器, 对 64-byte 时间片编码.

    注意: WaveTokenizerComplex 的 seq_len/stride 是在初始化时设定的.
    我们用 seq_len=64, stride=4, 得到 M=16.
    """
    def __init__(self, ckpt_path=TOKENIZER_CKPT):
        super().__init__()
        # 用 64-byte 的配置加载 (权重与 seq_len 无关, 只取决于 d 和 vocab)
        self.tokenizer = WaveTokenizerComplex(vocab_size=VOCAB_SIZE, d=WAVE_D,
                                              seq_len=SLICE_LEN, stride=4)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        tok_sd = {}
        for k, v in ckpt.items():
            if k.startswith("tokenizer."):
                tok_sd[k[len("tokenizer."):]] = v
        self.tokenizer.load_state_dict(tok_sd)
        for p in self.tokenizer.parameters():
            p.requires_grad = False
        self.tokenizer.eval()

    @torch.no_grad()
    def encode_slice(self, byte_ids):
        """byte_ids (B, 64) -> z (B, 16, 32) cfloat."""
        return self.tokenizer.encode(byte_ids)  # (B, M, d) cfloat


# ============================================================================
# 演化器
# ============================================================================
class RealEvolver(nn.Module):
    """实数 MLP 演化器: cat[Re, Im] -> Linear -> GELU -> Linear -> split back.

    参数量与复数演化器匹配 (复数权重 = 2x 实数权重, 但实数版输入维度 2x).
    re_only=True 时只用 Re(z) 作为输入 (Im 置零), 测试 Im(相位)是否有用.
    """
    def __init__(self, d_flat=WAVE_DIM, re_only=False):
        super().__init__()
        self.re_only = re_only
        # 输入: cat[Re, Im] = 2 * d_flat  (re_only 时 Im 部分置零)
        # 隐藏: 2 * d_flat
        # 输出: 2 * d_flat (Re + Im)
        self.fc1 = nn.Linear(2 * d_flat, 2 * d_flat)
        self.fc2 = nn.Linear(2 * d_flat, 2 * d_flat)
        self.d_flat = d_flat

    def forward(self, z):
        # z: (B, M, d) cfloat -> flatten -> (B, M*d)
        B, M, D = z.shape
        z_flat = z.reshape(B, M * D)
        if self.re_only:
            # 只用实部, 虚部输入置零
            x = torch.cat([z_flat.real, torch.zeros_like(z_flat.real)], dim=-1)
        else:
            x = torch.cat([z_flat.real, z_flat.imag], dim=-1)  # (B, 2*M*d)
        h = F.gelu(self.fc1(x))
        out = self.fc2(h)  # (B, 2*M*d)
        re, im = out.chunk(2, dim=-1)
        out_z = torch.complex(re, im).reshape(B, M, D)
        return out_z

    def predict_from_real(self, z):
        """只用实部预测: Re(z) -> predict full z."""
        B, M, D = z.shape
        z_flat = z.reshape(B, M * D)
        x = torch.cat([z_flat.real, torch.zeros_like(z_flat.real)], dim=-1)
        h = F.gelu(self.fc1(x))
        out = self.fc2(h)
        re, im = out.chunk(2, dim=-1)
        out_z = torch.complex(re, im).reshape(B, M, D)
        return out_z


class ComplexEvolver(nn.Module):
    """复数 MLP 演化器: ComplexLinear -> modReLU -> ComplexLinear.

    输入/输出: (B, M, d) cfloat. 全程复数运算, 相位参与乘法.
    """
    def __init__(self, d=WAVE_D, d_flat=WAVE_DIM):
        super().__init__()
        # 复数线性层: d_flat -> d_flat (flattened)
        # 用 ComplexLinear 在展平的维度上操作
        scale = 1.0 / math.sqrt(d_flat)
        self.W1 = nn.Parameter(torch.randn(d_flat, d_flat, dtype=torch.complex64) * scale)
        self.W2 = nn.Parameter(torch.randn(d_flat, d_flat, dtype=torch.complex64) * scale)
        self.b1 = nn.Parameter(torch.zeros(d_flat, dtype=torch.complex64))
        self.b2 = nn.Parameter(torch.zeros(d_flat, dtype=torch.complex64))
        self.d_flat = d_flat

    def forward(self, z):
        # z: (B, M, d) cfloat
        B, M, D = z.shape
        z_flat = z.reshape(B, M * D)  # (B, d_flat) cfloat
        h = complex_modrelu(z_flat @ self.W1.t() + self.b1)
        out = h @ self.W2.t() + self.b2  # (B, d_flat) cfloat
        return out.reshape(B, M, D)


class IdentityEvolver(nn.Module):
    """恒等映射: ẑ_{t+1} = z_t."""
    def forward(self, z):
        return z


# ============================================================================
# CONFIGS
# ============================================================================
CONFIGS = {
    "baseline":   IdentityEvolver,
    "r_evolve":   RealEvolver,
    "r_evolve_re": lambda: RealEvolver(re_only=True),
    "c_evolve":   ComplexEvolver,
}

_slice_encoder = None

def get_slice_encoder():
    global _slice_encoder
    if _slice_encoder is None:
        _slice_encoder = FrozenSliceEncoder().to(DEVICE)
    return _slice_encoder

def build_model(config_name):
    if config_name == "baseline":
        return IdentityEvolver()
    elif config_name == "r_evolve":
        return RealEvolver()
    elif config_name == "r_evolve_re":
        return RealEvolver(re_only=True)
    elif config_name == "c_evolve":
        return ComplexEvolver()


# ============================================================================
# 诊断
# ============================================================================
@torch.no_grad()
def compute_pred_error(pred, target):
    """复数 L2: |pred - target|^2, 对幅度和相位同等惩罚."""
    diff = pred - target
    return (diff.real.pow(2) + diff.imag.pow(2)).mean().item()


@torch.no_grad()
def compute_phase_pred_error(pred, target):
    """相位预测误差: |arg(pred) - arg(target)| (wrapped to [-π, π])."""
    phase_diff = torch.angle(pred) - torch.angle(target)
    # wrap to [-π, π]
    phase_diff = torch.atan2(phase_diff.sin(), phase_diff.cos())
    return phase_diff.abs().mean().item()


@torch.no_grad()
def compute_mag_pred_error(pred, target):
    """幅度预测误差: ||pred| - |target||."""
    return (pred.abs() - target.abs()).abs().mean().item()


@torch.no_grad()
def compute_phase_alignment(pred, target):
    """相位对齐指数: |<pred, target>| / (|pred| * |target|) = |cos(Δφ)|.
    0 = 正交 (相位完全随机), 1 = 完全对齐."""
    inner = (pred.conj() * target).sum().abs()
    norm_prod = pred.abs().sum() * target.abs().sum() + 1e-8
    return (inner / norm_prod).item()


@torch.no_grad()
def compute_mag_ratio(pred, target):
    """幅值比: |pred| / |target| (mean). 1 = 完美匹配."""
    return (pred.abs().mean() / (target.abs().mean() + 1e-8)).item()


@torch.no_grad()
def eval_evolution(model, encoder, val_ids, n_batches=20):
    """评估单步和多步演化误差."""
    model.eval()
    single_step = []  # |ẑ_t - z_t|^2
    phase_err = []
    mag_err = []
    align = []     # phase alignment |cos(Δφ)|
    ratio = []     # magnitude ratio |pred|/|target|
    multi_step = []   # 3 步自回归

    for _ in range(n_batches):
        slices = get_batch_slices(val_ids)  # (B, 4, 64)
        # 编码所有时间片
        zs = []
        for t in range(N_SLICES):
            z = encoder.encode_slice(slices[:, t])  # (B, 16, 32) cfloat
            zs.append(z)

        # 单步: z_0 -> pred z_1, z_1 -> pred z_2, z_2 -> pred z_3
        for t in range(N_SLICES - 1):
            pred = model(zs[t])
            single_step.append(compute_pred_error(pred, zs[t + 1]))
            phase_err.append(compute_phase_pred_error(pred, zs[t + 1]))
            mag_err.append(compute_mag_pred_error(pred, zs[t + 1]))
            align.append(compute_phase_alignment(pred, zs[t + 1]))
            ratio.append(compute_mag_ratio(pred, zs[t + 1]))

        # 多步自回归: z_0 -> ẑ_1 -> ẑ_2 -> ẑ_3
        cur = zs[0]
        for t in range(N_SLICES - 1):
            cur = model(cur)
            multi_step.append(compute_pred_error(cur, zs[t + 1]))

    model.train()
    return {
        "single_step_mse": round(sum(single_step) / len(single_step), 6),
        "phase_error": round(sum(phase_err) / len(phase_err), 6),
        "mag_error": round(sum(mag_err) / len(mag_err), 6),
        "phase_alignment": round(sum(align) / len(align), 6),
        "mag_ratio": round(sum(ratio) / len(ratio), 6),
        "multi_step_mse": round(sum(multi_step) / len(multi_step), 6),
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

    print(f"\n{'='*70}\n[{tag}] config={config_name}  seed={seed}  "
          f"steps={steps}\n{'='*70}")
    torch.manual_seed(seed)
    model = build_model(config_name).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {config_name}  trainable params: {n_params:,}")

    encoder = get_slice_encoder()

    # baseline 不需要训练
    if config_name == "baseline":
        train_ids, val_ids = load_data()
        eval_metrics = eval_evolution(model, encoder, val_ids)
        results = {
            "config": config_name, "seed": seed, "params": 0,
            "loss_type": LOSS_TYPE,
            "train_steps": 0, "trace": [],
            "nan_step": None, "diverged": False,
            "final_eval": eval_metrics,
            "total_time_s": 0.0,
            "best_single_step": eval_metrics["single_step_mse"],
            "best_multi_step": eval_metrics["multi_step_mse"],
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[baseline] single_step_mse={eval_metrics['single_step_mse']}")
        print(f"[saved] -> {out_json}")
        del model
        return results

    opt = torch.optim.AdamW(
        list(p for p in model.parameters() if p.requires_grad),
        lr=LR, weight_decay=WD, betas=(0.9, 0.95))
    train_ids, val_ids = load_data()

    results = {
        "config": config_name, "seed": seed, "params": n_params,
        "loss_type": LOSS_TYPE,
        "train_steps": steps, "trace": [],
        "nan_step": None, "diverged": False,
    }
    t0 = time.time()

    for step in range(1, steps + 1):
        if step <= WARMUP_STEPS:
            lr = LR * step / WARMUP_STEPS
            for g in opt.param_groups:
                g["lr"] = lr

        slices = get_batch_slices(train_ids)  # (B, 4, 64)
        # 编码所有时间片 (冻结)
        with torch.no_grad():
            zs = []
            for t in range(N_SLICES):
                z = encoder.encode_slice(slices[:, t])
                zs.append(z)

        # 单步预测损失: 对每个 t, pred = model(z_t), target = z_{t+1}
        total_loss = torch.tensor(0.0, device=DEVICE, dtype=torch.float32)
        for t in range(N_SLICES - 1):
            pred = model(zs[t])
            target = zs[t + 1]
            if LOSS_TYPE == "l2":
                diff = pred - target
                loss_t = (diff.real.pow(2) + diff.imag.pow(2)).mean()
            elif LOSS_TYPE == "l2_norm":
                # 幅度归一化 L2: 强制对齐方向 (含相位), 不能压幅度
                pred_flat = pred.reshape(pred.shape[0], -1)
                tgt_flat = target.reshape(target.shape[0], -1)
                pred_n = pred_flat / (pred_flat.abs().norm(dim=-1, keepdim=True) + 1e-8)
                tgt_n = tgt_flat / (tgt_flat.abs().norm(dim=-1, keepdim=True) + 1e-8)
                diff = pred_n - tgt_n
                loss_t = (diff.real.pow(2) + diff.imag.pow(2)).mean()
            elif LOSS_TYPE == "cos":
                # 复余弦: 1 - Re(pred^H target)/(|pred||target|)
                # 不取绝对值, 对相位方向敏感 (非 Born)
                inner = (pred.conj() * target).sum().real
                norm_prod = pred.abs().sum() * target.abs().sum() + 1e-8
                loss_t = 1.0 - (inner / norm_prod).mean()
            total_loss = total_loss + loss_t
        total_loss = total_loss / (N_SLICES - 1)

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"  !!! NaN at step {step}")
            results["nan_step"] = step
            results["diverged"] = True
            break
        opt.zero_grad()
        total_loss.backward()
        gn = grad_norm(model)
        if math.isnan(gn) or math.isinf(gn):
            print(f"  !!! grad NaN at step {step}")
            results["nan_step"] = step
            results["diverged"] = True
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 250 == 0 or step == 100:
            print(f"  step {step:>4}/{steps}  loss={total_loss.item():.6f}  "
                  f"|g|={gn:.2e}  t={time.time()-t0:.0f}s", flush=True)
        if step in EVAL_STEPS:
            em = eval_evolution(model, encoder, val_ids)
            results["trace"].append({
                "step": step,
                "train_loss": round(total_loss.item(), 6),
                "single_step_mse": em["single_step_mse"],
                "phase_error": em["phase_error"],
                "mag_error": em["mag_error"],
                "phase_alignment": em["phase_alignment"],
                "mag_ratio": em["mag_ratio"],
                "multi_step_mse": em["multi_step_mse"],
                "grad_norm": round(gn, 4),
            })
            print(f"  >>> single={em['single_step_mse']:.6f}  "
                  f"phase_err={em['phase_error']:.6f}  "
                  f"align={em['phase_alignment']:.4f}  "
                  f"mag_ratio={em['mag_ratio']:.4f}  "
                  f"multi={em['multi_step_mse']:.6f}", flush=True)

    results["total_time_s"] = round(time.time() - t0, 2)
    final = results["trace"][-1] if results["trace"] else {}
    results["final_eval"] = final
    results["best_single_step"] = min(
        (t["single_step_mse"] for t in results["trace"]), default=None)
    results["best_multi_step"] = min(
        (t["multi_step_mse"] for t in results["trace"]), default=None)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[saved] -> {out_json}  (best_single={results['best_single_step']})")
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return results


# ============================================================================
# compute_verdict
# ============================================================================
def compute_verdict(seeds=tuple(SEEDS_DEFAULT)):
    print("\n" + "=" * 70)
    print("VERDICT: 波场内部动态演化 - 复数演化器能否优于实数?")
    print("=" * 70)

    configs = list(CONFIGS.keys())
    singles = {c: [] for c in configs}
    multis = {c: [] for c in configs}
    phases = {c: [] for c in configs}

    for c in configs:
        for seed in seeds:
            p = RESULTS_DIR / f"{c}_s{seed}.json"
            if not p.exists():
                singles[c].append(None)
                multis[c].append(None)
                phases[c].append(None)
                continue
            d = json.load(open(p))
            singles[c].append(d.get("best_single_step"))
            multis[c].append(d.get("best_multi_step"))
            fe = d.get("final_eval", {})
            if isinstance(fe, dict):
                phases[c].append(fe.get("phase_error"))
            else:
                phases[c].append(None)

    print("\n--- 单步预测 MSE (lower=better) ---")
    means_s = {}
    for c in configs:
        vals = [v for v in singles[c] if v is not None]
        if vals:
            m = sum(vals) / len(vals)
            means_s[c] = m
            print(f"  {c:12s}: mean={m:.6f}  (seeds: {[round(v,6) if v else None for v in singles[c]]})")

    print("\n--- 多步自回归 MSE (3步, lower=better) ---")
    means_m = {}
    for c in configs:
        vals = [v for v in multis[c] if v is not None]
        if vals:
            m = sum(vals) / len(vals)
            means_m[c] = m
            print(f"  {c:12s}: mean={m:.6f}  (seeds: {[round(v,6) if v else None for v in multis[c]]})")

    print("\n--- 相位预测误差 (rad, lower=better) ---")
    for c in configs:
        vals = [v for v in phases[c] if v is not None]
        if vals:
            m = sum(vals) / len(vals)
            print(f"  {c:14s}: mean={m:.6f}  (π={math.pi:.4f}, π/2={math.pi/2:.4f})")

    # phase alignment
    aligns = {c: [] for c in configs}
    ratios = {c: [] for c in configs}
    for c in configs:
        for seed in seeds:
            p = RESULTS_DIR / f"{c}_s{seed}.json"
            if not p.exists():
                aligns[c].append(None)
                ratios[c].append(None)
                continue
            d = json.load(open(p))
            fe = d.get("final_eval", {})
            if isinstance(fe, dict):
                aligns[c].append(fe.get("phase_alignment"))
                ratios[c].append(fe.get("mag_ratio"))
            else:
                aligns[c].append(None)
                ratios[c].append(None)

    print("\n--- 相位对齐指数 (|cos(Δφ)|, 0=正交, 1=完全对齐) ---")
    for c in configs:
        vals = [v for v in aligns[c] if v is not None]
        if vals:
            m = sum(vals) / len(vals)
            print(f"  {c:14s}: mean={m:.6f}")

    print("\n--- 幅值比 (|pred|/|target|, 1=完美匹配) ---")
    for c in configs:
        vals = [v for v in ratios[c] if v is not None]
        if vals:
            m = sum(vals) / len(vals)
            print(f"  {c:14s}: mean={m:.6f}")

    print("\n--- 判决 ---")
    base_s = means_s.get("baseline")
    r_s = means_s.get("r_evolve")
    re_s = means_s.get("r_evolve_re")
    c_s = means_s.get("c_evolve")

    per_condition = {}
    if r_s and base_s:
        ratio = r_s / base_s
        if ratio < 0.9:
            per_condition["r_vs_base"] = "R_LEARNS"
            print(f"  R vs baseline: {r_s:.6f} / {base_s:.6f} = {ratio:.3f}  -> 实数演化器学到了动态")
        else:
            per_condition["r_vs_base"] = "R_NO_LEARN"
            print(f"  R vs baseline: {r_s:.6f} / {base_s:.6f} = {ratio:.3f}  -> 实数演化器没学到 (波场太平滑?)")

    # 关键: Re-only vs cat[Re,Im] - Im(相位)是否有用?
    if re_s and r_s:
        ratio = re_s / r_s
        if ratio > 1.1:
            per_condition["re_vs_r"] = "IM_USEFUL"
            print(f"  Re-only vs cat[Re,Im]: {re_s:.6f} / {r_s:.6f} = {ratio:.3f}  -> Im(相位)有用! 丢掉它变差了")
        elif ratio < 0.9:
            per_condition["re_vs_r"] = "IM_HARMFUL"
            print(f"  Re-only vs cat[Re,Im]: {re_s:.6f} / {r_s:.6f} = {ratio:.3f}  -> Im(相位)有害, 丢掉更好")
        else:
            per_condition["re_vs_r"] = "IM_NEUTRAL"
            print(f"  Re-only vs cat[Re,Im]: {re_s:.6f} / {r_s:.6f} = {ratio:.3f}  -> Im(相位)无额外信息")

    if c_s and r_s:
        ratio = c_s / r_s
        if ratio < 0.9:
            per_condition["c_vs_r"] = "COMPLEX_BETTER"
            print(f"  C vs R: {c_s:.6f} / {r_s:.6f} = {ratio:.3f}  -> 复数显著优于实数! 相位有用!")
        elif ratio > 1.1:
            per_condition["c_vs_r"] = "COMPLEX_WORSE"
            print(f"  C vs R: {c_s:.6f} / {r_s:.6f} = {ratio:.3f}  -> 复数更差")
        else:
            per_condition["c_vs_r"] = "NEUTRAL"
            print(f"  C vs R: {c_s:.6f} / {r_s:.6f} = {ratio:.3f}  -> 中性 (modReLU等变约束限制)")

    # 总判决逻辑
    im_useful = per_condition.get("re_vs_r") == "IM_USEFUL"
    c_better = per_condition.get("c_vs_r") == "COMPLEX_BETTER"

    if c_better:
        verdict = "WAVE_DYNAMICS_VIABLE"
        print(f"\n  *** 总判决: {verdict} ***")
        print("  复数演化器优于实数 -> 相位在模拟中有用! 继续波场世界模型!")
    elif im_useful:
        verdict = "PHASE_INFO_EXISTS_BUT_EQUIVARIANT_BLOCKED"
        print(f"\n  *** 总判决: {verdict} ***")
        print("  Im(相位)携带信息 (Re-only变差), 但 modReLU 等变约束阻止复数演化器利用它")
        print("  -> 换非等变复数非线性 (如 Siren/zReLU) 可能解锁! 需 exp28b")
    elif per_condition.get("c_vs_r") == "NEUTRAL":
        verdict = "PHASE_NEUTRAL"
        print(f"\n  *** 总判决: {verdict} ***")
        print("  相位在演化预测中无额外信息 (Im也无用, modReLU也无害) -> 波场世界模型方向存疑")
    else:
        verdict = "PHASE_HARMFUL"
        print(f"\n  *** 总判决: {verdict} ***")

    summary = {
        "means_single_step": means_s,
        "means_multi_step": means_m,
        "per_condition": per_condition,
        "overall_verdict": verdict,
    }
    (RESULTS_DIR / "exp28_verdict_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


# ============================================================================
# main
# ============================================================================
def main():
    p = argparse.ArgumentParser(description="Exp28: Wavefield Dynamics Probe")
    p.add_argument("--config", choices=list(CONFIGS.keys()) + ["all"], default="all")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    p.add_argument("--steps", type=int, default=TRAIN_STEPS)
    p.add_argument("--loss", choices=["l2", "l2_norm", "cos"], default="l2",
                   help="loss type: l2 (default), l2_norm (magnitude-normalized), cos (non-Born cosine)")
    p.add_argument("--verdict_only", action="store_true")
    args = p.parse_args()
    global LOSS_TYPE
    LOSS_TYPE = args.loss
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
