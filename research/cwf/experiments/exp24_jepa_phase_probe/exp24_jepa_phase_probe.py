"""
Exp 24: Wave-JEPA 相位探针 - 预测损失能否钉住相位?
===================================================

exp23 证明: 即使拆除 junction 泄漏 (复数 V + 复数 FFN), 判别任务中相位仍是
自由乘客 (phase_std ≈ π/2, 最大熵). 梯度可达 (ratio≈1) 但不携带信息约束.

本实验测试第三种机制: 跨表示预测 (JEPA). 与 exp23 的单次前向判别不同,
JEPA 比较两次前向 (context encoder vs target encoder). 规范从 U(1)^{L+1}
(exp23) 降到 U(1) (JEPA), 且相对相位结构被预测损失直接约束.

核心问题: 预测损失能否把 phase_std 从 π/2 推下来?
  - 能 -> JEPA 让相位携带判别信息, 继续 Wave-JEPA
  - 不能 -> 预测也不解决问题, 波化判别终结

5 条件:
  a_recon:        重建损失 only (Stage A 复刻, 上界参考)
  a_pred_l2:      预测损失, L2 (|pred - target|^2), stop-grad target
  a_pred_cos:      预测损失, -Re(pred^H target) (Born 陷阱, 预期不动)
  a_both:          重建 + 预测 (Wave-JEPA 双任务)
  a_pred_l2_anchor: L2 + 硬锚点 (固定参考向量, 打破 U(1), 上界)

防坍缩 (VICReg 式):
  - var_loss:    维持每维 std(|z|) >= gamma
  - cov_loss:    去相关 (协方差非对角项 -> 0)
  - imag_ratio:  维持 mean(|Im(z)|/|z|) >= target_ratio (防相位坍缩到实数)
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

from wave_autoencoder import load_data, VOCAB_SIZE, UNIFORM_LOSS, grad_norm  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 256
BATCH_SIZE = 32
TRAIN_STEPS = 3000
LR = 1e-3
WD = 0.01
WARMUP_STEPS = 100
EVAL_STEPS = [200, 500, 1000, 2000, 3000]
SEEDS_DEFAULT = [42, 123, 2024]

D_C = 128             # 复数表示维度
PRED_HORIZON = 8      # 预测未来第 k 步的表示
MOMENTUM = 0.99       # target encoder 动量更新系数
VAR_GAMMA = 1.0       # variance 正则目标 std
COV_WEIGHT = 0.02     # covariance 正则权重
VAR_WEIGHT = 0.5      # variance 正则权重
IMAG_RATIO_TARGET = 0.4  # 期望 |Im|/|z| 比例 (0.5 = 完全复数)
IMAG_RATIO_WEIGHT = 0.1  # 虚部比例正则权重
RECON_WEIGHT = 1.0   # 重建损失权重 (用于 a_both)
PRED_WEIGHT = 1.0    # 预测损失权重 (用于 a_both)


# ============================================================================
# 数据: 复用 exp05 load_data
# ============================================================================
def get_batch_jepa(ids, bs, sl, device=DEVICE):
    """采样 (x_ctx, x_tgt): x_ctx = ids[s:s+sl], x_tgt = ids[s+PRED_HORIZON:s+PRED_HORIZON+sl].

    两个窗口长度相同, 错开 PRED_HORIZON 步. 用于预测未来表示.
    """
    n = len(ids) - 2 * sl - PRED_HORIZON - 1
    starts = torch.randint(0, n, (bs,))
    x_ctx = torch.stack([ids[s:s + sl] for s in starts])
    x_tgt = torch.stack([ids[s + PRED_HORIZON:s + PRED_HORIZON + sl] for s in starts])
    return x_ctx.to(device), x_tgt.to(device)


def get_batch_lm(ids, bs, sl, device=DEVICE):
    """标准 LM: x = ids[s:s+sl], y = ids[s+1:s+sl+1]. 用于 linear probe."""
    n = len(ids) - sl - 1
    starts = torch.randint(0, n, (bs,))
    x = torch.stack([ids[s:s + sl] for s in starts])
    y = torch.stack([ids[s + 1:s + sl + 1] for s in starts])
    return x.to(device), y.to(device)


# ============================================================================
# 复数算子 (复用 exp23 设计)
# ============================================================================
class ComplexLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        scale = 1.0 / math.sqrt(in_dim)
        self.weight = nn.Parameter(
            torch.randn(out_dim, in_dim, dtype=torch.complex64) * scale)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim, dtype=torch.complex64))
        else:
            self.bias = None

    def forward(self, x):
        out = x @ self.weight.t()
        if self.bias is not None:
            out = out + self.bias
        return out


class EquivariantComplexLayerNorm(nn.Module):
    """等变复数 LayerNorm: z / sqrt(E[|z|^2] + eps). U(1)-等变."""
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, z):
        variance = z.real.pow(2).mean(dim=-1, keepdim=True) + \
                   z.imag.pow(2).mean(dim=-1, keepdim=True)
        return z / torch.sqrt(variance + self.eps)


def complex_modrelu(z):
    """modReLU: g(z) = tanh(|z|) * z / |z|. 等变, 相位保留."""
    mag = torch.abs(z)
    phase = z / torch.clamp(mag, min=1e-8)
    return torch.tanh(mag) * phase


# ============================================================================
# Wave-JEPA 模型
# ============================================================================
class ComplexEncoder(nn.Module):
    """复数 encoder: byte_ids -> z (B, T, d_c) cfloat.

    Embedding + 复数 linear + modReLU + 等变 LN.
    """
    def __init__(self, vocab=VOCAB_SIZE, d_c=D_C, seq_len=SEQ_LEN):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_c)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_c) * 0.02)
        self.fc1 = ComplexLinear(d_c, d_c * 2)
        self.fc2 = ComplexLinear(d_c * 2, d_c)
        self.ln = EquivariantComplexLayerNorm(d_c)

    def forward(self, idx):
        # idx: (B, T) long
        e = self.embed(idx)  # (B, T, d_c) real
        z = torch.complex(e, torch.zeros_like(e))
        z = z + torch.complex(self.pos, torch.zeros_like(self.pos))
        h = complex_modrelu(self.fc1(z))
        z = self.ln(self.fc2(h))
        return z  # (B, T, d_c) cfloat


class ComplexPredictor(nn.Module):
    """复数 predictor: z_ctx -> pred (B, T, d_c) cfloat.

    预测未来表示. 用复数 MLP.
    """
    def __init__(self, d_c=D_C):
        super().__init__()
        self.fc1 = ComplexLinear(d_c, d_c * 2)
        self.fc2 = ComplexLinear(d_c * 2, d_c)

    def forward(self, z):
        h = complex_modrelu(self.fc1(z))
        return self.fc2(h)


class WaveJEPA(nn.Module):
    """Wave-JEPA: context encoder + predictor + target encoder (动量).

    target_encoder 不接收梯度, 用动量更新.
    """
    def __init__(self, d_c=D_C):
        super().__init__()
        self.encoder = ComplexEncoder(d_c=d_c)
        self.predictor = ComplexPredictor(d_c=d_c)
        # target encoder: 初始拷贝, 动量更新
        self.target_encoder = ComplexEncoder(d_c=d_c)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def momentum_update(self):
        for p, p_t in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            p_t.data.mul_(MOMENTUM).add_(p.data, alpha=1 - MOMENTUM)

    def forward(self, x_ctx, x_tgt):
        # encode both
        z_ctx = self.encoder(x_ctx)       # (B, T, d_c) cfloat
        with torch.no_grad():
            z_tgt = self.target_encoder(x_tgt)  # stop-grad target
        # predict
        pred = self.predictor(z_ctx)       # (B, T, d_c) cfloat
        return pred, z_tgt, z_ctx


# ============================================================================
# 损失函数
# ============================================================================
def recon_loss_fn(z_ctx, x_ctx, decoder):
    """重建损失: 从 z_ctx 解码回 byte_ids.
    split-real head: cat[Re, Im] -> Linear -> logits.
    decoder 接收复数 z_ctx, 内部做 cat[Re,Im].
    """
    logits = decoder(z_ctx)  # (B, T, vocab)
    return F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), x_ctx.reshape(-1))


def pred_loss_l2(pred, target):
    """L2 预测损失: |pred - target|^2.
    公共旋转不变, 但惩罚相对相位差.
    """
    diff = pred - target
    return (diff.real.pow(2) + diff.imag.pow(2)).mean()


def pred_loss_cos(pred, target):
    """Born 预测损失: -Re(pred^H target) / (|pred| |target|).
    归一化的复余弦. 公共旋转不变, 且不敏感于相位差 (Born 陷阱).
    归一化防止 magnitude 爆炸 (unbounded Born collapse).
    """
    inner = (pred.conj() * target).sum(dim=-1).real  # (B, T)
    norm_prod = pred.abs().sum(dim=-1) * target.abs().sum(dim=-1) + 1e-8
    cos = inner / norm_prod  # [-1, 1]
    return -cos.mean()  # 最小化 -> cos=1 (对齐)


def pred_loss_l2_anchor(pred, target, anchor):
    """L2 + 硬锚点: 在 L2 基础上加绝对相位对齐项.

    anchor: 固定的参考方向 (不可训练, register_buffer).
    锚点项: -Re(pred^H anchor) - Re(target^H anchor)
    强制 pred 和 target 的相位对齐到 anchor, 打破 U(1).
    """
    l2 = pred_loss_l2(pred, target)
    # 锚点: 要求 pred 的平均方向对齐 anchor
    pred_mean = pred.mean(dim=(0, 1))  # (d_c,) cfloat
    tgt_mean = target.mean(dim=(0, 1))
    anchor_loss = -(pred_mean.conj() * anchor).real.mean() - \
                  (tgt_mean.conj() * anchor).real.mean()
    return l2 + anchor_loss


# ============================================================================
# 防坍缩正则 (VICReg 式 + 虚部保护)
# ============================================================================
def var_loss(z, gamma=VAR_GAMMA):
    """方差正则: 维持每维 |z| 的 std >= gamma.
    z: (B, T, d_c) cfloat
    """
    # 沿 batch*time 维度计算 std
    mag = z.abs()  # (B, T, d_c)
    std = mag.reshape(-1, mag.shape[-1]).std(dim=0)  # (d_c,)
    return F.relu(gamma - std).mean()


def cov_loss(z):
    """协方差正则: 去相关.
    对 |z| 的 d_c x d_c 协方差矩阵, 非对角项 -> 0.
    """
    mag = z.abs()  # (B*T, d_c)
    mag_flat = mag.reshape(-1, mag.shape[-1])
    B = mag_flat.shape[0]
    mean = mag_flat.mean(dim=0)
    centered = mag_flat - mean
    cov = (centered.t() @ centered) / (B - 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    return (off_diag ** 2).mean()


def imag_ratio_loss(z, target_ratio=IMAG_RATIO_TARGET):
    """虚部比例正则: 维持 mean(|Im|/|z|) >= target_ratio.
    防止相位坍缩到 0 (退化为实数).
    """
    mag = z.abs()
    imag_ratio = z.imag.abs() / (mag + 1e-8)
    return F.relu(target_ratio - imag_ratio.mean()).mean()


# ============================================================================
# CONFIGS
# ============================================================================
class Decoder(nn.Module):
    """重建 decoder: z -> logits. split-real head."""
    def __init__(self, d_c=D_C, vocab=VOCAB_SIZE):
        super().__init__()
        self.head = nn.Linear(2 * d_c, vocab)

    def forward(self, z):
        h_flat = torch.cat([z.real, z.imag], dim=-1)
        return self.head(h_flat)


class LinearProbe(nn.Module):
    """linear probe: 冻结 encoder, 训 linear head 测 next-byte PPL."""
    def __init__(self, d_c=D_C, vocab=VOCAB_SIZE, seq_len=SEQ_LEN):
        super().__init__()
        self.head = nn.Linear(2 * d_c, vocab)

    def forward(self, z):
        # z: (B, T, d_c) cfloat (来自冻结的 encoder)
        h_flat = torch.cat([z.real, z.imag], dim=-1)
        return self.head(h_flat)


CONFIGS = {
    "a_recon":         "recon",
    "a_pred_l2":       "pred_l2",
    "a_pred_cos":      "pred_cos",
    "a_both":          "both",
    "a_pred_l2_anchor": "pred_l2_anchor",
}

LOSS_MODES = {
    "recon":          "recon_only",
    "pred_l2":        "pred_l2_only",
    "pred_cos":       "pred_cos_only",
    "both":           "recon_plus_pred_l2",
    "pred_l2_anchor": "pred_l2_plus_anchor",
}


def build_model(config_name):
    model = WaveJEPA(d_c=D_C)
    decoder = Decoder(d_c=D_C)
    probe = LinearProbe(d_c=D_C)
    # 硬锚点 (固定参考方向, 不可训练)
    anchor = F.normalize(torch.randn(D_C, dtype=torch.complex64), dim=0)
    return model, decoder, probe, anchor


# ============================================================================
# 诊断
# ============================================================================
@torch.no_grad()
def compute_phase_std(z):
    """计算 phase std: std of arg(z). π/2 = 均匀分布 (自由乘客)."""
    angle = torch.angle(z)
    return round(angle.std().item(), 4)


@torch.no_grad()
def compute_collapse_metrics(z):
    """坍缩检测:
    - mag_std: |z| 的 std (->0 = 幅度坍缩)
    - phase_std: arg(z) 的 std (->0 = 相位坍缩到实数; π/2 = 均匀分布)
    - imag_ratio: mean(|Im|/|z|) (->0 = 退化为实数)
    """
    mag = z.abs()
    angle = torch.angle(z)
    imag_ratio = (z.imag.abs() / (mag + 1e-8)).mean().item()
    return {
        "mag_std": round(mag.std().item(), 4),
        "phase_std": round(angle.std().item(), 4),
        "imag_ratio": round(imag_ratio, 4),
    }


@torch.no_grad()
def linear_probe_eval(encoder, probe, val_ids, n_batches=20):
    """linear probe: 冻结 encoder, 用 probe 预测 next-byte.
    返回 val_loss (CE) 和 PPL.
    """
    encoder.eval()
    probe.eval()
    total_loss, total_n = 0.0, 0
    for _ in range(n_batches):
        x, y = get_batch_lm(val_ids, BATCH_SIZE, SEQ_LEN)
        z = encoder(x)  # 冻结
        logits = probe(z)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        total_loss += loss.item() * x.shape[0]
        total_n += x.shape[0]
    encoder.train()
    probe.train()
    vl = total_loss / total_n
    ppl = math.exp(vl) if vl < 50 else float('inf')
    return vl, ppl


@torch.no_grad()
def pred_accuracy_eval(model, val_ids, n_batches=10):
    """预测准确度: pred 和 target 的复数余弦相似度 (相位对齐度)."""
    model.eval()
    total_sim = 0.0
    n = 0
    for _ in range(n_batches):
        x_ctx, x_tgt = get_batch_jepa(val_ids, BATCH_SIZE, SEQ_LEN)
        pred, z_tgt, _ = model(x_ctx, x_tgt)
        # 复数余弦: Re(pred^H target) / (|pred| |target|)
        inner = (pred.conj() * z_tgt).sum(dim=-1).real  # (B, T)
        norm_prod = pred.abs().sum(dim=-1) * z_tgt.abs().sum(dim=-1) + 1e-8
        sim = (inner / norm_prod).mean().item()
        total_sim += sim
        n += 1
    model.train()
    return round(total_sim / n, 4) if n > 0 else 0.0


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

    loss_mode = LOSS_MODES[CONFIGS[config_name]]
    print(f"\n{'='*70}\n[{tag}] config={config_name}  loss={loss_mode}  "
          f"seed={seed}  steps={steps}\n{'='*70}")
    torch.manual_seed(seed)

    model, decoder, probe, anchor = build_model(config_name)
    model = model.to(DEVICE)
    decoder = decoder.to(DEVICE)
    probe = probe.to(DEVICE)
    anchor = anchor.to(DEVICE)

    # 可训练参数: encoder + predictor + decoder (if recon) + probe
    params = list(model.encoder.parameters()) + list(model.predictor.parameters())
    if "recon" in loss_mode:
        params += list(decoder.parameters())
    # probe 始终训练 (但 encoder 冻结时只训 probe head)
    probe_params = list(probe.parameters())

    n_params = sum(p.numel() for p in params) + sum(p.numel() for p in probe_params)
    print(f"[model] {config_name}  params: {n_params:,}")

    opt = torch.optim.AdamW(
        params + probe_params, lr=LR, weight_decay=WD, betas=(0.9, 0.95))
    train_ids, val_ids = load_data()

    results = {
        "config": config_name, "seed": seed, "params": n_params,
        "loss_mode": loss_mode, "train_steps": steps, "trace": [],
        "nan_step": None, "diverged": False,
    }
    t0 = time.time()

    for step in range(1, steps + 1):
        if step <= WARMUP_STEPS:
            lr = LR * step / WARMUP_STEPS
            for g in opt.param_groups:
                g["lr"] = lr

        x_ctx, x_tgt = get_batch_jepa(train_ids, BATCH_SIZE, SEQ_LEN)
        pred, z_tgt, z_ctx = model(x_ctx, x_tgt)

        # 主损失
        if loss_mode == "recon_only":
            main_loss = recon_loss_fn(z_ctx, x_ctx, decoder)
        elif loss_mode == "pred_l2_only":
            main_loss = pred_loss_l2(pred, z_tgt)
        elif loss_mode == "pred_cos_only":
            main_loss = pred_loss_cos(pred, z_tgt)
        elif loss_mode == "recon_plus_pred_l2":
            main_loss = (RECON_WEIGHT * recon_loss_fn(z_ctx, x_ctx, decoder) +
                         PRED_WEIGHT * pred_loss_l2(pred, z_tgt))
        elif loss_mode == "pred_l2_plus_anchor":
            main_loss = pred_loss_l2_anchor(pred, z_tgt, anchor)
        else:
            raise ValueError(f"unknown loss_mode: {loss_mode}")

        # 防坍缩正则 (所有条件都加, 公平)
        reg_loss = (VAR_WEIGHT * var_loss(z_ctx) +
                    COV_WEIGHT * cov_loss(z_ctx) +
                    IMAG_RATIO_WEIGHT * imag_ratio_loss(z_ctx))
        loss = main_loss + reg_loss

        # linear probe 损失 (独立, 不反传到 encoder)
        x_lm, y_lm = get_batch_lm(train_ids, BATCH_SIZE, SEQ_LEN)
        with torch.no_grad():
            z_lm = model.encoder(x_lm)
        probe_logits = probe(z_lm)
        probe_loss = F.cross_entropy(
            probe_logits.reshape(-1, VOCAB_SIZE), y_lm.reshape(-1))

        total_loss = loss + probe_loss

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
        torch.nn.utils.clip_grad_norm_(params + probe_params, 1.0)
        opt.step()

        # 动量更新 target encoder
        model.momentum_update()

        if step % 250 == 0 or step == 100:
            print(f"  step {step:>4}/{steps}  main={main_loss.item():.4f}  "
                  f"reg={reg_loss.item():.4f}  probe={probe_loss.item():.4f}  "
                  f"|g|={gn:.2e}  t={time.time()-t0:.0f}s", flush=True)
        if step in EVAL_STEPS:
            collapse = compute_collapse_metrics(z_ctx)
            vl, ppl = linear_probe_eval(model.encoder, probe, val_ids)
            sim = pred_accuracy_eval(model, val_ids)
            results["trace"].append({
                "step": step,
                "main_loss": round(main_loss.item(), 4),
                "reg_loss": round(reg_loss.item(), 4),
                "probe_loss": round(probe_loss.item(), 4),
                "probe_val_loss": round(vl, 4),
                "probe_val_ppl": round(ppl, 2) if ppl != float('inf') else None,
                "pred_similarity": sim,
                "phase_std": collapse["phase_std"],
                "mag_std": collapse["mag_std"],
                "imag_ratio": collapse["imag_ratio"],
                "grad_norm": round(gn, 4),
            })
            print(f"  >>> phase_std={collapse['phase_std']}  "
                  f"imag_ratio={collapse['imag_ratio']}  "
                  f"probe_ppl={ppl:.2f}  pred_sim={sim}", flush=True)

    results["total_time_s"] = round(time.time() - t0, 2)
    # 最终 phase_std
    final_phase = results["trace"][-1]["phase_std"] if results["trace"] else None
    results["final_phase_std"] = final_phase
    # best probe ppl
    best_ppl = min(
        (t["probe_val_ppl"] for t in results["trace"]
         if t.get("probe_val_ppl") is not None), default=None)
    results["best_probe_ppl"] = best_ppl

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[saved] -> {out_json}  (final_phase_std={final_phase})")
    del model, decoder, probe
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return results


# ============================================================================
# compute_verdict
# ============================================================================
def compute_verdict(seeds=tuple(SEEDS_DEFAULT)):
    print("\n" + "=" * 70)
    print("VERDICT: Wave-JEPA 相位探针 - 预测损失能否钉住相位?")
    print("=" * 70)

    configs = list(CONFIGS.keys())
    phase_stds = {c: [] for c in configs}
    probe_ppls = {c: [] for c in configs}
    pred_sims = {c: [] for c in configs}

    for c in configs:
        for seed in seeds:
            p = RESULTS_DIR / f"{c}_s{seed}.json"
            if not p.exists():
                phase_stds[c].append(None)
                probe_ppls[c].append(None)
                pred_sims[c].append(None)
                continue
            d = json.load(open(p))
            phase_stds[c].append(d.get("final_phase_std"))
            probe_ppls[c].append(d.get("best_probe_ppl"))
            # last trace pred_sim
            tr = d.get("trace", [])
            pred_sims[c].append(tr[-1].get("pred_similarity") if tr else None)

    print("\n--- 最终 phase_std (π/2=1.571 = 均匀/自由, <0.5 = 被约束) ---")
    means_phase = {}
    for c in configs:
        vals = [v for v in phase_stds[c] if v is not None]
        if vals:
            m = sum(vals) / len(vals)
            means_phase[c] = m
            print(f"  {c:18s}: mean={m:.4f}  (seeds: {phase_stds[c]})")

    print("\n--- linear probe PPL (越低=表示越有判别信息) ---")
    means_ppl = {}
    for c in configs:
        vals = [v for v in probe_ppls[c] if v is not None]
        if vals:
            m = sum(vals) / len(vals)
            means_ppl[c] = m
            print(f"  {c:18s}: mean={m:.2f}  (seeds: {probe_ppls[c]})")

    print("\n--- 预测相似度 (复余弦, 越高=预测越准) ---")
    means_sim = {}
    for c in configs:
        vals = [v for v in pred_sims[c] if v is not None]
        if vals:
            m = sum(vals) / len(vals)
            means_sim[c] = m
            print(f"  {c:18s}: mean={m:.4f}  (seeds: {pred_sims[c]})")

    print("\n--- 判决 ---")
    recon_phase = means_phase.get("a_recon")
    l2_phase = means_phase.get("a_pred_l2")
    cos_phase = means_phase.get("a_pred_cos")
    anchor_phase = means_phase.get("a_pred_l2_anchor")
    PI_HALF = math.pi / 2

    per_condition = {}
    if cos_phase is not None:
        if abs(cos_phase - PI_HALF) < 0.1:
            per_condition["a_pred_cos"] = "BORN_TRAP_CONFIRMED"
            print(f"  a_pred_cos: phase_std={cos_phase:.4f} ≈ π/2  -> Born 陷阱确认")
        else:
            per_condition["a_pred_cos"] = "BORN_TRAP_REFUTED"
            print(f"  a_pred_cos: phase_std={cos_phase:.4f} != π/2  -> Born 陷阱被打破?!")

    if l2_phase is not None and recon_phase is not None:
        if l2_phase < recon_phase - 0.2:
            per_condition["a_pred_l2"] = "PRED_CONSTRAINS_PHASE"
            print(f"  a_pred_l2: phase_std={l2_phase:.4f} << recon={recon_phase:.4f}  "
                  "-> 预测约束相位!")
        elif l2_phase < PI_HALF - 0.2:
            per_condition["a_pred_l2"] = "PRED_PARTIAL"
            print(f"  a_pred_l2: phase_std={l2_phase:.4f} 介于 π/2 和 recon 之间  "
                  "-> 部分约束")
        else:
            per_condition["a_pred_l2"] = "PRED_NO_EFFECT"
            print(f"  a_pred_l2: phase_std={l2_phase:.4f} ≈ π/2  -> 预测无效果")

    if anchor_phase is not None:
        if anchor_phase < l2_phase - 0.2:
            per_condition["a_pred_l2_anchor"] = "ANCHOR_HELPFUL"
            print(f"  a_pred_l2_anchor: phase_std={anchor_phase:.4f} << l2={l2_phase:.4f}  "
                  "-> 锚点有效")
        else:
            per_condition["a_pred_l2_anchor"] = "ANCHOR_NO_HELP"
            print(f"  a_pred_l2_anchor: phase_std={anchor_phase:.4f} ≈ l2={l2_phase:.4f}  "
                  "-> 锚点无额外效果")

    # 总判决
    l2_works = per_condition.get("a_pred_l2") in ("PRED_CONSTRAINS_PHASE", "PRED_PARTIAL")
    if l2_works:
        verdict = "JEPA_VIABLE"
        print(f"\n  *** 总判决: {verdict} ***")
        print("  预测损失能约束相位 -> 继续 Wave-JEPA")
    else:
        verdict = "JEPA_DEAD"
        print(f"\n  *** 总判决: {verdict} ***")
        print("  预测损失不能约束相位 -> 波化判别终结")

    summary = {
        "means_phase_std": means_phase,
        "means_probe_ppl": means_ppl,
        "means_pred_sim": means_sim,
        "per_condition": per_condition,
        "overall_verdict": verdict,
        "pi_half": PI_HALF,
    }
    (RESULTS_DIR / "exp24_verdict_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


# ============================================================================
# main
# ============================================================================
def main():
    p = argparse.ArgumentParser(description="Exp24: Wave-JEPA Phase Probe")
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
