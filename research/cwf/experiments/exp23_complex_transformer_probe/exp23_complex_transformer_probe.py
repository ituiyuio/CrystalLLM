"""
Exp 23: 复数 Transformer 算子探针 - 相位能否在判别通路中存活?
=============================================================

争论核心: 直接波化 transformer 的 attention/FFN 算子是死路还是活路?
  - 死路论 (前 22 次实验): 复数在判别任务中相位自由漂移成噪声.
    exp15 (θ->0), exp21 (复数差 3.2x), exp22 (频域差 5x).
  - 活路论 (本次测试): 之前失败因 "junction 泄漏" - V 是实数, softmax 取实部,
    相位在 attention->FFN 接口变成自由规范自由度. 如果全流水线复数化
    (复数 V + 复数 FFN + split-real final), 相位可通过复数乘法被损失间接读到.

命门 (非线性三难):
  - modReLU:  等变, 相位是自由乘客 -> 预期失败 (exp21 复刻)
  - Siren:    全纯, 相位敏感但 CR 约束 -> 预期失败 (exp17 类比)
  - zReLU:    非全纯, 相位敏感但 75% 清零 -> 预期持平
  - modReLU+冻结偏置: 等变但 gauge-fix -> 预期中性 (exp13 复刻)

设计: 全复数流水线, 关键是 V 为复数 (不像 exp15 的实数 V).
  复数 Q/K -> Re(QK^H) 实 logits -> softmax 实权重 -> 复数 V 加权 -> 复数 FFN -> ... -> split-real head

裁决: 任一复数条件以 >20% 优势击败 R_paramatch -> 复数算子有活路;
      否则 -> 直接波化 transformer 算子是死路, 转向外挂压缩波场 (A 方案).
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
TRAIN_STEPS = 5000
LR = 1e-3
WD = 0.01
WARMUP_STEPS = 100
EVAL_STEPS = [200, 500, 1000, 2000, 3000, 5000]
SEEDS_DEFAULT = [42, 123, 2024]

D_COMPLEX = 128        # 复数维度 (等效 256 实宽)
D_REAL = 256           # 实数基线宽度
N_LAYERS = 4
N_HEADS = 4
D_FF_REAL = 512
D_FF_COMPLEX = 256     # 等效 512 实宽
DROPOUT = 0.0
SIREN_W0 = 1.0


# ============================================================================
# 数据: 标准 LM shifted-sequence batching (复用 exp05 load_data)
# ============================================================================
def get_batch_lm(ids, bs, sl, device=DEVICE):
    """采样 (x, y): x = ids[s:s+sl], y = ids[s+1:s+sl+1]. 标准 shifted LM."""
    n = len(ids) - sl - 1
    starts = torch.randint(0, n, (bs,))
    x = torch.stack([ids[s:s + sl] for s in starts])
    y = torch.stack([ids[s + 1:s + sl + 1] for s in starts])
    return x.to(device), y.to(device)


# ============================================================================
# 复数算子 (native torch.cfloat)
# ============================================================================
class ComplexLinear(nn.Module):
    """复数线性层: y = x @ W^T + b. 权重为复数.

    Args:
        in_dim, out_dim: 复数维度
        bias: 是否有偏置
        freeze_bias: 若 True, 偏置冻结为 1+0i (相位锚点测试)
    """
    def __init__(self, in_dim, out_dim, bias=True, freeze_bias=False):
        super().__init__()
        scale = 1.0 / math.sqrt(in_dim)
        self.weight = nn.Parameter(
            (torch.randn(out_dim, in_dim, dtype=torch.complex64) * scale)
        )
        if bias:
            if freeze_bias:
                # 冻结偏置为 1+0i: 提供固定相位参考系, 不可学习
                b = torch.ones(out_dim, dtype=torch.complex64)
                self.register_buffer("bias", b)
            else:
                self.bias = nn.Parameter(torch.zeros(out_dim, dtype=torch.complex64))
        else:
            self.bias = None

    def forward(self, x):
        # x: (..., in_dim) cfloat -> (..., out_dim) cfloat
        out = x @ self.weight.t()
        if self.bias is not None:
            out = out + self.bias
        return out


class EquivariantComplexLayerNorm(nn.Module):
    """等变复数 LayerNorm: z / sqrt(E[|z|^2] + eps).

    只缩放模长, 不碰相位. U(1)-等变: |e^{iθ}z|^2 = |z|^2.
    无可学习参数 (避免 affine 破坏等变性).
    """
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, z):
        # z: (..., d) cfloat
        variance = z.real.pow(2).mean(dim=-1, keepdim=True) + \
                   z.imag.pow(2).mean(dim=-1, keepdim=True)
        return z / torch.sqrt(variance + self.eps)


def complex_modrelu(z):
    """modReLU: g(z) = tanh(|z|) * z / |z|. 等变, 相位保留. (复用 exp05)"""
    mag = torch.abs(z)
    phase = z / torch.clamp(mag, min=1e-8)
    return torch.tanh(mag) * phase


def complex_siren(z):
    """Siren: sin(z). 全纯, 非等变.
    sin(a+bi) = sin(a)cosh(b) + i*cos(a)sinh(b) [PyTorch 原生支持]."""
    return torch.sin(z)


def complex_zrelu(z):
    """zReLU: z if Re(z)>0 and Im(z)>0 else 0. 非全纯, 相位敏感, ~75% 清零."""
    mask = (z.real > 0) & (z.imag > 0)
    return z * mask.to(z.dtype)


COMPLEX_ACTS = {
    "modrelu": complex_modrelu,
    "siren": complex_siren,
    "zrelu": complex_zrelu,
}


# ============================================================================
# 实数 Transformer (基线 R, R_paramatch)
# ============================================================================
class RealAttention(nn.Module):
    """标准多头注意力, 手写 (公平对照, 不用 nn.MultiheadAttention)."""
    def __init__(self, d, n_heads, seq_len=SEQ_LEN):
        super().__init__()
        assert d % n_heads == 0
        self.nh = n_heads
        self.dh = d // n_heads
        self.scale = 1.0 / math.sqrt(self.dh)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal", mask)

    def forward(self, x):
        # x: (B, T, d)
        B, T, d = x.shape
        qkv = self.qkv(x)  # (B, T, 3d)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.nh, self.dh).transpose(1, 2)  # (B, H, T, dh)
        k = k.view(B, T, self.nh, self.dh).transpose(1, 2)
        v = v.view(B, T, self.nh, self.dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        scores = scores.masked_fill(self.causal[:T, :T].unsqueeze(0), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # (B, H, T, dh)
        out = out.transpose(1, 2).reshape(B, T, d)
        return self.out(out)


class RealTransformerBlock(nn.Module):
    def __init__(self, d, n_heads, d_ff, act="gelu"):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = RealAttention(d, n_heads)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d_ff)
        self.fc2 = nn.Linear(d_ff, d)
        self.act = {"gelu": F.gelu, "relu": F.relu}[act]

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(self.act(self.fc1(self.ln2(x))))
        return x


class RealTransformer(nn.Module):
    def __init__(self, d=D_REAL, n_layers=N_LAYERS, n_heads=N_HEADS,
                 d_ff=D_FF_REAL, vocab=VOCAB_SIZE, seq_len=SEQ_LEN, act="gelu"):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d) * 0.02)
        self.blocks = nn.ModuleList([
            RealTransformerBlock(d, n_heads, d_ff, act) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        # weight tying
        self.head.weight = self.embed.weight
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal", mask)

    def forward(self, idx):
        # idx: (B, T) long
        x = self.embed(idx) + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)


# ============================================================================
# 复数 Transformer (C-mod, C-siren, C-zrelu, C-mod-anchor)
# ============================================================================
class ComplexAttention(nn.Module):
    """复数多头注意力. V 为复数 (关键!).

    score = Re(QK^H) / sqrt(dh)  -> 实 logits -> 实 softmax -> 复数 V 加权.
    相位通过复数 V 穿过 attention->FFN junction (不像 exp15 的实数 V).
    """
    def __init__(self, d_c, n_heads, seq_len=SEQ_LEN):
        super().__init__()
        assert d_c % n_heads == 0
        self.nh = n_heads
        self.dh = d_c // n_heads
        self.scale = 1.0 / math.sqrt(self.dh)
        self.q = ComplexLinear(d_c, d_c, bias=False)
        self.k = ComplexLinear(d_c, d_c, bias=False)
        self.v = ComplexLinear(d_c, d_c, bias=False)
        self.out = ComplexLinear(d_c, d_c, bias=False)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal", mask)

    def forward(self, z):
        # z: (B, T, d_c) cfloat
        B, T, d_c = z.shape
        q = self.q(z).view(B, T, self.nh, self.dh).transpose(1, 2)  # (B, H, T, dh)
        k = self.k(z).view(B, T, self.nh, self.dh).transpose(1, 2)
        v = self.v(z).view(B, T, self.nh, self.dh).transpose(1, 2)
        # Hermitian QK^H: <q_i, k_j> = Σ q_i[d] * conj(k_j[d])
        scores = torch.einsum('bhtd,bhsd->bhts', q, k.conj())  # (B, H, T, T) cfloat
        logits = scores.real * self.scale  # Re(QK^H), 实 logits
        logits = logits.masked_fill(self.causal[:T, :T].unsqueeze(0), float('-inf'))
        attn = F.softmax(logits, dim=-1)  # (B, H, T, T) real
        # 复数 V 被实权重加权 -> 相位保留
        attn_c = attn.to(v.dtype)  # real -> complex (imag=0), 保留 attn 的实值
        out = torch.einsum('bhts,bhsd->bhtd', attn_c, v)  # (B, H, T, dh) cfloat
        out = out.transpose(1, 2).reshape(B, T, d_c)  # (B, T, d_c) cfloat
        return self.out(out)


class ComplexFFN(nn.Module):
    """复数 FFN: ComplexLinear -> σ -> ComplexLinear.

    σ 的选择决定相位是否被读到:
      modrelu: 等变, 相位自由
      siren:   全纯, 非等变
      zrelu:   非全纯, 非等变
    """
    def __init__(self, d_c, d_ff, act="modrelu", freeze_bias=False):
        super().__init__()
        self.fc1 = ComplexLinear(d_c, d_ff, bias=True, freeze_bias=freeze_bias)
        self.fc2 = ComplexLinear(d_ff, d_c, bias=True, freeze_bias=freeze_bias)
        self.act_fn = COMPLEX_ACTS[act]
        self.act_name = act

    def forward(self, z):
        h = self.act_fn(self.fc1(z))
        return self.fc2(h)


class ComplexTransformerBlock(nn.Module):
    def __init__(self, d_c, n_heads, d_ff, act="modrelu", freeze_bias=False):
        super().__init__()
        self.ln1 = EquivariantComplexLayerNorm(d_c)
        self.attn = ComplexAttention(d_c, n_heads)
        self.ln2 = EquivariantComplexLayerNorm(d_c)
        self.ffn = ComplexFFN(d_c, d_ff, act, freeze_bias)

    def forward(self, z):
        z = z + self.attn(self.ln1(z))
        z = z + self.ffn(self.ln2(z))
        return z


class ComplexTransformer(nn.Module):
    def __init__(self, d_c=D_COMPLEX, n_layers=N_LAYERS, n_heads=N_HEADS,
                 d_ff=D_FF_COMPLEX, vocab=VOCAB_SIZE, seq_len=SEQ_LEN,
                 ffn_act="modrelu", freeze_bias=False):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_c)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_c) * 0.02)
        self.blocks = nn.ModuleList([
            ComplexTransformerBlock(d_c, n_heads, d_ff, ffn_act, freeze_bias)
            for _ in range(n_layers)
        ])
        self.ln_f = EquivariantComplexLayerNorm(d_c)
        # split-real head: cat[Re, Im] -> Linear. 匹配 wave_autoencoder 解码器.
        self.head = nn.Linear(2 * d_c, vocab, bias=True)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal", mask)
        self.ffn_act_name = ffn_act

    def forward(self, idx):
        # idx: (B, T) long
        e = self.embed(idx)  # (B, T, d_c) real
        z = torch.complex(e, torch.zeros_like(e))  # 初始虚部 0, 相位由后续层生成
        z = z + torch.complex(self.pos, torch.zeros_like(self.pos))
        for blk in self.blocks:
            z = blk(z)
        z = self.ln_f(z)
        h_flat = torch.cat([z.real, z.imag], dim=-1)  # (B, T, 2*d_c)
        return self.head(h_flat)


# ============================================================================
# 参数匹配搜索: 找到与 C-mod 参数量匹配的实数 d
# ============================================================================
def count_params(model):
    return sum(p.numel() for p in model.parameters())


def find_paramatch_d(target_params):
    """二分搜索实数 d 使 RealTransformer 参数量 ≈ target_params."""
    lo, hi = 64, 512
    best_d, best_diff = lo, abs(count_params(RealTransformer(d=lo)) - target_params)
    for d in range(lo, hi + 1, 4):
        p = count_params(RealTransformer(d=d, d_ff=2 * d))
        diff = abs(p - target_params)
        if diff < best_diff:
            best_d, best_diff = d, diff
    return best_d


# ============================================================================
# 诊断
# ============================================================================
@torch.no_grad()
def compute_val_loss(model, val_ids, n_batches=20, sl=SEQ_LEN, bs=BATCH_SIZE):
    model.eval()
    total_loss, total_n = 0.0, 0
    for _ in range(n_batches):
        x, y = get_batch_lm(val_ids, bs, sl)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        total_loss += loss.item() * x.shape[0]
        total_n += x.shape[0]
    model.train()
    return total_loss / total_n


@torch.no_grad()
def compute_diagnostics(model, val_ids, n_batches=5):
    """计算相位漂移、梯度分裂比、FFN 激活率."""
    model.eval()
    phase_stds = []
    ffn_act_rates = []

    # 收集中间激活
    hooks = []
    activations = {}

    def make_hook(name):
        def hook_fn(module, inp, out):
            activations[name] = out.detach()
        return hook_fn

    for i, blk in enumerate(getattr(model, "blocks", [])):
        h = blk.register_forward_hook(make_hook(f"block{i}"))
        hooks.append(h)

    # 跑一个 batch
    x, _ = get_batch_lm(val_ids, BATCH_SIZE, SEQ_LEN)
    _ = model(x)

    for i in range(len(getattr(model, "blocks", []))):
        act = activations.get(f"block{i}")
        if act is None:
            continue
        if act.is_complex():
            # 相位标准差
            angle = torch.angle(act)
            phase_stds.append(round(angle.std().item(), 4))
        # else: real model, no phase

    # FFN 激活率 (仅复数模型有意义)
    if hasattr(model, "blocks") and hasattr(model.blocks[0], "ffn"):
        for i, blk in enumerate(model.blocks):
            if hasattr(blk, "ffn") and hasattr(blk.ffn, "act_name"):
                # 重跑 FFN 中间结果
                with torch.no_grad():
                    z = activations.get(f"block{i}")
                    if z is not None and z.is_complex():
                        h_pre = blk.ln2(z)
                        h_post_act = blk.ffn.act_fn(blk.ffn.fc1(h_pre))
                        rate = (h_post_act.abs() > 1e-6).float().mean().item()
                        ffn_act_rates.append(round(rate, 4))

    for h in hooks:
        h.remove()

    model.train()
    return {
        "phase_std_per_layer": phase_stds,
        "ffn_activation_rate": ffn_act_rates,
    }


def compute_grad_ratio(model):
    """计算 grad_imag / grad_real 比率, 分 attn vs ffn."""
    attn_im_sum, attn_re_sum = 0.0, 0.0
    ffn_im_sum, ffn_re_sum = 0.0, 0.0
    for name, p in model.named_parameters():
        if not p.is_complex() or p.grad is None:
            continue
        g_re = p.grad.real.abs().sum().item()
        g_im = p.grad.imag.abs().sum().item()
        if "attn" in name or ".q." in name or ".k." in name or ".v." in name or ".out." in name:
            attn_re_sum += g_re
            attn_im_sum += g_im
        elif "ffn" in name or "fc1" in name or "fc2" in name:
            ffn_re_sum += g_re
            ffn_im_sum += g_im
    return {
        "grad_ratio_attn": round(attn_im_sum / (attn_re_sum + 1e-12), 4) if attn_re_sum > 0 else 0.0,
        "grad_ratio_ffn": round(ffn_im_sum / (ffn_re_sum + 1e-12), 4) if ffn_re_sum > 0 else 0.0,
    }


# ============================================================================
# CONFIGS
# ============================================================================
CONFIGS = {
    "r_real":       lambda: RealTransformer(d=D_REAL, d_ff=D_FF_REAL, act="gelu"),
    "r_paramatch":  lambda: RealTransformer(
                         d=D_PARAMATCH, d_ff=2 * D_PARAMATCH, act="gelu"),
    "c_mod":        lambda: ComplexTransformer(
                         d_c=D_COMPLEX, d_ff=D_FF_COMPLEX, ffn_act="modrelu", freeze_bias=False),
    "c_siren":      lambda: ComplexTransformer(
                         d_c=D_COMPLEX, d_ff=D_FF_COMPLEX, ffn_act="siren", freeze_bias=False),
    "c_zrelu":      lambda: ComplexTransformer(
                         d_c=D_COMPLEX, d_ff=D_FF_COMPLEX, ffn_act="zrelu", freeze_bias=False),
    "c_mod_anchor": lambda: ComplexTransformer(
                         d_c=D_COMPLEX, d_ff=D_FF_COMPLEX, ffn_act="modrelu", freeze_bias=True),
}


def build_model(config_name):
    return CONFIGS[config_name]()


# Siren 初始化 (复数版, ω0=1.0)
def init_siren(model):
    for blk in model.blocks:
        if hasattr(blk.ffn, "act_name") and blk.ffn.act_name == "siren":
            fc1, fc2 = blk.ffn.fc1, blk.ffn.fc2
            for layer in [fc1]:
                fan_in = layer.weight.shape[1]
                bound = math.sqrt(6.0 / fan_in) / SIREN_W0
                with torch.no_grad():
                    layer.weight.uniform_(-bound, bound)
            for layer in [fc2]:
                fan_in = layer.weight.shape[1]
                bound = math.sqrt(6.0 / fan_in)
                with torch.no_grad():
                    layer.weight.uniform_(-bound, bound)


def build_model_with_init(config_name):
    model = CONFIGS[config_name]()
    if config_name == "c_siren":
        init_siren(model)
    return model


# ============================================================================
# run_one
# ============================================================================
def run_one(config_name, seed, steps=TRAIN_STEPS, force=False):
    tag = f"{config_name}_s{seed}"
    out_json = RESULTS_DIR / f"{tag}.json"
    if not force and out_json.exists():
        d = json.load(open(out_json))
        if d.get("train_steps") == steps and not d.get("diverged"):
            print(f"[skip] {tag} already done (best_val={d.get('best_val')})")
            return d
    print(f"\n{'='*70}\n[{tag}] config={config_name}  seed={seed}  steps={steps}\n{'='*70}")
    torch.manual_seed(seed)
    model = build_model_with_init(config_name).to(DEVICE)
    n_params = count_params(model)
    print(f"[model] {config_name}  params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.95))
    train_ids, val_ids = load_data()

    results = {
        "config": config_name, "seed": seed, "params": n_params,
        "train_steps": steps, "trace": [],
        "nan_step": None, "diverged": False,
    }
    t0 = time.time()

    for step in range(1, steps + 1):
        # linear warmup
        if step <= WARMUP_STEPS:
            lr = LR * step / WARMUP_STEPS
            for g in opt.param_groups:
                g["lr"] = lr

        x, y = get_batch_lm(train_ids, BATCH_SIZE, SEQ_LEN)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))

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
            print(f"  step {step:>4}/{steps}  loss={loss.item():.4f}  |g|={gn:.2e}  "
                  f"t={time.time()-t0:.0f}s", flush=True)
        if step in EVAL_STEPS:
            vl = compute_val_loss(model, val_ids)
            diag = compute_diagnostics(model, val_ids)
            grad_r = compute_grad_ratio(model)
            ppl = math.exp(vl) if vl < 50 else float('inf')
            results["trace"].append({
                "step": step,
                "train_loss": round(loss.item(), 4),
                "val_loss": round(vl, 4),
                "val_ppl": round(ppl, 2) if ppl != float('inf') else None,
                "phase_std": diag["phase_std_per_layer"],
                "ffn_act_rate": diag["ffn_activation_rate"],
                "grad_ratio": grad_r,
                "grad_norm": round(gn, 4),
            })
            print(f"  >>> val_loss={vl:.4f}  ppl={ppl:.2f}  "
                  f"phase_std={diag['phase_std_per_layer']}  "
                  f"grad_ratio={grad_r}", flush=True)

    results["total_time_s"] = round(time.time() - t0, 2)
    best_val = min((t["val_loss"] for t in results["trace"]), default=None)
    results["best_val"] = best_val
    results["best_step"] = next(
        (t["step"] for t in results["trace"] if t["val_loss"] == best_val), None)

    out_json = RESULTS_DIR / f"{tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[saved] -> {out_json}  (best_val={best_val})")
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return results


# ============================================================================
# compute_verdict
# ============================================================================
def compute_verdict(seeds=tuple(SEEDS_DEFAULT)):
    print("\n" + "=" * 70)
    print("VERDICT: 复数 Transformer 算子探针 - 相位能否在判别通路中存活?")
    print("=" * 70)

    configs = list(CONFIGS.keys())
    bests = {c: [] for c in configs}
    for c in configs:
        for seed in seeds:
            p = RESULTS_DIR / f"{c}_s{seed}.json"
            if not p.exists():
                bests[c].append(None)
                continue
            d = json.load(open(p))
            bests[c].append(d.get("best_val"))

    print("\n--- 结果 (best val_loss, lower=better) ---")
    means = {}
    for c in configs:
        vals = [b for b in bests[c] if b is not None]
        if vals:
            m = sum(vals) / len(vals)
            means[c] = m
            print(f"  {c:14s}: mean={m:.4f}  (seeds: {[round(b,4) if b else None for b in bests[c]]})")
        else:
            print(f"  {c:14s}: (no data)")

    # 参数量对比
    print("\n--- 参数量 ---")
    param_json = RESULTS_DIR / f"{configs[0]}_s{seeds[0]}.json"
    if param_json.exists():
        for c in configs:
            p = RESULTS_DIR / f"{c}_s{seeds[0]}.json"
            if p.exists():
                d = json.load(open(p))
                print(f"  {c:14s}: {d.get('params', '?'):,}")

    # 判决: 每个复数条件 vs R_paramatch
    print("\n--- 判决 (vs R_paramatch) ---")
    baseline = means.get("r_paramatch")
    if baseline is None:
        baseline = means.get("r_real")
        print(f"  (R_paramatch 无数据, 回退到 R_real={baseline})")

    per_condition = {}
    overall_viable = False
    if baseline:
        for c in ["c_mod", "c_siren", "c_zrelu", "c_mod_anchor"]:
            cm = means.get(c)
            if cm is None:
                per_condition[c] = "NO_DATA"
                continue
            ratio = cm / baseline
            if ratio < 0.8:
                verdict_c = "COMPLEX_ADVANTAGE"
                overall_viable = True
            elif ratio > 1.25:
                verdict_c = "REAL_ADVANTAGE"
            else:
                verdict_c = "NEUTRAL"
            per_condition[c] = verdict_c
            print(f"  {c:14s}: {cm:.4f} / {baseline:.4f} = {ratio:.3f}  -> {verdict_c}")

    overall = "COMPLEX_OPERATOR_VIABLE" if overall_viable else "COMPLEX_OPERATOR_DEAD"
    print(f"\n  *** 总判决: {overall} ***")

    # R vs R_paramatch (参数量效应)
    rm = means.get("r_real")
    rpm = means.get("r_paramatch")
    if rm and rpm:
        print(f"  (参考: R_real={rm:.4f} vs R_paramatch={rpm:.4f}, "
              f"参数效应={rm/rpm:.3f})")

    summary = {
        "means": means,
        "per_condition_verdict": per_condition,
        "overall_verdict": overall,
        "baseline": "r_paramatch" if "r_paramatch" in means else "r_real",
    }
    (RESULTS_DIR / "exp23_verdict_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


# ============================================================================
# main
# ============================================================================
# 预计算 D_PARAMATCH (延迟到第一次需要时, 因为它依赖 C-mod 参数量)
_D_PARAMATCH = None


def get_paramatch_d():
    global _D_PARAMATCH
    if _D_PARAMATCH is None:
        c_mod_params = count_params(
            ComplexTransformer(d_c=D_COMPLEX, d_ff=D_FF_COMPLEX, ffn_act="modrelu"))
        _D_PARAMATCH = find_paramatch_d(c_mod_params)
        print(f"[paramatch] C-mod params={c_mod_params:,}  "
              f"-> R_paramatch d={_D_PARAMATCH}  "
              f"params={count_params(RealTransformer(d=_D_PARAMATCH, d_ff=2*_D_PARAMATCH)):,}")
    return _D_PARAMATCH


class _ParamatchWrapper:
    """延迟计算 D_PARAMATCH, 在 build_model 时才求值."""
    def __call__(self):
        return RealTransformer(d=get_paramatch_d(), d_ff=2 * get_paramatch_d(), act="gelu")


# 替换 CONFIGS 中的 lambda 为延迟版本
CONFIGS["r_paramatch"] = _ParamatchWrapper()


def main():
    p = argparse.ArgumentParser(description="Exp23: Complex Transformer Operator Probe")
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
