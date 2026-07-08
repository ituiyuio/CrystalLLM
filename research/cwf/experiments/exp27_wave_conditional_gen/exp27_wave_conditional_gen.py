"""
Exp 27: 波场条件生成 - 冻结波编码器作为生成条件的唯一通道
=========================================================

exp25/26 证明: 波场全局注入对判别无增量 (信息冗余). 根本原因是 decoder 能看到
和波场同样的 tokens. 在生成任务中, 这个冗余可以消除: decoder 只生成 target,
context 只通过波场传入 (cross-attention).

关键区别:
  - exp25/26: transformer 和 wave encoder 看同样的 tokens -> 冗余
  - exp27: decoder 只看 target, context 只通过波场传入 -> 无冗余

架构: 512-byte 窗口, context=bytes[0:256], target=bytes[256:512].
  decoder 在 target 上做因果 self-attention, cross-attend 到 context 的波场表示.

4 条件:
  R_gen:          无条件 decoder (学习边缘分布, 下界)
  R+pool_gen:     朴素实数 mean-pool (stride-4, 64 位置) -> cross-attn
  R+wave_gen:     冻结复数波场 (gauge-fixed, split-real, 64 位置) -> cross-attn
  R+wave_real_gen: 波场实部 only (64 位置) -> cross-attn

判决:
  R+wave < R_gen -> 波场作为生成条件有效
  R+wave < R+pool_gen -> 波场比朴素池化更好 (学习性滤波器优势)
  R+wave < R+wave_real_gen -> 复数有额外优势
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
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONTEXT_LEN = 256
TARGET_LEN = 256
SEQ_LEN = CONTEXT_LEN + TARGET_LEN  # 512
BATCH_SIZE = 32
TRAIN_STEPS = 5000
LR = 1e-3
WD = 0.01
WARMUP_STEPS = 100
EVAL_STEPS = [200, 500, 1000, 2000, 3000, 5000]
SEEDS_DEFAULT = [42, 123, 2024]

D_MODEL = 256
N_LAYERS = 4
N_HEADS = 4
D_FF = 512
DROPOUT = 0.0

TOKENIZER_CKPT = EXP05_DIR / "results" / "tokenizer_complex.pt"
WAVE_D = 32
WAVE_M = 64  # 256 / 4
POOL_STRIDE = 4  # 朴素池化的 stride, 匹配波场位置数


# ============================================================================
# 数据
# ============================================================================
def get_batch_gen(ids, bs=BATCH_SIZE, device=DEVICE):
    """采样 (context, target): context = ids[s:s+256], target = ids[s+256:s+512].
    decoder 只看 target, context 通过波场传入."""
    n = len(ids) - SEQ_LEN - 1
    starts = torch.randint(0, n, (bs,))
    context = torch.stack([ids[s:s + CONTEXT_LEN] for s in starts])
    target = torch.stack([ids[s + CONTEXT_LEN:s + SEQ_LEN] for s in starts])
    return context.to(device), target.to(device)


# ============================================================================
# 冻结波编码器
# ============================================================================
class FrozenWaveEncoder(nn.Module):
    def __init__(self, ckpt_path=TOKENIZER_CKPT):
        super().__init__()
        self.tokenizer = WaveTokenizerComplex(vocab_size=VOCAB_SIZE, d=WAVE_D,
                                              seq_len=CONTEXT_LEN, stride=4)
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
    def extract_full(self, byte_ids):
        """提取完整波场 (B, M, d) -> gauge-fix -> split-real (B, M, 2d).
        保留时序结构 (64 位置), 用于 cross-attention."""
        psi = self.tokenizer.encode(byte_ids)  # (B, M, d) cfloat
        # gauge-fix: 旋转使 mean direction 的相位 = 0
        mu = psi.sum(dim=1)  # (B, d) cfloat
        mu_phase = mu / (mu.abs() + 1e-8)
        psi_fixed = psi * mu_phase.conj().unsqueeze(1)  # (B, M, d)
        # split-real: cat[Re, Im] over last dim
        z_flat = torch.cat([psi_fixed.real, psi_fixed.imag], dim=-1)  # (B, M, 2d)
        return z_flat  # (B, M, 2d) real

    @torch.no_grad()
    def extract_full_real_only(self, byte_ids):
        """实部 only: (B, M, d) real."""
        psi = self.tokenizer.encode(byte_ids)
        mu = psi.sum(dim=1)
        mu_phase = mu / (mu.abs() + 1e-8)
        psi_fixed = psi * mu_phase.conj().unsqueeze(1)
        return psi_fixed.real  # (B, M, d) real


# ============================================================================
# 朴素池化 (匹配波场位置数)
# ============================================================================
class NaivePool(nn.Module):
    """朴素实数 mean-pool: context embedding -> stride-4 mean-pool -> (B, M, d)."""
    def __init__(self, embed_layer, d_model=D_MODEL, stride=POOL_STRIDE):
        super().__init__()
        self.embed = embed_layer  # 共享 (冻结) 的 embedding
        self.stride = stride
        self.proj = nn.Linear(d_model, d_model, bias=False)  # 投影到 d_model

    @torch.no_grad()
    def extract(self, byte_ids):
        """byte_ids (B, 256) -> pooled (B, M, d) real. M=64."""
        emb = self.embed(byte_ids)  # (B, 256, d)
        B, L, D = emb.shape
        M = L // self.stride
        # reshape to (B, M, stride, D) and mean over stride
        emb = emb[:, :M * self.stride].view(B, M, self.stride, D)
        pooled = emb.mean(dim=2)  # (B, M, D)
        return self.proj(pooled)  # (B, M, d_model)


# ============================================================================
# Attention 模块
# ============================================================================
class SelfAttention(nn.Module):
    """因果 self-attention on target sequence."""
    def __init__(self, d, n_heads, seq_len=TARGET_LEN):
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
        B, T, d = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.nh, self.dh).transpose(1, 2)
        k = k.view(B, T, self.nh, self.dh).transpose(1, 2)
        v = v.view(B, T, self.nh, self.dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(self.causal[:T, :T].unsqueeze(0), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, d)
        return self.out(out)


class CrossAttention(nn.Module):
    """Cross-attention: target queries attend to context keys/values."""
    def __init__(self, d, n_heads):
        super().__init__()
        assert d % n_heads == 0
        self.nh = n_heads
        self.dh = d // n_heads
        self.scale = 1.0 / math.sqrt(self.dh)
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d, 2 * d, bias=False)  # context -> K, V
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, context):
        """x: (B, T_target, d), context: (B, M, d) -> (B, T_target, d)."""
        B, T, d = x.shape
        M = context.shape[1]
        q = self.q(x).view(B, T, self.nh, self.dh).transpose(1, 2)  # (B, H, T, dh)
        kv = self.kv(context)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(B, M, self.nh, self.dh).transpose(1, 2)  # (B, H, M, dh)
        v = v.view(B, M, self.nh, self.dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, M)
        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # (B, H, T, dh)
        out = out.transpose(1, 2).reshape(B, T, d)
        return self.out(out)


# ============================================================================
# Decoder Block
# ============================================================================
class DecoderBlock(nn.Module):
    """Self-attn + (optional) cross-attn + FFN."""
    def __init__(self, d, n_heads, d_ff, use_cross=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.self_attn = SelfAttention(d, n_heads)
        self.use_cross = use_cross
        if use_cross:
            self.ln2 = nn.LayerNorm(d)
            self.cross_attn = CrossAttention(d, n_heads)
        self.ln3 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d_ff)
        self.fc2 = nn.Linear(d_ff, d)

    def forward(self, x, context=None):
        x = x + self.self_attn(self.ln1(x))
        if self.use_cross and context is not None:
            x = x + self.cross_attn(self.ln2(x), context)
        x = x + self.fc2(F.gelu(self.fc1(self.ln3(x))))
        return x


# ============================================================================
# Conditional Generator
# ============================================================================
class CondGenerator(nn.Module):
    """条件生成器: target 自回归 + context cross-attention.

    cond_type:
      'none':      无条件 (R_gen)
      'pool':      朴素 mean-pool (R+pool_gen)
      'wave':      冻结波场 split-real (R+wave_gen)
      'wave_real': 冻结波场实部 (R+wave_real_gen)
    """
    def __init__(self, d=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                 d_ff=D_FF, vocab=VOCAB_SIZE, cond_type="none"):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, TARGET_LEN, d) * 0.02)
        self.cond_type = cond_type

        # 条件投影 (除 'none' 外都需要)
        if cond_type == "pool":
            self.cond_proj = nn.Linear(d, d, bias=False)  # pool 已经 d_model
        elif cond_type == "wave":
            self.cond_proj = nn.Linear(2 * WAVE_D, d, bias=False)  # split-real
        elif cond_type == "wave_real":
            self.cond_proj = nn.Linear(WAVE_D, d, bias=False)  # real only

        use_cross = cond_type != "none"
        self.blocks = nn.ModuleList([
            DecoderBlock(d, n_heads, d_ff, use_cross=use_cross)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight  # weight tying

    def forward(self, target_ids, context_cond=None):
        """target_ids: (B, T) long. context_cond: (B, M, cond_d) or None."""
        x = self.embed(target_ids) + self.pos[:, :target_ids.shape[1]]
        # 投影条件
        context = None
        if context_cond is not None:
            context = self.cond_proj(context_cond)  # (B, M, d)
        for blk in self.blocks:
            x = blk(x, context)
        x = self.ln_f(x)
        return self.head(x)


# ============================================================================
# CONFIGS
# ============================================================================
CONFIGS = {
    "r_gen":          "none",
    "r_pool_gen":     "pool",
    "r_wave_gen":     "wave",
    "r_wave_real_gen": "wave_real",
}

_wave_encoder = None

def get_wave_encoder():
    global _wave_encoder
    if _wave_encoder is None:
        _wave_encoder = FrozenWaveEncoder().to(DEVICE)
    return _wave_encoder

def build_model(config_name):
    cond_type = CONFIGS[config_name]
    return CondGenerator(cond_type=cond_type)


# ============================================================================
# 诊断
# ============================================================================
@torch.no_grad()
def compute_val_loss(model, val_ids, wave_enc, config_name, n_batches=20):
    model.eval()
    total_loss, total_n = 0.0, 0
    for _ in range(n_batches):
        ctx, tgt = get_batch_gen(val_ids)
        cond = None
        if config_name == "r_wave_gen":
            cond = wave_enc.extract_full(ctx)  # (B, M, 2d)
        elif config_name == "r_wave_real_gen":
            cond = wave_enc.extract_full_real_only(ctx)  # (B, M, d)
        elif config_name == "r_pool_gen":
            # 朴素池化: 用 model 的冻结 embedding
            emb = model.embed(ctx)  # (B, 256, d)
            B, L, D = emb.shape
            M = L // POOL_STRIDE
            emb = emb[:, :M * POOL_STRIDE].view(B, M, POOL_STRIDE, D)
            cond = emb.mean(dim=2)  # (B, M, d) - 未投影, proj 在 model 内
        # shifted: input = tgt[:-1], predict tgt[1:]
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        logits = model(tgt_in, cond)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))
        total_loss += loss.item() * tgt.shape[0]
        total_n += tgt.shape[0]
    model.train()
    return total_loss / total_n


# ============================================================================
# run_one
# ============================================================================
def run_one(config_name, seed, steps=TRAIN_STEPS):
    tag = f"{config_name}_s{seed}"
    out_json = RESULTS_DIR / f"{tag}.json"
    if out_json.exists():
        d = json.load(open(out_json))
        if d.get("train_steps") == steps and not d.get("diverged"):
            print(f"[skip] {tag} already done (best_val={d.get('best_val')})")
            return d

    cond_type = CONFIGS[config_name]
    print(f"\n{'='*70}\n[{tag}] config={config_name}  cond={cond_type}  "
          f"seed={seed}  steps={steps}\n{'='*70}")
    torch.manual_seed(seed)
    model = build_model(config_name).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {config_name}  trainable params: {n_params:,}")

    wave_enc = get_wave_encoder() if cond_type in ("wave", "wave_real") else None

    opt = torch.optim.AdamW(
        list(p for p in model.parameters() if p.requires_grad),
        lr=LR, weight_decay=WD, betas=(0.9, 0.95))
    train_ids, val_ids = load_data()

    results = {
        "config": config_name, "seed": seed, "params": n_params,
        "cond_type": cond_type, "train_steps": steps, "trace": [],
        "nan_step": None, "diverged": False,
    }
    t0 = time.time()

    for step in range(1, steps + 1):
        if step <= WARMUP_STEPS:
            lr = LR * step / WARMUP_STEPS
            for g in opt.param_groups:
                g["lr"] = lr

        ctx, tgt = get_batch_gen(train_ids)
        cond = None
        if cond_type == "wave":
            cond = wave_enc.extract_full(ctx)
        elif cond_type == "wave_real":
            cond = wave_enc.extract_full_real_only(ctx)
        elif cond_type == "pool":
            emb = model.embed(ctx)
            B, L, D = emb.shape
            M = L // POOL_STRIDE
            emb = emb[:, :M * POOL_STRIDE].view(B, M, POOL_STRIDE, D)
            cond = emb.mean(dim=2)  # (B, M, d) - 未投影

        # shifted: input = tgt[:-1], predict tgt[1:]
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        logits = model(tgt_in, cond)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))

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
            print(f"  step {step:>4}/{steps}  loss={loss.item():.4f}  "
                  f"|g|={gn:.2e}  t={time.time()-t0:.0f}s", flush=True)
        if step in EVAL_STEPS:
            vl = compute_val_loss(model, val_ids, wave_enc, config_name)
            ppl = math.exp(vl) if vl < 50 else float('inf')
            results["trace"].append({
                "step": step,
                "train_loss": round(loss.item(), 4),
                "val_loss": round(vl, 4),
                "val_ppl": round(ppl, 2) if ppl != float('inf') else None,
                "grad_norm": round(gn, 4),
            })
            print(f"  >>> val_loss={vl:.4f}  ppl={ppl:.2f}", flush=True)

    results["total_time_s"] = round(time.time() - t0, 2)
    best_val = min((t["val_loss"] for t in results["trace"]), default=None)
    results["best_val"] = best_val
    results["best_step"] = next(
        (t["step"] for t in results["trace"] if t["val_loss"] == best_val), None)

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
    print("VERDICT: 波场条件生成 - 冻结波场作为生成条件能否提供增益?")
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

    print("\n--- 结果 (best val_loss on target, lower=better) ---")
    means = {}
    for c in configs:
        vals = [b for b in bests[c] if b is not None]
        if vals:
            m = sum(vals) / len(vals)
            means[c] = m
            print(f"  {c:20s}: mean={m:.4f}  (seeds: {[round(b,4) if b else None for b in bests[c]]})")
        else:
            print(f"  {c:20s}: (no data)")

    print("\n--- 参数量 ---")
    for c in configs:
        p = RESULTS_DIR / f"{c}_s{seeds[0]}.json"
        if p.exists():
            d = json.load(open(p))
            print(f"  {c:20s}: {d.get('params', '?'):,}")

    print("\n--- 判决矩阵 ---")
    baseline = means.get("r_gen")
    pool_mean = means.get("r_pool_gen")
    wave_mean = means.get("r_wave_gen")
    wave_real_mean = means.get("r_wave_real_gen")

    per_condition = {}
    if baseline and pool_mean:
        ratio_p = pool_mean / baseline
        if ratio_p < 0.95:
            per_condition["pool_vs_base"] = "POOL_HELPFUL"
            print(f"  pool vs base: {pool_mean:.4f} / {baseline:.4f} = {ratio_p:.3f}  -> 朴素池化有效!")
        elif ratio_p > 1.05:
            per_condition["pool_vs_base"] = "POOL_HARMFUL"
            print(f"  pool vs base: {pool_mean:.4f} / {baseline:.4f} = {ratio_p:.3f}  -> 朴素池化有害")
        else:
            per_condition["pool_vs_base"] = "POOL_NEUTRAL"
            print(f"  pool vs base: {pool_mean:.4f} / {baseline:.4f} = {ratio_p:.3f}  -> 朴素池化中性")

    if baseline and wave_mean:
        ratio_w = wave_mean / baseline
        if ratio_w < 0.95:
            per_condition["wave_vs_base"] = "WAVE_HELPFUL"
            print(f"  wave vs base: {wave_mean:.4f} / {baseline:.4f} = {ratio_w:.3f}  -> 波场有效!")
        elif ratio_w > 1.05:
            per_condition["wave_vs_base"] = "WAVE_HARMFUL"
            print(f"  wave vs base: {wave_mean:.4f} / {baseline:.4f} = {ratio_w:.3f}  -> 波场有害")
        else:
            per_condition["wave_vs_base"] = "WAVE_NEUTRAL"
            print(f"  wave vs base: {wave_mean:.4f} / {baseline:.4f} = {ratio_w:.3f}  -> 波场中性")

    if wave_mean and wave_real_mean:
        ratio_c = wave_mean / wave_real_mean
        if ratio_c < 0.95:
            per_condition["complex_vs_real"] = "COMPLEX_BETTER"
            print(f"  complex vs real: {wave_mean:.4f} / {wave_real_mean:.4f} = {ratio_c:.3f}  -> 复数更好!")
        elif ratio_c > 1.05:
            per_condition["complex_vs_real"] = "REAL_BETTER"
            print(f"  complex vs real: {wave_mean:.4f} / {wave_real_mean:.4f} = {ratio_c:.3f}  -> 实数更好")
        else:
            per_condition["complex_vs_real"] = "NEUTRAL"
            print(f"  complex vs real: {wave_mean:.4f} / {wave_real_mean:.4f} = {ratio_c:.3f}  -> 中性")

    if wave_mean and pool_mean:
        ratio_wv = wave_mean / pool_mean
        if ratio_wv < 0.95:
            per_condition["wave_vs_pool"] = "WAVE_BETTER"
            print(f"  wave vs pool: {wave_mean:.4f} / {pool_mean:.4f} = {ratio_wv:.3f}  -> 波场优于朴素池化!")
        elif ratio_wv > 1.05:
            per_condition["wave_vs_pool"] = "POOL_BETTER"
            print(f"  wave vs pool: {wave_mean:.4f} / {pool_mean:.4f} = {ratio_wv:.3f}  -> 朴素池化优于波场")
        else:
            per_condition["wave_vs_pool"] = "NEUTRAL"
            print(f"  wave vs pool: {wave_mean:.4f} / {pool_mean:.4f} = {ratio_wv:.3f}  -> 中性")

    # 总判决
    if per_condition.get("wave_vs_base") == "WAVE_HELPFUL":
        if per_condition.get("wave_vs_pool") == "WAVE_BETTER":
            verdict = "WAVE_SPECIAL_GEN"
            print(f"\n  *** 总判决: {verdict} ***")
            print("  波场在生成中有效且优于朴素池化 -> 深挖波化生成器!")
        else:
            verdict = "GEN_GLOBAL_HELPFUL"
            print(f"\n  *** 总判决: {verdict} ***")
            print("  生成中全局信息有用但波场不特殊 -> 朴素方案足够")
    elif per_condition.get("wave_vs_base") == "WAVE_NEUTRAL":
        if per_condition.get("pool_vs_base") == "POOL_HELPFUL":
            verdict = "POOL_ONLY_GEN"
            print(f"\n  *** 总判决: {verdict} ***")
            print("  朴素池化有用但波场无用 -> 波场不如简单池化")
        else:
            verdict = "GEN_GLOBAL_DEAD"
            print(f"\n  *** 总判决: {verdict} ***")
            print("  生成中全局信息也无增量 -> 波化生成路线封死")
    else:
        verdict = "WAVE_HARMFUL_GEN"
        print(f"\n  *** 总判决: {verdict} ***")

    summary = {
        "means": means,
        "per_condition": per_condition,
        "overall_verdict": verdict,
    }
    (RESULTS_DIR / "exp27_verdict_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


# ============================================================================
# main
# ============================================================================
def main():
    p = argparse.ArgumentParser(description="Exp27: Wave Conditional Generation")
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
