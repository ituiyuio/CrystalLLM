"""
Exp 26: 局部 attention + 全局波场 - 信息不对称能否产生增益?
=============================================================

exp25 证明: 冻结波场全局注入对全注意力 transformer 无增益 (probe 23.3% 但信息冗余).
原因: transformer 的逐位置 attention 已能从同窗口提取全局上下文.

本实验: 强制 transformer 只看局部窗口 (前 64 token), 波场看完整 256 token.
制造信息不对称: 波场提供 transformer 看不到的信息.

4 条件:
  R_local:           局部 transformer, 无全局注入
  R_local+pool:      局部 transformer + 朴素实数 mean-pool 注入 (冻结 embedding)
  R_local+wave:      局部 transformer + 冻结复数 wave code 注入 (gauge-fixed, split-real)
  R_local+wave_real: 局部 transformer + 冻结 wave code 实部 only

判决矩阵:
  R_local+wave > R_local, R_local+pool ≈ R_local  -> 波场特殊, 深挖
  R_local+wave ≈ R_local+pool > R_local          -> 任何全局信息有用, 波场不特殊
  全部 ≈ R_local                                  -> 全局信息无增量, 长上下文封死
  R_local+wave > R_local+wave_real               -> 复数在受限设定下有额外优势
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

SEQ_LEN = 256
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
WAVE_M = 64
LOCAL_WINDOW = 64  # 局部 attention 窗口大小


# ============================================================================
# 数据
# ============================================================================
def get_batch_lm(ids, bs, sl, device=DEVICE):
    n = len(ids) - sl - 1
    starts = torch.randint(0, n, (bs,))
    x = torch.stack([ids[s:s + sl] for s in starts])
    y = torch.stack([ids[s + 1:s + sl + 1] for s in starts])
    return x.to(device), y.to(device)


# ============================================================================
# 冻结波编码器 (复用 exp25)
# ============================================================================
class FrozenWaveEncoder(nn.Module):
    def __init__(self, ckpt_path=TOKENIZER_CKPT):
        super().__init__()
        self.tokenizer = WaveTokenizerComplex(vocab_size=VOCAB_SIZE, d=WAVE_D,
                                              seq_len=SEQ_LEN, stride=4)
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
    def extract_global(self, byte_ids):
        """split-real gauge-fixed global code."""
        psi = self.tokenizer.encode(byte_ids)  # (B, M, d) cfloat
        mu = psi.sum(dim=1)  # (B, d) cfloat
        mu_phase = mu / (mu.abs() + 1e-8)
        psi_fixed = psi * mu_phase.conj().unsqueeze(1)
        z_global = psi_fixed.mean(dim=1)  # (B, d) cfloat
        z_flat = torch.cat([z_global.real, z_global.imag], dim=-1)  # (B, 2d)
        return z_flat, z_global

    @torch.no_grad()
    def extract_global_real_only(self, byte_ids):
        psi = self.tokenizer.encode(byte_ids)
        mu = psi.sum(dim=1)
        mu_phase = mu / (mu.abs() + 1e-8)
        psi_fixed = psi * mu_phase.conj().unsqueeze(1)
        z_global = psi_fixed.mean(dim=1)
        z_flat = z_global.real  # (B, d) real only
        return z_flat, z_global


# ============================================================================
# 局部 Attention
# ============================================================================
def make_local_causal_mask(seq_len, window):
    """局部因果 mask: 位置 i 只能看到 [max(0, i-window+1), i].

    返回 bool mask: True = 屏蔽 (置 -inf).
    """
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool)  # 全屏蔽
    for i in range(seq_len):
        start = max(0, i - window + 1)
        mask[i, start:i + 1] = False  # 允许看 [start, i]
    return mask


class LocalAttention(nn.Module):
    """多头注意力, 使用局部因果 mask (只看前 window 个 token)."""
    def __init__(self, d, n_heads, seq_len=SEQ_LEN, window=LOCAL_WINDOW):
        super().__init__()
        assert d % n_heads == 0
        self.nh = n_heads
        self.dh = d // n_heads
        self.scale = 1.0 / math.sqrt(self.dh)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        mask = make_local_causal_mask(seq_len, window)
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


class RealTransformerBlock(nn.Module):
    def __init__(self, d, n_heads, d_ff, window=LOCAL_WINDOW):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = LocalAttention(d, n_heads, window=window)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d_ff)
        self.fc2 = nn.Linear(d_ff, d)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


# ============================================================================
# Transformer with global injection
# ============================================================================
class RealTransformer(nn.Module):
    """局部 attention transformer, 可选全局注入.

    injection_type:
      'none':      无注入
      'pool':      朴素实数 mean-pool (冻结 embedding -> mean -> Linear -> bias)
      'wave':      冻结复数 wave code (gauge-fixed, split-real)
      'wave_real': 冻结 wave code 实部 only
    """
    def __init__(self, d=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                 d_ff=D_FF, vocab=VOCAB_SIZE, seq_len=SEQ_LEN,
                 injection_type="none"):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d) * 0.02)
        self.blocks = nn.ModuleList([
            RealTransformerBlock(d, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight  # weight tying

        self.injection_type = injection_type
        if injection_type == "pool":
            # 朴素池化: 冻结 embedding, mean-pool, Linear 投影
            self.pool_proj = nn.Linear(d, d, bias=False)
            # 冻结 embedding (不随 probe/transformer 训练变化)
            self.embed.weight.requires_grad = False
        elif injection_type == "wave":
            self.wave_proj = nn.Linear(2 * WAVE_D, d)
        elif injection_type == "wave_real":
            self.wave_proj = nn.Linear(WAVE_D, d)

    def forward(self, idx, global_code=None):
        x = self.embed(idx) + self.pos
        if self.injection_type == "pool":
            # 朴素池化: 用冻结的 embedding 做 mean-pool
            with torch.no_grad():
                emb = self.embed(idx)  # (B, T, d) - 冻结
                pooled = emb.mean(dim=1)  # (B, d)
            bias = self.pool_proj(pooled).unsqueeze(1)  # (B, 1, d)
            x = x + bias
        elif global_code is not None:
            bias = self.wave_proj(global_code).unsqueeze(1)
            x = x + bias
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)


# ============================================================================
# Global probe
# ============================================================================
class GlobalProbe(nn.Module):
    def __init__(self, input_dim, vocab=VOCAB_SIZE):
        super().__init__()
        self.head = nn.Linear(input_dim, vocab)

    def forward(self, z):
        return self.head(z)


# ============================================================================
# CONFIGS
# ============================================================================
def build_r_local():
    return RealTransformer(injection_type="none")

def build_r_local_pool():
    return RealTransformer(injection_type="pool")

def build_r_local_wave():
    return RealTransformer(injection_type="wave")

def build_r_local_wave_real():
    return RealTransformer(injection_type="wave_real")

CONFIGS = {
    "r_local":           build_r_local,
    "r_local_pool":      build_r_local_pool,
    "r_local_wave":      build_r_local_wave,
    "r_local_wave_real": build_r_local_wave_real,
}

_wave_encoder = None

def get_wave_encoder():
    global _wave_encoder
    if _wave_encoder is None:
        _wave_encoder = FrozenWaveEncoder().to(DEVICE)
    return _wave_encoder

def build_model(config_name):
    return CONFIGS[config_name]()


# ============================================================================
# 诊断
# ============================================================================
@torch.no_grad()
def compute_val_loss(model, val_ids, wave_enc, config_name, n_batches=20):
    model.eval()
    total_loss, total_n = 0.0, 0
    for _ in range(n_batches):
        x, y = get_batch_lm(val_ids, BATCH_SIZE, SEQ_LEN)
        wave_code = None
        if config_name in ("r_local_wave", "r_local_wave_real"):
            wave_enc_fn = (wave_enc.extract_global_real_only
                           if config_name == "r_local_wave_real"
                           else wave_enc.extract_global)
            wave_code, _ = wave_enc_fn(x)
        logits = model(x, wave_code)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        total_loss += loss.item() * x.shape[0]
        total_n += x.shape[0]
    model.train()
    return total_loss / total_n


@torch.no_grad()
def global_probe_eval(probe, wave_enc, val_ids, real_only, n_batches=20):
    probe.eval()
    total_loss, total_n, correct, total = 0.0, 0, 0, 0
    for _ in range(n_batches):
        x, y = get_batch_lm(val_ids, BATCH_SIZE, SEQ_LEN)
        if real_only:
            z_flat, _ = wave_enc.extract_global_real_only(x)
        else:
            z_flat, _ = wave_enc.extract_global(x)
        logits = probe(z_flat)
        last_y = y[:, -1]
        loss = F.cross_entropy(logits, last_y)
        total_loss += loss.item() * x.shape[0]
        total_n += x.shape[0]
        pred = logits.argmax(dim=-1)
        correct += (pred == last_y).sum().item()
        total += x.shape[0]
    probe.train()
    return total_loss / total_n, correct / total


@torch.no_grad()
def pool_probe_eval(probe, model, val_ids, n_batches=20):
    """probe for naive pool: mean-pool frozen embedding -> linear -> last token."""
    probe.eval()
    total_loss, total_n, correct, total = 0.0, 0, 0, 0
    for _ in range(n_batches):
        x, y = get_batch_lm(val_ids, BATCH_SIZE, SEQ_LEN)
        with torch.no_grad():
            emb = model.embed(x)  # (B, T, d) frozen
            pooled = emb.mean(dim=1)  # (B, d)
        logits = probe(pooled)
        last_y = y[:, -1]
        loss = F.cross_entropy(logits, last_y)
        total_loss += loss.item() * x.shape[0]
        total_n += x.shape[0]
        pred = logits.argmax(dim=-1)
        correct += (pred == last_y).sum().item()
        total += x.shape[0]
    probe.train()
    return total_loss / total_n, correct / total


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

    print(f"\n{'='*70}\n[{tag}] config={config_name}  seed={seed}  "
          f"steps={steps}\n{'='*70}")
    torch.manual_seed(seed)
    model = build_model(config_name).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {config_name}  trainable params: {n_params:,}")

    wave_enc = None
    real_only = False
    probe = None
    if config_name in ("r_local_wave", "r_local_wave_real"):
        wave_enc = get_wave_encoder()
        real_only = config_name == "r_local_wave_real"
        probe_dim = WAVE_D if real_only else 2 * WAVE_D
        probe = GlobalProbe(probe_dim).to(DEVICE)
    elif config_name == "r_local_pool":
        probe = GlobalProbe(D_MODEL).to(DEVICE)  # pool probe: d_model input
    probe_params = list(probe.parameters()) if probe is not None else []

    opt = torch.optim.AdamW(
        list(p for p in model.parameters() if p.requires_grad) + probe_params,
        lr=LR, weight_decay=WD, betas=(0.9, 0.95))
    train_ids, val_ids = load_data()

    results = {
        "config": config_name, "seed": seed, "params": n_params,
        "train_steps": steps, "trace": [],
        "nan_step": None, "diverged": False,
    }
    t0 = time.time()

    for step in range(1, steps + 1):
        if step <= WARMUP_STEPS:
            lr = LR * step / WARMUP_STEPS
            for g in opt.param_groups:
                g["lr"] = lr

        x, y = get_batch_lm(train_ids, BATCH_SIZE, SEQ_LEN)
        wave_code = None
        z_global_for_probe = None
        if config_name in ("r_local_wave", "r_local_wave_real"):
            fn = (wave_enc.extract_global_real_only if real_only
                  else wave_enc.extract_global)
            wave_code, z_global_for_probe = fn(x)

        logits = model(x, wave_code)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))

        # probe loss
        probe_loss = torch.tensor(0.0, device=DEVICE)
        if probe is not None:
            if config_name == "r_local_pool":
                with torch.no_grad():
                    emb = model.embed(x)
                    pooled = emb.mean(dim=1)
                probe_logits = probe(pooled)
            else:
                probe_logits = probe(wave_code)
            probe_loss = F.cross_entropy(probe_logits, y[:, -1])

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 250 == 0 or step == 100:
            print(f"  step {step:>4}/{steps}  loss={loss.item():.4f}  "
                  f"probe={probe_loss.item():.4f}  |g|={gn:.2e}  "
                  f"t={time.time()-t0:.0f}s", flush=True)
        if step in EVAL_STEPS:
            vl = compute_val_loss(model, val_ids, wave_enc, config_name)
            ppl = math.exp(vl) if vl < 50 else float('inf')
            probe_acc = None
            if probe is not None:
                if config_name == "r_local_pool":
                    _, pa = pool_probe_eval(probe, model, val_ids)
                else:
                    _, pa = global_probe_eval(probe, wave_enc, val_ids, real_only)
                probe_acc = pa
            results["trace"].append({
                "step": step,
                "train_loss": round(loss.item(), 4),
                "val_loss": round(vl, 4),
                "val_ppl": round(ppl, 2) if ppl != float('inf') else None,
                "probe_acc": round(probe_acc, 4) if probe_acc is not None else None,
                "grad_norm": round(gn, 4),
            })
            print(f"  >>> val_loss={vl:.4f}  ppl={ppl:.2f}  "
                  f"probe_acc={probe_acc}", flush=True)

    results["total_time_s"] = round(time.time() - t0, 2)
    best_val = min((t["val_loss"] for t in results["trace"]), default=None)
    results["best_val"] = best_val
    results["best_step"] = next(
        (t["step"] for t in results["trace"] if t["val_loss"] == best_val), None)
    if probe is not None:
        best_probe_acc = max(
            (t["probe_acc"] for t in results["trace"]
             if t.get("probe_acc") is not None), default=None)
        results["best_probe_acc"] = best_probe_acc

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[saved] -> {out_json}  (best_val={best_val})")
    del model
    if probe is not None:
        del probe
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return results


# ============================================================================
# compute_verdict
# ============================================================================
def compute_verdict(seeds=tuple(SEEDS_DEFAULT)):
    print("\n" + "=" * 70)
    print("VERDICT: 局部 attention + 全局波场 - 信息不对称能否产生增益?")
    print("=" * 70)

    configs = list(CONFIGS.keys())
    bests = {c: [] for c in configs}
    probe_accs = {c: [] for c in configs}
    for c in configs:
        for seed in seeds:
            p = RESULTS_DIR / f"{c}_s{seed}.json"
            if not p.exists():
                bests[c].append(None)
                probe_accs[c].append(None)
                continue
            d = json.load(open(p))
            bests[c].append(d.get("best_val"))
            probe_accs[c].append(d.get("best_probe_acc"))

    print("\n--- 结果 (best val_loss, lower=better) ---")
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

    print("\n--- global probe accuracy ---")
    for c in ["r_local_pool", "r_local_wave", "r_local_wave_real"]:
        vals = [a for a in probe_accs[c] if a is not None]
        if vals:
            m = sum(vals) / len(vals)
            print(f"  {c:20s}: probe_acc mean={m:.4f}  (random=0.0039, seeds: {probe_accs[c]})")

    print("\n--- 判决矩阵 ---")
    baseline = means.get("r_local")
    pool_mean = means.get("r_local_pool")
    wave_mean = means.get("r_local_wave")
    wave_real_mean = means.get("r_local_wave_real")

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

    # 总判决
    if per_condition.get("wave_vs_base") == "WAVE_HELPFUL":
        if per_condition.get("pool_vs_base") in ("POOL_NEUTRAL", "POOL_HARMFUL"):
            verdict = "WAVE_SPECIAL"  # 波场特殊: 有增益而朴素池化无
            print(f"\n  *** 总判决: {verdict} ***")
            print("  波场提供朴素池化没有的信息 -> 深挖长上下文!")
        else:
            verdict = "GLOBAL_HELPFUL"  # 任何全局信息都有用
            print(f"\n  *** 总判决: {verdict} ***")
            print("  全局信息有用但波场不特殊 -> 朴素方案更简单")
    elif per_condition.get("wave_vs_base") == "WAVE_NEUTRAL":
        if per_condition.get("pool_vs_base") == "POOL_HELPFUL":
            verdict = "POOL_ONLY"  # 朴素池化有用但波场无用
            print(f"\n  *** 总判决: {verdict} ***")
            print("  朴素池化有用但波场无用 -> 波场不如简单池化")
        else:
            verdict = "GLOBAL_DEAD"  # 全局信息无增量
            print(f"\n  *** 总判决: {verdict} ***")
            print("  全局信息无增量 -> 长上下文路线封死")
    else:
        verdict = "WAVE_HARMFUL"
        print(f"\n  *** 总判决: {verdict} ***")

    summary = {
        "means": means,
        "per_condition": per_condition,
        "overall_verdict": verdict,
    }
    (RESULTS_DIR / "exp26_verdict_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


# ============================================================================
# main
# ============================================================================
def main():
    p = argparse.ArgumentParser(description="Exp26: Local Attn + Wave Probe")
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
