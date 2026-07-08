"""
Exp 25: 冻结 Stage-A 波编码器注入 - 波场能否增强下游判别?
=============================================================

exp23/24 证明: 复数算子在判别/预测中相位是信息真空 (phase_std ≈ 均匀分布).
唯一存活的复数优势是 Stage A 压缩 (2.7x), 它的损失直接读整个波场.

本实验: 冻结 Stage A 编码器, 把 gauge-fixed 全局波场 code 注入实数 transformer,
测下游语言建模是否受益. 波场在它擅长的角色 (压缩全局结构), transformer 在
它擅长的角色 (逐位置判别).

设计:
  - 冻结 WaveTokenizerComplex.encode(bytes) -> psi (B, 64, 32) cfloat
  - gauge-fix: max-magnitude neuron 旋转到 0 (去全局相位, exp12 方法)
  - mean_pool over time -> z_global (B, 32) cfloat
  - split-real: cat[Re, Im] -> (B, 64) real  (或 real-only: Re -> (B, 32))
  - Linear(64, d_model) -> global bias, add to every position's embedding

3 条件:
  R:           plain real transformer (exp23 基座)
  R+wave:      + 冻结复数 wave code (gauge-fixed, split-real) 全局注入
  R+wave_real: + 冻结复数 wave code 的实部 only (控制: 复数 vs 实数)

诊断:
  - val PPL (主指标)
  - global probe: z_global -> linear -> next-token accuracy (波场含多少判别信息?)
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
WAVE_D = 32         # 编码器输出维度
WAVE_M = 64         # 编码器输出时间长度 (256/4)


# ============================================================================
# 数据: 复用 exp05 load_data, 标准 shifted LM
# ============================================================================
def get_batch_lm(ids, bs, sl, device=DEVICE):
    n = len(ids) - sl - 1
    starts = torch.randint(0, n, (bs,))
    x = torch.stack([ids[s:s + sl] for s in starts])
    y = torch.stack([ids[s + 1:s + sl + 1] for s in starts])
    return x.to(device), y.to(device)


# ============================================================================
# 冻结波编码器 + gauge-fix + 全局池化
# ============================================================================
class FrozenWaveEncoder(nn.Module):
    """冻结的 Stage A 波编码器, 提取全局 gauge-fixed 复数 code.

    encode(bytes) -> psi (B, M, d) cfloat
    gauge_fix(psi) -> 旋转使 max-magnitude neuron 的相位 = 0
    mean_pool(psi) -> z_global (B, d) cfloat
    """
    def __init__(self, ckpt_path=TOKENIZER_CKPT):
        super().__init__()
        self.tokenizer = WaveTokenizerComplex(vocab_size=VOCAB_SIZE, d=WAVE_D,
                                              seq_len=SEQ_LEN, stride=4)
        # 加载冻结权重 (checkpoint 中键前缀为 'tokenizer.')
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # 提取 tokenizer 子模块的权重
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
        """byte_ids (B, L) -> z_global (B, 2*d) real (split-real, gauge-fixed)."""
        psi = self.tokenizer.encode(byte_ids)  # (B, M, d) cfloat
        # gauge-fix: 旋转使 max-magnitude neuron (over M) 的相位 = 0
        # 对每个样本, 找到 |psi| 最大的时间位置, 用其相位旋转整个序列
        mag = psi.abs()  # (B, M, d)
        # 对每个样本, 取所有位置-维度的 max magnitude 的相位作为参考
        # 更简单: 对每个样本, 取 mean phase 作为参考 (exp12 方法的简化版)
        # exp12 用 mean direction: mu = sum(psi) / |sum(psi)|, 然后 psi * conj(mu)
        mu = psi.sum(dim=1)  # (B, d) cfloat - sum over time
        mu_mag = mu.abs()  # (B, d)
        mu_phase = mu / (mu_mag + 1e-8)  # (B, d) unit complex
        # 旋转: psi_fixed = psi * conj(mu_phase) (broadcast over M)
        psi_fixed = psi * mu_phase.conj().unsqueeze(1)  # (B, M, d)
        # mean pool over time
        z_global = psi_fixed.mean(dim=1)  # (B, d) cfloat
        # split-real
        z_flat = torch.cat([z_global.real, z_global.imag], dim=-1)  # (B, 2d) real
        return z_flat, z_global  # 返回复数版本用于 probe

    @torch.no_grad()
    def extract_global_real_only(self, byte_ids):
        """实部 only 版本: 只取 Re(psi_fixed), 丢弃虚部."""
        psi = self.tokenizer.encode(byte_ids)
        mu = psi.sum(dim=1)
        mu_phase = mu / (mu.abs() + 1e-8)
        psi_fixed = psi * mu_phase.conj().unsqueeze(1)
        z_global = psi_fixed.mean(dim=1)  # (B, d) cfloat
        z_flat = z_global.real  # (B, d) real only
        return z_flat, z_global


# ============================================================================
# 实数 Transformer (复用 exp23 设计, 手写 attention)
# ============================================================================
class RealAttention(nn.Module):
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
    def __init__(self, d, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = RealAttention(d, n_heads)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d_ff)
        self.fc2 = nn.Linear(d_ff, d)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class RealTransformer(nn.Module):
    """实数 transformer, 可选全局波场注入."""
    def __init__(self, d=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                 d_ff=D_FF, vocab=VOCAB_SIZE, seq_len=SEQ_LEN,
                 wave_inject=False, wave_real_only=False):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d) * 0.02)
        self.blocks = nn.ModuleList([
            RealTransformerBlock(d, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight  # weight tying

        self.wave_inject = wave_inject
        self.wave_real_only = wave_real_only
        if wave_inject:
            if wave_real_only:
                self.wave_proj = nn.Linear(WAVE_D, d)  # Re only -> (B, d)
            else:
                self.wave_proj = nn.Linear(2 * WAVE_D, d)  # split-real -> (B, d)

    def forward(self, idx, wave_code=None):
        # idx: (B, T), wave_code: (B, 2d) or (B, d) real, or None
        x = self.embed(idx) + self.pos
        if self.wave_inject and wave_code is not None:
            # 全局 bias: 投影到 d_model, 加到每个位置
            wave_bias = self.wave_proj(wave_code).unsqueeze(1)  # (B, 1, d)
            x = x + wave_bias
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)


# ============================================================================
# Global probe: z_global -> linear -> next-token
# ============================================================================
class GlobalProbe(nn.Module):
    """linear probe: z_global -> next-token logits."""
    def __init__(self, input_dim, vocab=VOCAB_SIZE):
        super().__init__()
        self.head = nn.Linear(input_dim, vocab)

    def forward(self, z):
        return self.head(z)  # (B, vocab)


# ============================================================================
# CONFIGS
# ============================================================================
def build_r_real():
    return RealTransformer(wave_inject=False)

def build_r_wave():
    return RealTransformer(wave_inject=True, wave_real_only=False)

def build_r_wave_real():
    return RealTransformer(wave_inject=True, wave_real_only=True)

CONFIGS = {
    "r_real":      build_r_real,
    "r_wave":      build_r_wave,
    "r_wave_real": build_r_wave_real,
}

# 全局波编码器 (所有条件共享, 但仅 r_wave/r_wave_real 使用)
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
        if config_name in ("r_wave", "r_wave_real"):
            if config_name == "r_wave":
                wave_code, _ = wave_enc.extract_global(x)
            else:
                wave_code, _ = wave_enc.extract_global_real_only(x)
        logits = model(x, wave_code)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        total_loss += loss.item() * x.shape[0]
        total_n += x.shape[0]
    model.train()
    return total_loss / total_n


@torch.no_grad()
def global_probe_eval(probe, wave_enc, val_ids, real_only, n_batches=20):
    """global probe: z_global -> linear -> predict last token.
    返回 accuracy (%) 和 val_loss."""
    probe.eval()
    total_loss, total_n, correct, total = 0.0, 0, 0, 0
    for _ in range(n_batches):
        x, y = get_batch_lm(val_ids, BATCH_SIZE, SEQ_LEN)
        if real_only:
            z_flat, _ = wave_enc.extract_global_real_only(x)
        else:
            z_flat, _ = wave_enc.extract_global(x)
        # 用 z_flat 预测序列的最后一个 token (即 y 的最后一个)
        logits = probe(z_flat)  # (B, vocab)
        last_y = y[:, -1]  # (B,)
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

    wave_enc = get_wave_encoder() if config_name in ("r_wave", "r_wave_real") else None
    real_only = config_name == "r_wave_real"

    # global probe (仅波场条件)
    probe = None
    if config_name in ("r_wave", "r_wave_real"):
        probe_dim = WAVE_D if real_only else 2 * WAVE_D
        probe = GlobalProbe(probe_dim).to(DEVICE)
        probe_params = list(probe.parameters())
    else:
        probe_params = []

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
        if config_name in ("r_wave", "r_wave_real"):
            if real_only:
                wave_code, z_global = wave_enc.extract_global_real_only(x)
            else:
                wave_code, z_global = wave_enc.extract_global(x)

        logits = model(x, wave_code)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))

        # global probe loss (独立, 不反传到 model)
        probe_loss = torch.tensor(0.0, device=DEVICE)
        if probe is not None and z_global is not None:
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
            probe_loss_val = None
            if probe is not None:
                pl, pa = global_probe_eval(probe, wave_enc, val_ids, real_only)
                probe_loss_val = pl
                probe_acc = pa
            results["trace"].append({
                "step": step,
                "train_loss": round(loss.item(), 4),
                "val_loss": round(vl, 4),
                "val_ppl": round(ppl, 2) if ppl != float('inf') else None,
                "probe_acc": round(probe_acc, 4) if probe_acc is not None else None,
                "probe_loss": round(probe_loss_val, 4) if probe_loss_val is not None else None,
                "grad_norm": round(gn, 4),
            })
            print(f"  >>> val_loss={vl:.4f}  ppl={ppl:.2f}  "
                  f"probe_acc={probe_acc}  probe_loss={probe_loss_val}", flush=True)

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
    print("VERDICT: 冻结波编码器注入 - 波场能否增强下游判别?")
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
            print(f"  {c:14s}: mean={m:.4f}  (seeds: {[round(b,4) if b else None for b in bests[c]]})")
        else:
            print(f"  {c:14s}: (no data)")

    print("\n--- 参数量 ---")
    for c in configs:
        p = RESULTS_DIR / f"{c}_s{seeds[0]}.json"
        if p.exists():
            d = json.load(open(p))
            print(f"  {c:14s}: {d.get('params', '?'):,}")

    print("\n--- global probe accuracy (波场含多少判别信息?) ---")
    for c in ["r_wave", "r_wave_real"]:
        vals = [a for a in probe_accs[c] if a is not None]
        if vals:
            m = sum(vals) / len(vals)
            print(f"  {c:14s}: probe_acc mean={m:.4f}  "
                  f"(random=0.0039, seeds: {probe_accs[c]})")

    print("\n--- 判决 ---")
    baseline = means.get("r_real")
    wave_mean = means.get("r_wave")
    wave_real_mean = means.get("r_wave_real")

    per_condition = {}
    if baseline and wave_mean:
        ratio = wave_mean / baseline
        if ratio < 0.95:
            per_condition["r_wave"] = "WAVE_HELPFUL"
            print(f"  r_wave: {wave_mean:.4f} / {baseline:.4f} = {ratio:.3f}  -> 波场有效!")
        elif ratio > 1.05:
            per_condition["r_wave"] = "WAVE_HARMFUL"
            print(f"  r_wave: {wave_mean:.4f} / {baseline:.4f} = {ratio:.3f}  -> 波场有害")
        else:
            per_condition["r_wave"] = "NEUTRAL"
            print(f"  r_wave: {wave_mean:.4f} / {baseline:.4f} = {ratio:.3f}  -> 中性")

    if wave_mean and wave_real_mean:
        ratio2 = wave_mean / wave_real_mean
        if ratio2 < 0.95:
            per_condition["complex_vs_real"] = "COMPLEX_BETTER"
            print(f"  complex vs real: {wave_mean:.4f} / {wave_real_mean:.4f} = {ratio2:.3f}  -> 复数更好!")
        elif ratio2 > 1.05:
            per_condition["complex_vs_real"] = "REAL_BETTER"
            print(f"  complex vs real: {wave_mean:.4f} / {wave_real_mean:.4f} = {ratio2:.3f}  -> 实数更好")
        else:
            per_condition["complex_vs_real"] = "NEUTRAL"
            print(f"  complex vs real: {wave_mean:.4f} / {wave_real_mean:.4f} = {ratio2:.3f}  -> 中性")

    if per_condition.get("r_wave") == "WAVE_HELPFUL":
        verdict = "WAVE_INJECTION_VIABLE"
    elif per_condition.get("r_wave") == "NEUTRAL":
        verdict = "WAVE_INJECTION_NEUTRAL"
    else:
        verdict = "WAVE_INJECTION_DEAD"
    print(f"\n  *** 总判决: {verdict} ***")

    summary = {
        "means": means,
        "per_condition": per_condition,
        "overall_verdict": verdict,
    }
    (RESULTS_DIR / "exp25_verdict_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


# ============================================================================
# main
# ============================================================================
def main():
    p = argparse.ArgumentParser(description="Exp25: Frozen Wave Injection")
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
