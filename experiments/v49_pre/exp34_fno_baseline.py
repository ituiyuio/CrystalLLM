"""
Exp 34: FNO baseline for byte-level LM (验证波表示 vs Transformer 工程可行性)

设计:
  - 1D FNO 架构 (Li 2020 风格, 论文 §3.2)
  - 输入: byte sequence (vocab=256) → embed → d_model
  - FNO layer x N: spectral_conv + local_conv(1x1) + LayerNorm + skip
  - 输出: next-byte logits

关键差异 vs Transformer:
  - 无 attention (O(N²) → O(N log N) FFT)
  - 全局依赖通过频域 FFT 实现 (用户理论的"频域离散化")
  - 模态数 modes 是超参, 与序列长度 N 解耦
  - "频域截断 = 用户理论的 implicit bandlimit"

vs exp32 (MiniGPT 50M) 的差异:
  - 模型: MiniGPT (Transformer) → FNO1d (no attention)
  - 复用 v28 parquet 数据加载 + UTF-8 encode

判决标准 (smoke 500 步):
  - TF PPL 持续下降 + 无 NaN → 工程可行, 跑长程 (8k) 对照
  - NaN / PPL 不下降 → FNO 在 LM 上不可行, 归档
  - 注意: 本 smoke 用极小模型 (~320K params), 不是 50M 严格对照
"""
import argparse
import math
import time
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

torch.manual_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_PARQUET = PROJECT_ROOT / "crystalllm" / "data" / "processed" / "v28_train.parquet"
VAL_PARQUET = PROJECT_ROOT / "crystalllm" / "data" / "processed" / "v28_val.parquet"
VOCAB_SIZE = 256
RESULTS_DIR = PROJECT_ROOT / "experiments" / "v49_pre" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 复用 exp32 数据加载
# ============================================================================
def load_data(train_n_bytes=2_000_000, val_n_bytes=100_000):
    """从 v28 parquet 读 text → UTF-8 bytes → tensor[int64]."""
    print(f"[load] reading {TRAIN_PARQUET.name} ...")
    train_text = "\n".join(pd.read_parquet(TRAIN_PARQUET)["text"].astype(str).tolist())
    val_text = "\n".join(pd.read_parquet(VAL_PARQUET)["text"].astype(str).tolist())
    train_bytes = train_text.encode("utf-8")[:train_n_bytes]
    val_bytes = val_text.encode("utf-8")[:val_n_bytes]
    print(f"[load] train bytes: {len(train_bytes):,}  val bytes: {len(val_bytes):,}")
    return (torch.tensor(list(train_bytes), dtype=torch.int64),
            torch.tensor(list(val_bytes), dtype=torch.int64))


def get_batch(ids, bs, sl):
    """随机采样 batch: input/output 错位一格."""
    n = len(ids) - sl - 1
    starts = torch.randint(0, n, (bs,))
    x = torch.stack([ids[s:s + sl] for s in starts])
    y = torch.stack([ids[s + 1:s + sl + 1] for s in starts])
    return x.to(DEVICE), y.to(DEVICE)


# ============================================================================
# 核心: SpectralConv1d — 频域卷积 (FNO 1D 论文 §3.2)
# ============================================================================
class SpectralConv1d(nn.Module):
    """1D 谱卷积层.

    数学:
      x_ft   = RFFT(x)                              # (B, C, T//2+1)
      y_ft[m]= sum_c W[m,c,o] * x_ft[c,o]  for o < modes   # 截断前 modes
      y      = IRFFT(y_ft, n=T)                     # (B, C, T)

    关键: 截断高频 = 用户理论的"频域离散化",模态数 modes << T//2+1
    ponytail: complex weights 单独 init (scale=1/(C*modes)), 数值稳定.
    """
    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (channels * modes)
        # complex-valued learnable weights, one per (in_ch, out_ch, mode)
        self.weights = nn.Parameter(
            scale * torch.randn(channels, channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)  # (B, C, T//2+1)
        n_freq = x_ft.size(-1)
        # ponytail: 防止 T < 2*modes 时越界 (argmax rollout seq_len=1 时触发)
        eff_modes = min(self.modes, n_freq)
        x_ft_trunc = x_ft[:, :, :eff_modes]
        # per-mode linear: einsum over channels, slice weights to eff_modes
        out_ft = torch.einsum(
            'bct,cot->bot', x_ft_trunc, self.weights[:, :, :eff_modes]
        )
        # pad zeros for higher modes (FNO paper §3.2: modes beyond truncation = 0)
        full_ft = torch.zeros(
            B, out_ft.size(1), n_freq,
            dtype=torch.cfloat, device=x.device
        )
        full_ft[:, :, :eff_modes] = out_ft
        # IFFT back to time domain
        out = torch.fft.irfft(full_ft, n=T, dim=-1)
        return out


# ============================================================================
# FNO1d model — minimal 1D FNO for byte-level LM
# ============================================================================
class FNO1d(nn.Module):
    """最小 1D FNO for byte-level LM.

    架构:
      embed + pos_embed → [spectral + local + LN + skip] x N → head

    关键差异 vs Transformer:
      - 无 attention (无 O(N²) 全局依赖矩阵)
      - 全局依赖通过 SpectralConv1d 的 FFT 实现 (O(N log N))
      - "频域模态数 modes" 是超参, 与序列长度 N 解耦

    加 learned positional encoding 因为 FNO 本身是 permutation-equivariant
    (Fourier transform 不编码 grid 的绝对位置)
    """
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=64, n_layers=4,
                 modes=16, max_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.modes = modes

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

        self.spec_convs = nn.ModuleList([
            SpectralConv1d(d_model, modes) for _ in range(n_layers)
        ])
        self.locals = nn.ModuleList([
            nn.Conv1d(d_model, d_model, 1) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

        self.head = nn.Linear(d_model, vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S) token ids
        B, S = x.shape
        pos = torch.arange(S, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)  # (B, S, d_model)
        return self.forward_from_embed(h)

    def forward_from_embed(self, h: torch.Tensor) -> torch.Tensor:
        """Forward from pre-computed embedding (B, S, d_model).

        ponytail: 用于 Soft-Exp inference — soft feedback (probs @ embed.weight)
        产生 (B, S-1, D) 嵌入,需要直接喂入 spectral 层,不能 token-ize.
        """
        h = h.transpose(1, 2)  # (B, d_model, S)
        for spec, local, norm in zip(self.spec_convs, self.locals, self.norms):
            h1 = spec(h) + local(h)
            h = norm(h1.transpose(1, 2)).transpose(1, 2) + h
        logits = self.head(h.transpose(1, 2))  # (B, S, vocab)
        return logits


# ============================================================================
# Eval: teacher-forcing PPL + autoregressive argmax PPL (暴露偏差诊断)
# ============================================================================
@torch.no_grad()
def eval_tf_ppl(model, val_ids, num_seqs=80, seq_len=64):
    model.eval()
    n = len(val_ids) - seq_len - 1
    starts = torch.randint(0, n, (num_seqs,))
    total_loss, total_tokens = 0.0, 0
    for s in starts:
        ids = val_ids[s:s + seq_len + 1].to(DEVICE).unsqueeze(0)
        logits = model(ids[:, :-1])
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), ids[:, 1:].view(-1), reduction="sum"
        )
        total_loss += loss.item()
        total_tokens += seq_len
    model.train()
    return math.exp(total_loss / total_tokens)


@torch.no_grad()
def eval_argmax_ppl(model, val_ids, num_seqs=20, seq_len=64):
    """Sliding-window autoregressive argmax PPL — 暴露偏差诊断.

    ponytail root-cause fix: 不能用 cat append 让 cur 越过 max_len=64,
    否则 pos_embed 索引 64 越界 (silently truncated by nn.Embedding impl).
    用 sliding window: cur = cat([cur[:, 1:], fb], dim=1), 始终保持 seq_len.
    exp32v_line193-219: 同款模式.
    """
    model.eval()
    n = len(val_ids) - seq_len - 1
    starts = torch.randint(0, n, (num_seqs,))
    total_loss, total_tokens = 0.0, 0
    for s in starts:
        ids = val_ids[s:s + seq_len + 1].to(DEVICE).unsqueeze(0)
        cur = ids[:, :seq_len]  # (1, seq_len) — full ground-truth window
        for t in range(seq_len):
            logits = model(cur)[:, -1, :]  # next-token logits
            target = ids[:, t + 1]  # ground truth at position t+1
            total_loss += F.cross_entropy(
                logits, target, reduction="sum"
            ).item()
            total_tokens += 1
            next_id = logits.argmax(dim=-1, keepdim=True)  # (1, 1)
            cur = torch.cat([cur[:, 1:], next_id], dim=1)  # sliding window
    model.train()
    return math.exp(total_loss / total_tokens)


@torch.no_grad()
def eval_soft_ppl(model, val_ids, num_seqs=20, seq_len=64):
    """Sliding-window autoregressive eval with Soft-Exp (probs @ embed) feedback.

    复用 exp32 feedback_soft 模式: probs = softmax(logits), fb = probs @ embed.weight.
    ponytail: 这是 v50 = V49 + Soft-Exp 推理的核对点.
    cur 保持 (B, S, D) embedding 状态,而不是 token ids.
    """
    model.eval()
    n = len(val_ids) - seq_len - 1
    starts = torch.randint(0, n, (num_seqs,))
    total_loss, total_tokens = 0.0, 0
    pos = torch.arange(seq_len, device=DEVICE).unsqueeze(0)
    for s in starts:
        ids = val_ids[s:s + seq_len + 1].to(DEVICE).unsqueeze(0)
        # initial: token embed + pos embed (sliding window, ground truth first)
        cur_emb = model.embed(ids[:, :seq_len]) + model.pos_embed(pos)  # (1, S, D)
        for t in range(seq_len):
            logits = model.forward_from_embed(cur_emb)[:, -1, :]  # next-token logits
            target = ids[:, t + 1]
            total_loss += F.cross_entropy(logits, target, reduction="sum").item()
            total_tokens += 1
            # Soft-Exp feedback: probs @ embed.weight
            probs = F.softmax(logits.float(), dim=-1)
            fb = torch.matmul(probs, model.embed.weight)  # (1, d_model)
            cur_emb = torch.cat([cur_emb[:, 1:, :], fb.unsqueeze(1)], dim=1)
    model.train()
    return math.exp(total_loss / total_tokens)


# ============================================================================
# Main: smoke test
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Exp34: FNO byte-level baseline smoke")
    parser.add_argument("--steps", type=int, default=8000,
                       help="8k 步已证明 FNO 进入 LM 区域 (exp32 byte-level 8k)")
    parser.add_argument("--d_model", type=int, default=256,
                       help="17.5M 模型规模 (bottleneck 诊断: 模型 54x → gap -3.5x)")
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--peak_lr", type=float, default=5e-5,
                       help="exp32 验证 lr=5e-5 在 byte-level 上稳定; 5e-4 训练 loss 震荡")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--out", type=str,
                       default=str(RESULTS_DIR / "exp34_fno_results.json"))
    parser.add_argument("--train_n_bytes", type=int, default=2_000_000,
                       help="训练字节数 (默认 2M)")
    args = parser.parse_args()

    train_ids, val_ids = load_data(train_n_bytes=args.train_n_bytes)
    print(f"[data] train={len(train_ids):,} bytes  val={len(val_ids):,} bytes")

    torch.manual_seed(42)
    model = FNO1d(
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        n_layers=args.n_layers,
        modes=args.modes,
        max_len=args.seq_len,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] FNO1d: d_model={args.d_model} layers={args.n_layers} "
          f"modes={args.modes} seq_len={args.seq_len}")
    print(f"[model] params: {n_params:,} ({n_params/1e6:.2f}M)")
    # ponytail: smoke 模型极小, 严格对照需另跑 --d_model 256/512

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.peak_lr,
        weight_decay=0.01, betas=(0.9, 0.95),
    )

    results = {
        "name": "fno_byte_baseline",
        "params": n_params,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "modes": args.modes,
        "seq_len": args.seq_len,
        "peak_lr": args.peak_lr,
        "vocab_size": VOCAB_SIZE,
        "checkpoints": [],
    }
    t0 = time.time()

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    nan_hit = False
    for step in range(1, args.steps + 1):
        lr = args.peak_lr * min(step / args.warmup, 1.0)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_ids, args.batch_size, args.seq_len)
        model.train()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        if torch.isnan(loss):
            print(f"!!! NaN at step {step}, aborting")
            nan_hit = True
            results["nan_step"] = step
            break

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0 or step == 1:
            mem = (torch.cuda.max_memory_allocated() / 1e9) if DEVICE == "cuda" else 0
            print(
                f"  step {step:>4}/{args.steps}  lr={lr:.2e}  loss={loss.item():.4f}  "
                f"mem={mem:.1f}GB  elapsed={time.time()-t0:.0f}s",
                flush=True,
            )

        # eval at fixed checkpoints (TF / argmax / soft — 暴露偏差 + Soft-Exp 收益)
        if step in (500, 1000, 2000, 4000, 6000, 8000):
            if step <= args.steps:
                tf_ppl = eval_tf_ppl(model, val_ids, seq_len=args.seq_len)
                arg_ppl = eval_argmax_ppl(model, val_ids, seq_len=args.seq_len)
                soft_ppl = eval_soft_ppl(model, val_ids, seq_len=args.seq_len)
                gap = (arg_ppl - tf_ppl) / tf_ppl * 100
                soft_adv = (arg_ppl - soft_ppl) / arg_ppl * 100
                results["checkpoints"].append({
                    "step": step,
                    "train_loss": round(loss.item(), 4),
                    "tf_ppl": round(tf_ppl, 4),
                    "argmax_ppl": round(arg_ppl, 4),
                    "soft_ppl": round(soft_ppl, 4),
                    "exposure_gap_pct": round(gap, 2),
                    "soft_advantage_pct": round(soft_adv, 2),
                })
                print(
                    f"  >>> ckpt@{step}: TF={tf_ppl:.3f}  argmax={arg_ppl:.3f}  "
                    f"soft={soft_ppl:.3f}  gap={gap:+.1f}%  soft-adv={soft_adv:+.1f}%",
                    flush=True,
                )

    results["total_time_s"] = round(time.time() - t0, 2)
    if DEVICE == "cuda":
        results["peak_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)

    # verdict
    if nan_hit:
        verdict = "FAIL_NaN"
    elif len(results["checkpoints"]) >= 2:
        first_ppl = results["checkpoints"][0]["tf_ppl"]
        last_ppl = results["checkpoints"][-1]["tf_ppl"]
        last_arg = results["checkpoints"][-1].get("argmax_ppl", float("inf"))
        last_gap = results["checkpoints"][-1].get("exposure_gap_pct", 0)
        if last_ppl < first_ppl * 0.9:
            # TF PPL decreased, check argmax gap (暴露偏差)
            if last_gap > 1000:
                verdict = "FAIL_memorizer_tf_low_argmax_huge_gap"
            elif last_gap > 100:
                verdict = f"WATCH_memorizer_suspect_gap_{last_gap:.0f}pct"
            elif last_arg < 50:
                verdict = "PASS_real_lm_signal"
            else:
                verdict = f"WATCH_argmax_high_{last_arg:.1f}"
        elif last_ppl < first_ppl:
            verdict = "WATCH_slow_decrease"
        else:
            verdict = "FAIL_not_decreasing"
    else:
        verdict = "INSUFFICIENT_DATA"
    results["verdict"] = verdict

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[saved] -> {args.out}")
    print(f"[verdict] {verdict}")


if __name__ == "__main__":
    main()
