"""
Exp 32: V50 byte-level sanity (基于 exp30_soft_exp_train.py)
=============================================================
目标: 验证 V49 架构在 byte-level 输入下能跑通完整训练-评估闭环
- 训练数据: v28_train.parquet → UTF-8 bytes → 前 2M bytes
- 验证数据: v28_val.parquet   → UTF-8 bytes → 前 100K bytes
- 模型:     MiniGPT(vocab_size=256), 与 exp30 同架构 (Pre-LN, learned pos_embed)
- 训练:     Soft-Exp 双前向 (α 调度) + 暴露偏差监控,完全继承 exp30 逻辑

vs exp30 的改动清单 (4 处,精确锁定):
  1. 加 `import pandas as pd` (line ~25)
  2. VOCAB_SIZE 4100 → 256,  数据路径换成 parquet (line ~36)
  3. load_data() 从 np.load 改为 pd.read_parquet + UTF-8 encode
  4. run_exp30 调用加 vocab_size=VOCAB_SIZE, seq_len=64, ckpts (100,200,300,400,500)

判决 (per spec):
  ✅ 5k 步内 loss 平滑下降 → byte-level 路径可行
  ❌ loss 卡住 / NaN / 暴露偏差 > 100x → 回退 V49 char-level
"""
import math
import time
import json
import sys
import argparse
from pathlib import Path

import pandas as pd          # 改点 1: byte-level 需要读 parquet
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

torch.manual_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# 改点 2: 数据路径 + vocab size
# ============================================================================
TRAIN_PARQUET = PROJECT_ROOT / "crystalllm" / "data" / "processed" / "v28_train.parquet"
VAL_PARQUET   = PROJECT_ROOT / "crystalllm" / "data" / "processed" / "v28_val.parquet"
VOCAB_SIZE = 256  # byte-level (UTF-8 bytes 0-255)
RESULTS_DIR = PROJECT_ROOT / "experiments" / "v49_pre" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 改点 3: load_data() — parquet text → UTF-8 bytes → tensor[int64]
# ---------------------------------------------------------------------------
def load_data():
    """从 v28 parquet 读 text 列 → UTF-8 encode → byte-level tensor (int64 0-255)."""
    print(f"[load] reading {TRAIN_PARQUET.name} ...")
    train_text = "\n".join(pd.read_parquet(TRAIN_PARQUET)["text"].astype(str).tolist())
    print(f"[load] reading {VAL_PARQUET.name} ...")
    val_text = "\n".join(pd.read_parquet(VAL_PARQUET)["text"].astype(str).tolist())

    train_bytes = train_text.encode("utf-8")[:2_000_000]   # sanity 2M bytes
    val_bytes   = val_text.encode("utf-8")[:100_000]      # sanity 100K bytes
    print(f"[load] train bytes: {len(train_bytes):,}  val bytes: {len(val_bytes):,}")

    return (torch.tensor(list(train_bytes), dtype=torch.int64),
            torch.tensor(list(val_bytes),   dtype=torch.int64))


# ---------------------------------------------------------------------------
# get_batch: 与 exp30 完全一致 (通用, 不依赖 vocab_size)
# ---------------------------------------------------------------------------
def get_batch(ids, bs, sl):
    """随机采样 batch: input/output 错位一格."""
    n = len(ids) - sl - 1
    starts = torch.randint(0, n, (bs,))
    x = torch.stack([ids[s:s + sl] for s in starts])
    y = torch.stack([ids[s + 1:s + sl + 1] for s in starts])
    return x.to(DEVICE), y.to(DEVICE)


# ---------------------------------------------------------------------------
# Model: 与 exp30 完全一致的 MiniGPT (Pre-LN, learned pos_embed, no tied weights)
# ---------------------------------------------------------------------------
class MiniGPT(nn.Module):
    """Pre-LN Transformer, 不 tie weights (与 exp28/29 baseline 一致)."""
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=512, nhead=8,
                 num_layers=8, max_len=128, tie_weights=False, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.token_embed.weight
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        B, S = x.shape
        pos = torch.arange(S, device=x.device).unsqueeze(0)
        h = self.token_embed(x) + self.pos_embed(pos)
        mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.encoder(h, mask=mask)
        return self.lm_head(self.ln(h))


# ---------------------------------------------------------------------------
# α schedule (与 exp30 一致)
# ---------------------------------------------------------------------------
def alpha_schedule(step: int, warmup_steps: int, alpha_max: float) -> float:
    """α 线性 warmup: 0 → alpha_max."""
    if step < warmup_steps:
        return alpha_max * step / warmup_steps
    return alpha_max


def exp30_train_step(model, x, y, opt, alpha: float) -> float:
    """双前向 Soft-Exp 训练 step (与 exp30 完全一致, 通用)."""
    B, S = x.shape

    # Pass 1: TF, no_grad 算 soft embeds
    with torch.no_grad():
        logits_t1 = model(x)
        probs_t1 = F.softmax(logits_t1, dim=-1)
        soft_embeds = torch.matmul(probs_t1, model.token_embed.weight)

    # 构造 mixed input embedding
    gt_embeds = model.token_embed(x)
    if S > 1:
        shifted_soft = torch.cat(
            [torch.zeros_like(soft_embeds[:, :1, :]), soft_embeds[:, :-1, :]],
            dim=1,
        )
        mixed_embeds = (1.0 - alpha) * gt_embeds + alpha * shifted_soft
    else:
        mixed_embeds = gt_embeds

    # Pass 2: forward from mixed embeddings
    pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)
    h = mixed_embeds + model.pos_embed(pos)
    mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
    h = model.encoder(h, mask=mask)
    logits_t2 = model.lm_head(model.ln(h))

    # Hard target CE
    loss = F.cross_entropy(logits_t2.view(-1, logits_t2.size(-1)), y.view(-1))
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()


# ---------------------------------------------------------------------------
# Eval functions (与 exp30 完全一致, 通用)
# ---------------------------------------------------------------------------
@torch.no_grad()
def feedback_argmax(logits, model):
    return model.token_embed(logits.argmax(dim=-1))


@torch.no_grad()
def feedback_soft(logits, model):
    probs = F.softmax(logits.float(), dim=-1)
    return torch.matmul(probs, model.token_embed.weight)


@torch.no_grad()
def eval_teacher_forcing(model, val_ids, num_seqs=80, seq_len=128):
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
def eval_autoregressive(model, val_ids, feedback_fn, num_seqs=50, seq_len=64):
    model.eval()
    n = len(val_ids) - seq_len - 1
    starts = torch.randint(0, n, (num_seqs,))
    total_loss, total_tokens = 0.0, 0
    for s in starts:
        ids = val_ids[s:s + seq_len + 1].to(DEVICE).unsqueeze(0)
        cur = model.token_embed(ids[:, 0]).unsqueeze(1)
        for t in range(seq_len):
            S = cur.shape[1]
            pos = torch.arange(S, device=cur.device).unsqueeze(0)
            h = cur + model.pos_embed(pos)
            mask = torch.triu(torch.ones(S, S, device=cur.device, dtype=torch.bool), diagonal=1)
            h = model.encoder(h, mask=mask)
            logits = model.lm_head(model.ln(h[:, -1:, :]))
            target = ids[:, t + 1]
            total_loss += F.cross_entropy(
                logits.view(-1, logits.size(-1)), target, reduction="sum"
            ).item()
            total_tokens += 1
            fb = feedback_fn(logits[:, -1, :], model)
            if fb.dim() == 2:
                fb = fb.unsqueeze(1)
            cur = torch.cat([cur[:, 1:], fb], dim=1)
    model.train()
    return math.exp(total_loss / total_tokens)


# ---------------------------------------------------------------------------
# LR schedule (与 exp30 一致)
# ---------------------------------------------------------------------------
def lr_at(step: int, warmup: int, peak: float) -> float:
    if step < warmup:
        return peak * step / warmup
    return peak


# ---------------------------------------------------------------------------
# run_exp30: 与 exp30 一致, 增加 vocab_size 参数 (改点 4)
# ---------------------------------------------------------------------------
def run_exp30(name, d_model, nhead, num_layers, train_ids, val_ids,
              steps=8000, peak_lr=5e-5, warmup=1000, alpha_max=0.5,
              batch_size=32, seq_len=128,
              ckpts=(1000, 2000, 4000, 8000),
              do_eval=True, vocab_size=VOCAB_SIZE):
    print(f"\n{'='*70}")
    print(f"[Exp32 {name}] d_model={d_model} nhead={nhead} layers={num_layers}")
    print(f"  steps={steps}  lr={peak_lr}  warmup={warmup}  α_max={alpha_max}")
    print(f"  batch={batch_size}  seq_len={seq_len}  vocab={vocab_size}  do_eval={do_eval}")
    print(f"{'='*70}")
    torch.manual_seed(42)
    model = MiniGPT(
        vocab_size=vocab_size,      # 改点 4a
        d_model=d_model, nhead=nhead, num_layers=num_layers,
        max_len=seq_len, tie_weights=False,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,} ({n_params/1e6:.1f}M)")

    opt = torch.optim.AdamW(
        model.parameters(), lr=peak_lr,
        weight_decay=0.01, betas=(0.9, 0.95),
    )

    results = {
        "name": name, "params": n_params, "peak_lr": peak_lr,
        "alpha_max": alpha_max, "batch_size": batch_size, "seq_len": seq_len,
        "vocab_size": vocab_size, "checkpoints": [],
    }
    t0 = time.time()
    loss_window = []

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for step in range(1, steps + 1):
        lr = lr_at(step, warmup, peak_lr)
        alpha = alpha_schedule(step, warmup, alpha_max)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_ids, batch_size, seq_len)
        model.train()
        loss = exp30_train_step(model, x, y, opt, alpha)
        loss_window.append(loss)

        if step % 100 == 0 or step == 1:
            recent = sum(loss_window[-min(100, len(loss_window)):]) / min(100, len(loss_window))
            mem = (torch.cuda.max_memory_allocated() / 1e9) if DEVICE == "cuda" else 0
            print(
                f"  step {step:>5}/{steps}  lr={lr:.2e}  α={alpha:.3f}  "
                f"loss={recent:.4f}  mem={mem:.1f}GB  "
                f"elapsed={time.time()-t0:.0f}s",
                flush=True,
            )

        if do_eval and step in ckpts:
            ckpt_t = time.time() - t0
            # ponytail bug fix: 必须传 seq_len,否则 eval 默认 128 会让 pos_embed(64) 索引越界
            tf_ppl = eval_teacher_forcing(model, val_ids, seq_len=seq_len)
            ppl_arg = eval_autoregressive(model, val_ids, feedback_argmax, seq_len=seq_len)
            ppl_soft = eval_autoregressive(model, val_ids, feedback_soft, seq_len=seq_len)
            soft_adv = (ppl_arg - ppl_soft) / ppl_arg * 100
            delta_infer = ppl_soft - tf_ppl
            results["checkpoints"].append({
                "step": step,
                "alpha": alpha,
                "train_loss": sum(loss_window[-50:]) / min(50, len(loss_window)),
                "tf_ppl": tf_ppl,
                "argmax_ppl": ppl_arg,
                "soft_ppl": ppl_soft,
                "soft_advantage_pct": soft_adv,
                "delta_train_infer": delta_infer,
            })
            print(
                f"  >>> ckpt@{step}: TF={tf_ppl:.2f}  argmax={ppl_arg:.2f}  "
                f"soft={ppl_soft:.2f}  soft-adv={soft_adv:+.2f}%  "
                f"Δ(infer-train)={delta_infer:.2f}  time={ckpt_t:.0f}s",
                flush=True,
            )

    results["total_time_s"] = time.time() - t0
    if DEVICE == "cuda":
        results["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return results, model


# ---------------------------------------------------------------------------
# main: 改点 5 — 默认 500 步 + ckpts (100,200,300,400,500) + seq_len=64
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Exp32: V50 byte-level sanity (基于 exp30)")
    parser.add_argument("--smoke", action="store_true", help="100 步 sanity 不 eval")
    parser.add_argument("--steps", type=int, default=500,  # 改点 5a
                       help="sanity 总步数 (默认 500)")
    parser.add_argument("--alpha_max", type=float, default=0.5)
    parser.add_argument("--ckpts", type=str, default=None,
                       help="eval 步数,逗号分隔 (默认: 根据 steps 自动)")
    parser.add_argument("--out", type=str,
                       default=str(PROJECT_ROOT / "experiments" / "v49_pre" / "exp32_results.json"))
    parser.add_argument("--ckpt", type=str, default=None,
                       help="可选: 保存 checkpoint 路径")
    args = parser.parse_args()

    train_ids, val_ids = load_data()
    print(f"[data] train={len(train_ids):,} bytes  val={len(val_ids):,} bytes")

    if args.smoke:
        # Smoke test: 50M / 100 步 / 不 eval
        print("\n>>> SMOKE MODE: 50M, 100 steps, no eval <<<")
        results, model = run_exp30(
            "smoke_50M", d_model=640, nhead=8, num_layers=10,
            train_ids=train_ids, val_ids=val_ids,
            steps=100, peak_lr=5e-5, warmup=50,
            alpha_max=args.alpha_max, batch_size=32, seq_len=64,  # 改点 5b
            ckpts=(), do_eval=False, vocab_size=VOCAB_SIZE,
        )
        print(f"\n[smoke summary] 100 steps 完成, 时间 {results['total_time_s']:.0f}s, "
              f"peak mem {results.get('peak_memory_gb', 0):.1f}GB")
        print(f"[smoke verdict] {'PASS - 无 NaN, 训练稳定' if results['total_time_s'] > 0 else 'FAIL'}")
    else:
        # Default ckpts based on step count
        if args.ckpts is not None:
            ckpts = tuple(int(x) for x in args.ckpts.split(","))
        elif args.steps <= 500:
            ckpts = (100, 200, 300, 400, 500)
        elif args.steps <= 2000:
            ckpts = (200, 500, 1000, 1500, 2000)
        elif args.steps <= 8000:
            ckpts = (500, 2000, 4000, 6000, 8000)
        else:
            ckpts = (1000, args.steps // 2, args.steps)

        print(f"\n>>> V50 BYTE-LEVEL SANITY: 50M, {args.steps} steps, "
              f"eval @ {ckpts}, α_max={args.alpha_max} <<<")
        results, model = run_exp30(
            "v50_byte_sanity", d_model=640, nhead=8, num_layers=10,
            train_ids=train_ids, val_ids=val_ids,
            steps=args.steps, peak_lr=5e-5, warmup=100,
            alpha_max=args.alpha_max, batch_size=32, seq_len=64,
            ckpts=ckpts,
            do_eval=True, vocab_size=VOCAB_SIZE,
        )
        ckpt_path = args.ckpt or str(RESULTS_DIR / "exp32_v50_byte_sanity.final.pt")
        torch.save({
            "step": args.steps,
            "model_state": model.state_dict(),
            "alpha_max": args.alpha_max,
            "config": "v50_byte_sanity_500",
        }, ckpt_path)
        print(f"[saved] -> {ckpt_path}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[saved] -> {args.out}")


if __name__ == "__main__":
    main()
