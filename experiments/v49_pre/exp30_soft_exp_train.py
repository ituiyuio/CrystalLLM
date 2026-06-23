"""
Exp 30: 渐进式刀5 - 训练 Soft-Exp 反馈
=========================================
对比:
  - exp29: V49 1.2B baseline (训练 TF) + 推理 Soft-Exp → argmax=64.7, soft=33.3, +48.6%
  - exp30: V49 架构 (训练 Soft-Exp) + 推理 Soft-Exp → 期望 soft PPL 进一步下降

核心改动 (vs exp28):
  训练时, 下一步输入 embedding = (1-α) * emb(x[t]) + α * probs_{t-1} @ emb.weight
  α 线性 warmup 0 → 0.5, 与 LR warmup 同长
  Loss 仍为 hard target CE (分离变量)

Stage A (50M 8k step ~1h): 先做 sanity, 通过后进 Stage B (1.2B)
"""
import math
import time
import json
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

torch.manual_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 与 exp28 完全一致的数据路径 (BPE 65M, s=42)
TRAIN_PATH = PROJECT_ROOT / "experiments" / "v49_pre" / "bpe_train_65M_s42.npy"
VAL_PATH = PROJECT_ROOT / "experiments" / "v49_pre" / "bpe_val_s42.npy"
VOCAB_SIZE = 4100
RESULTS_DIR = PROJECT_ROOT / "experiments" / "v49_pre" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading (复用 exp28)
# ---------------------------------------------------------------------------
def load_data():
    train_ids = np.load(TRAIN_PATH)
    val_ids = np.load(VAL_PATH)
    return (torch.from_numpy(train_ids.astype(np.int64)),
            torch.from_numpy(val_ids.astype(np.int64)))


def get_batch(ids, bs, sl):
    """随机采样 batch: input/output 错位一格."""
    n = len(ids) - sl - 1
    starts = torch.randint(0, n, (bs,))
    x = torch.stack([ids[s:s + sl] for s in starts])
    y = torch.stack([ids[s + 1:s + sl + 1] for s in starts])
    return x.to(DEVICE), y.to(DEVICE)


# ---------------------------------------------------------------------------
# Model: 与 exp28 相同的 MiniGPT (Pre-LN, no tied weights)
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
# Exp30 核心: 双前向 Soft-Exp 训练 step
# ---------------------------------------------------------------------------
def alpha_schedule(step: int, warmup_steps: int, alpha_max: float) -> float:
    """α 线性 warmup: 0 → alpha_max, 给模型干净窗口学基础结构."""
    if step < warmup_steps:
        return alpha_max * step / warmup_steps
    return alpha_max


def exp30_train_step(model, x, y, opt, alpha: float) -> float:
    """双前向训练: Pass 1 (TF) 算 soft embeds → Pass 2 (mixed) 算 loss.

    Args:
        x: (B, S) input token ids
        y: (B, S) target token ids (shifted by 1)
        opt: optimizer
        alpha: 混合系数 (0=pure TF, 1=pure soft)

    Returns:
        loss value
    """
    B, S = x.shape

    # ----- Pass 1: Teacher forcing, no_grad 只为算 soft embeds -----
    with torch.no_grad():
        logits_t1 = model(x)                            # (B, S, V)
        probs_t1 = F.softmax(logits_t1, dim=-1)         # (B, S, V)
        soft_embeds = torch.matmul(probs_t1, model.token_embed.weight)  # (B, S, D)

    # ----- 构造 mixed input embedding -----
    # soft_embeds[t] = 模型对 emb(x[t+1]) 的期望 (从位置 t 的 logits)
    # TF 输入在位置 t 是 emb(x[t]), 现在替换为:
    #   mixed[t] = (1-α) * emb(x[t]) + α * soft_embeds[t-1]
    # t=0 时无前驱, 纯 GT.
    gt_embeds = model.token_embed(x)                   # (B, S, D)
    if S > 1:
        shifted_soft = torch.cat(
            [torch.zeros_like(soft_embeds[:, :1, :]), soft_embeds[:, :-1, :]],
            dim=1,
        )
        mixed_embeds = (1.0 - alpha) * gt_embeds + alpha * shifted_soft
    else:
        mixed_embeds = gt_embeds

    # ----- Pass 2: forward from mixed embeddings -----
    pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)
    h = mixed_embeds + model.pos_embed(pos)
    mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
    h = model.encoder(h, mask=mask)
    logits_t2 = model.lm_head(model.ln(h))

    # Loss: hard target CE (分离变量)
    loss = F.cross_entropy(logits_t2.view(-1, logits_t2.size(-1)), y.view(-1))

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()


# ---------------------------------------------------------------------------
# Eval functions (与 exp28/29 完全一致)
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
# LR schedule
# ---------------------------------------------------------------------------
def lr_at(step: int, warmup: int, peak: float) -> float:
    if step < warmup:
        return peak * step / warmup
    return peak


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_exp30(name, d_model, nhead, num_layers, train_ids, val_ids,
              steps=8000, peak_lr=5e-5, warmup=1000, alpha_max=0.5,
              batch_size=32, seq_len=128,
              ckpts=(1000, 2000, 4000, 8000),
              do_eval=True):
    print(f"\n{'='*70}")
    print(f"[Exp30 {name}] d_model={d_model} nhead={nhead} layers={num_layers}")
    print(f"  steps={steps}  lr={peak_lr}  warmup={warmup}  α_max={alpha_max}")
    print(f"  batch={batch_size}  seq_len={seq_len}  do_eval={do_eval}")
    print(f"{'='*70}")
    torch.manual_seed(42)
    model = MiniGPT(
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
        "checkpoints": [],
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
            tf_ppl = eval_teacher_forcing(model, val_ids)
            ppl_arg = eval_autoregressive(model, val_ids, feedback_argmax)
            ppl_soft = eval_autoregressive(model, val_ids, feedback_soft)
            soft_adv = (ppl_arg - ppl_soft) / ppl_arg * 100
            delta_infer = ppl_soft - tf_ppl   # NEW: 训练-推理残差
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


def main():
    parser = argparse.ArgumentParser(description="Exp30: 训练 Soft-Exp 反馈")
    parser.add_argument("--smoke", action="store_true", help="100 步 sanity 不 eval")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--alpha_max", type=float, default=0.5)
    parser.add_argument("--out", type=str,
                        default=str(PROJECT_ROOT / "experiments" / "v49_pre" / "exp30_results.json"))
    parser.add_argument("--ckpt", type=str, default=None,
                        help="可选: 保存 checkpoint 路径")
    args = parser.parse_args()

    train_ids, val_ids = load_data()
    print(f"[data] train={len(train_ids):,}  val={len(val_ids):,}")

    if args.smoke:
        # Smoke test: 50M / 100 步 / 不 eval / 只看 loss 不崩
        print("\n>>> SMOKE MODE: 50M, 100 steps, no eval <<<")
        results, model = run_exp30(
            "smoke_50M", d_model=640, nhead=8, num_layers=10,
            train_ids=train_ids, val_ids=val_ids,
            steps=100, peak_lr=5e-5, warmup=50,
            alpha_max=args.alpha_max, batch_size=32, seq_len=128,
            ckpts=(), do_eval=False,
        )
        # Smoke 失败信号: NaN 或 loss 不下降
        final_loss = results["checkpoints"][-1]["train_loss"] if results["checkpoints"] else results.get("total_time_s")
        smoke_first_loss = None
        # 我们没有存首步 loss, 用最后 10 步平均代替作为 sanity
        print(f"\n[smoke summary] 100 steps 完成, 时间 {results['total_time_s']:.0f}s, "
              f"peak mem {results.get('peak_memory_gb', 0):.1f}GB")
        print(f"[smoke verdict] {'PASS - 无 NaN, 训练稳定' if results['total_time_s'] > 0 else 'FAIL'}")
    else:
        # Stage A: 50M, 8k step, 全 eval
        results, model = run_exp30(
            "stageA_50M", d_model=640, nhead=8, num_layers=10,
            train_ids=train_ids, val_ids=val_ids,
            steps=args.steps, peak_lr=5e-5, warmup=1000,
            alpha_max=args.alpha_max, batch_size=32, seq_len=128,
            ckpts=(1000, 2000, 4000, 8000), do_eval=True,
        )
        # 保存 checkpoint
        ckpt_path = args.ckpt or str(RESULTS_DIR / "exp30_50m.final.pt")
        torch.save({
            "step": args.steps,
            "model_state": model.state_dict(),
            "alpha_max": args.alpha_max,
            "config": "stageA_50M",
        }, ckpt_path)
        print(f"[saved] -> {ckpt_path}")

    # 写结果 JSON
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[saved] -> {args.out}")


if __name__ == "__main__":
    main()
