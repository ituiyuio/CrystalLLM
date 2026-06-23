"""
Exp 28: 32M / 16M 重跑 - warmup + lower LR + no weight tying
============================================================
修复 32M 训练失败问题: lr=3e-4 太高导致 loss 锁在 7.12.
策略: 1000-step warmup, peak lr=5e-5, 不 tie weights, 16k 步
"""
import math, time, json, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(42)
DEVICE = "cuda"

TRAIN_PATH = "D:/CrystaLLM/experiments/v49_pre/bpe_train_65M_s42.npy"
VAL_PATH = "D:/CrystaLLM/experiments/v49_pre/bpe_val_s42.npy"
VOCAB_SIZE = 4100


def load_data():
    train_ids = np.load(TRAIN_PATH)
    val_ids = np.load(VAL_PATH)
    return torch.from_numpy(train_ids.astype(np.int64)), torch.from_numpy(val_ids.astype(np.int64))


def get_batch(ids, bs, sl):
    n = len(ids) - sl - 1
    starts = torch.randint(0, n, (bs,))
    x = torch.stack([ids[s:s+sl] for s in starts])
    y = torch.stack([ids[s+1:s+sl+1] for s in starts])
    return x.to(DEVICE), y.to(DEVICE)


class MiniGPT(nn.Module):
    def __init__(self, vocab_size=4100, d_model=512, nhead=8, num_layers=8, max_len=128, tie_weights=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        # pre-LN 更稳
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model*4,
            dropout=0.0, batch_first=True, activation="gelu",
            norm_first=True  # Pre-LN
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
        ids = val_ids[s:s+seq_len+1].to(DEVICE).unsqueeze(0)
        logits = model(ids[:, :-1])
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), ids[:, 1:].view(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += seq_len
    return math.exp(total_loss / total_tokens)


@torch.no_grad()
def eval_autoregressive(model, val_ids, feedback_fn, num_seqs=50, seq_len=64):
    model.eval()
    n = len(val_ids) - seq_len - 1
    starts = torch.randint(0, n, (num_seqs,))
    total_loss, total_tokens = 0.0, 0
    for s in starts:
        ids = val_ids[s:s+seq_len+1].to(DEVICE).unsqueeze(0)
        cur = model.token_embed(ids[:, 0]).unsqueeze(1)
        for t in range(seq_len):
            S = cur.shape[1]
            pos = torch.arange(S, device=cur.device).unsqueeze(0)
            h = cur + model.pos_embed(pos)
            mask = torch.triu(torch.ones(S, S, device=cur.device, dtype=torch.bool), diagonal=1)
            h = model.encoder(h, mask=mask)
            logits = model.lm_head(model.ln(h[:, -1:, :]))
            target = ids[:, t+1]
            total_loss += F.cross_entropy(logits.view(-1, logits.size(-1)), target, reduction="sum").item()
            total_tokens += 1
            fb = feedback_fn(logits[:, -1, :], model)
            if fb.dim() == 2: fb = fb.unsqueeze(1)
            cur = torch.cat([cur[:, 1:], fb], dim=1)
    return math.exp(total_loss / total_tokens)


def lr_at(step, warmup, peak):
    """Linear warmup, then constant."""
    if step < warmup:
        return peak * step / warmup
    return peak


def run_model(name, d_model, nhead, num_layers, train_ids, val_ids,
              steps=16000, peak_lr=5e-5, warmup=1000, ckpts=(1000, 2000, 4000, 8000, 16000)):
    print(f"\n{'='*70}\n[{name}] d_model={d_model} nhead={nhead} layers={num_layers}  peak_lr={peak_lr}\n{'='*70}")
    model = MiniGPT(d_model=d_model, nhead=nhead, num_layers=num_layers, tie_weights=False).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01, betas=(0.9, 0.95))

    results = {"name": name, "params": n_params, "peak_lr": peak_lr, "checkpoints": []}
    t0 = time.time()
    loss_window = []

    for step in range(1, steps + 1):
        lr = lr_at(step, warmup, peak_lr)
        for g in opt.param_groups: g['lr'] = lr
        x, y = get_batch(train_ids, 32, 128)
        model.train()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        loss_window.append(loss.item())
        if step % 200 == 0:
            avg = sum(loss_window[-200:]) / 200
            print(f"  step {step:>5}/{steps}  lr={lr:.2e}  loss={avg:.4f}  elapsed={time.time()-t0:.0f}s", flush=True)

        if step in ckpts:
            ckpt_t = time.time() - t0
            tf_ppl = eval_teacher_forcing(model, val_ids)
            ppl_arg = eval_autoregressive(model, val_ids, feedback_argmax)
            ppl_soft = eval_autoregressive(model, val_ids, feedback_soft)
            soft_adv = (ppl_arg - ppl_soft) / ppl_arg * 100
            results["checkpoints"].append({
                "step": step, "train_loss": sum(loss_window[-50:])/50,
                "tf_ppl": tf_ppl, "argmax_ppl": ppl_arg, "soft_ppl": ppl_soft,
                "soft_advantage_pct": soft_adv,
            })
            print(f"  >>> ckpt@{step}: TF={tf_ppl:.2f}  argmax={ppl_arg:.2f}  soft={ppl_soft:.2f}  soft-adv={soft_adv:+.2f}%  time={ckpt_t:.0f}s", flush=True)

    results["total_time_s"] = time.time() - t0
    return results


def main():
    train_ids, val_ids = load_data()
    print(f"data: train={len(train_ids):,}, val={len(val_ids):,}")
    configs = [
        ("M_16M",   384, 6, 6, 5e-5, 1000),  # 16M model
        ("M_32M",   512, 8, 8, 3e-5, 1500),  # 32M model, even lower lr
    ]
    all_results = []
    for name, d, h, l, lr, w in configs:
        r = run_model(name, d, h, l, train_ids, val_ids,
                      steps=16000, peak_lr=lr, warmup=w)
        all_results.append(r)
        out_path = "D:/CrystaLLM/experiments/v49_pre/exp28_fixed_lr_results.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[saved] -> {out_path}")

    print("\n" + "="*70)
    print("EXP 28 SUMMARY: 16M / 32M with proper LR")
    print("="*70)
    print(f"{'model':<10} {'params':>10} {'step':>6} {'TF':>8} {'argmax':>10} {'soft':>10} {'soft-adv':>10}")
    for r in all_results:
        for ck in r["checkpoints"]:
            print(f"{r['name']:<10} {r['params']:>10,} {ck['step']:>6} {ck['tf_ppl']:>8.2f} {ck['argmax_ppl']:>10.2f} {ck['soft_ppl']:>10.2f} {ck['soft_advantage_pct']:>+9.2f}%")


if __name__ == "__main__":
    main()