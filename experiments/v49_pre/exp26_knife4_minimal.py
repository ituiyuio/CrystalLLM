"""
Exp 26: Knife-4 minimal counter-example (LOCAL DATA VERSION)
=============================================================
测试"刀4: 连续期望反馈解码"在实数 baseline Transformer 上是否有 empirical benefit.

数据: crystalllm BPE, vocab=4100, train=803k tokens, val=26k tokens
模型: d_model=256, nhead=4, n_layers=2, max_len=128
训练: teacher forcing + CE, 2000 steps
评估: 自回归 PPL (3 种反馈策略)
"""

import math, time, json, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 数据加载: 本地 BPE
# ============================================================
def load_data():
    print("[data] loading local BPE tokens ...")
    train_ids = np.load("D:/CrystaLLM/experiments/v49_pre/bpe_train_10000_s42.npy")
    val_ids = np.load("D:/CrystaLLM/experiments/v49_pre/bpe_val_s42.npy")
    print(f"[data] train tokens: {len(train_ids):,}, val tokens: {len(val_ids):,}")
    print(f"[data] vocab range: {min(train_ids.min(), val_ids.min())} - {max(train_ids.max(), val_ids.max())}")
    return torch.from_numpy(train_ids.astype(np.int64)), torch.from_numpy(val_ids.astype(np.int64))


def get_batch(ids, batch_size, seq_len):
    n = len(ids) - seq_len - 1
    starts = torch.randint(0, n, (batch_size,))
    x = torch.stack([ids[s:s+seq_len] for s in starts])
    y = torch.stack([ids[s+1:s+seq_len+1] for s in starts])
    return x.to(DEVICE), y.to(DEVICE)


# ============================================================
# 模型
# ============================================================
class MiniGPT(nn.Module):
    def __init__(self, vocab_size=4100, d_model=256, nhead=4, num_layers=2, max_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512,
                                            dropout=0.0, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight  # weight tying
        self.max_len = max_len
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        B, S = x.shape
        pos = torch.arange(S, device=x.device).unsqueeze(0)
        h = self.token_embed(x) + self.pos_embed(pos)
        mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.encoder(h, mask=mask)
        h = self.ln(h)
        return self.lm_head(h)


# ============================================================
# 反馈策略
# ============================================================
@torch.no_grad()
def feedback_argmax(logits, model):
    next_id = logits.argmax(dim=-1)
    return model.token_embed(next_id), next_id


@torch.no_grad()
def feedback_soft(logits, model):
    probs = F.softmax(logits.float(), dim=-1)
    expected = torch.matmul(probs, model.token_embed.weight)
    return expected, None


@torch.no_grad()
def feedback_hard(logits, model, temperature=1.0):
    probs = F.softmax(logits.float() / temperature, dim=-1)
    gumbel = -torch.log(-torch.log(torch.rand_like(probs) + 1e-9) + 1e-9)
    next_id = (probs + gumbel).argmax(dim=-1)
    return model.token_embed(next_id), next_id


# ============================================================
# 训练
# ============================================================
def train_model(model, train_ids, steps=2000, batch_size=32, seq_len=128, lr=3e-4, log_every=200):
    print(f"[train] starting {steps} steps, bs={batch_size}, seq={seq_len}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    t0 = time.time()
    losses = []
    for step in range(1, steps + 1):
        x, y = get_batch(train_ids, batch_size, seq_len)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % log_every == 0 or step == 1:
            elapsed = time.time() - t0
            avg = sum(losses[-log_every:]) / min(log_every, len(losses))
            print(f"  step {step:>4}/{steps}  loss={avg:.4f}  elapsed={elapsed:.1f}s", flush=True)
    return losses


# ============================================================
# 自回归评估
# ============================================================
def eval_autoregressive(model, val_ids, feedback_fn, num_seqs=80, seq_len=128, name="", temperature=1.0):
    model.eval()
    n = len(val_ids) - seq_len - 1
    starts = torch.randint(0, n, (num_seqs,))
    total_loss = 0.0
    total_tokens = 0
    for s in starts:
        ids = val_ids[s:s+seq_len+1].to(DEVICE).unsqueeze(0)
        # 用第一个 token 的 embedding 作为初始输入
        cur = model.token_embed(ids[:, 0])  # (1, D)
        cur = cur.unsqueeze(1)  # (1, 1, D)
        for t in range(seq_len):
            S = cur.shape[1]
            pos = torch.arange(S, device=cur.device).unsqueeze(0)
            h = cur + model.pos_embed(pos)
            mask = torch.triu(torch.ones(S, S, device=cur.device, dtype=torch.bool), diagonal=1)
            h = model.encoder(h, mask=mask)
            h = model.ln(h)
            logits = model.lm_head(h[:, -1:, :])  # (1, 1, V)
            target = ids[:, t+1]
            loss_t = F.cross_entropy(logits.view(-1, logits.size(-1)), target, reduction="sum")
            total_loss += loss_t.item()
            total_tokens += 1

            feedback_input, _ = feedback_fn(logits[:, -1, :], model)
            if feedback_input.dim() == 2:
                feedback_input = feedback_input.unsqueeze(1)  # (1, 1, D)
            # 滑动窗口
            cur = torch.cat([cur[:, 1:], feedback_input], dim=1)

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    print(f"  [eval/{name:>10}]  seqs={num_seqs}, len={seq_len}, loss={avg_loss:.4f}, PPL={ppl:.2f}", flush=True)
    return ppl


def main():
    train_ids, val_ids = load_data()

    print("\n" + "="*60)
    print("PHASE 1: 训练共享模型 (teacher forcing + CE)")
    print("="*60)
    model = MiniGPT().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params: {n_params:,}")
    losses = train_model(model, train_ids, steps=2000)
    final_loss = sum(losses[-50:]) / 50
    print(f"[train] final avg loss: {final_loss:.4f}")

    print("\n" + "="*60)
    print("PHASE 2: 自回归评估 (相同模型, 3 种反馈策略)")
    print("="*60)

    print("\n[1/3] Argmax (baseline)")
    ppl_argmax = eval_autoregressive(model, val_ids, feedback_argmax, name="argmax")

    print("\n[2/3] Soft-Exp (continuous expected_embed)")
    ppl_soft = eval_autoregressive(model, val_ids, feedback_soft, name="soft")

    print("\n[3/3] Hard-Exp (Gumbel-ST argmax)")
    ppl_hard = eval_autoregressive(model, val_ids,
        lambda l, m: feedback_hard(l, m, 1.0), name="hard")

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"  Baseline  (argmax)         PPL = {ppl_argmax:>8.2f}")
    print(f"  Soft-Exp  (continuous)     PPL = {ppl_soft:>8.2f}    Δ = {(ppl_soft-ppl_argmax)/ppl_argmax*100:+.2f}%")
    print(f"  Hard-Exp  (Gumbel-ST)      PPL = {ppl_hard:>8.2f}    Δ = {(ppl_hard-ppl_argmax)/ppl_argmax*100:+.2f}%")
    print(f"  Final train loss:          {final_loss:.4f}")

    verdict = "VERIFIED" if ppl_soft < ppl_argmax * 0.99 else "FALSIFIED"
    print(f"\n>>> Knife-4 验证: {verdict}")
    if verdict == "VERIFIED":
        print("    Soft-Exp 比 baseline 低 ≥1% → 连续反馈有独立价值")
    else:
        print("    Soft-Exp 没有显著优势 → 刀4 证伪, CMT 线归档")

    out = {
        "train_final_loss": final_loss,
        "ppl_argmax": ppl_argmax,
        "ppl_soft": ppl_soft,
        "ppl_hard": ppl_hard,
        "delta_soft_pct": (ppl_soft - ppl_argmax) / ppl_argmax * 100,
        "delta_hard_pct": (ppl_hard - ppl_argmax) / ppl_argmax * 100,
        "verdict": verdict,
        "config": {
            "d_model": 256, "nhead": 4, "n_layers": 2,
            "max_len": 128, "train_steps": 2000,
            "batch_size": 32, "lr": 3e-4,
            "num_val_seqs": 80, "vocab": 4100,
        }
    }
    out_path = "D:/CrystaLLM/experiments/v49_pre/exp26_knife4_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[save] -> {out_path}")


if __name__ == "__main__":
    main()