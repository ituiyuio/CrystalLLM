"""
Exp 27: Knife-4 边界扫描 - Soft-Exp 优势随模型规模/训练步数的变化
================================================================
模型: 2M / 8M / 32M
训练: 16k step, checkpoint @ 1k/2k/4k/8k/16k
评估: teacher-forcing PPL, argmax PPL, soft-exp PPL
数据: 65M BPE tokens
"""
import math, time, json, sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[init] device={DEVICE}")

# 数据
TRAIN_PATH = "D:/CrystaLLM/experiments/v49_pre/bpe_train_65M_s42.npy"
VAL_PATH = "D:/CrystaLLM/experiments/v49_pre/bpe_val_s42.npy"
VOCAB_SIZE = 4100


def load_data():
    print(f"[data] loading {TRAIN_PATH}")
    train_ids = np.load(TRAIN_PATH)
    val_ids = np.load(VAL_PATH)
    print(f"[data] train: {len(train_ids):,}, val: {len(val_ids):,}")
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
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_len = max_len
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4,
                                            dropout=0.0, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
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


# ============================================================
# 反馈策略
# ============================================================
@torch.no_grad()
def feedback_argmax(logits, model):
    return model.token_embed(logits.argmax(dim=-1))


@torch.no_grad()
def feedback_soft(logits, model):
    probs = F.softmax(logits.float(), dim=-1)
    return torch.matmul(probs, model.token_embed.weight)


# ============================================================
# 评估
# ============================================================
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
        cur = model.token_embed(ids[:, 0]).unsqueeze(1)  # (1,1,D)
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


# ============================================================
# 训练 + checkpoint 评估
# ============================================================
def run_model(name, d_model, nhead, num_layers, train_ids, val_ids, steps=16000, ckpts=(1000, 2000, 4000, 8000, 16000)):
    print(f"\n{'='*60}\n[{name}] d_model={d_model} nhead={nhead} layers={num_layers}\n{'='*60}")
    model = MiniGPT(d_model=d_model, nhead=nhead, num_layers=num_layers).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    results = {"name": name, "params": n_params, "checkpoints": []}
    t0 = time.time()
    loss_window = []

    for step in range(1, steps + 1):
        x, y = get_batch(train_ids, 32, 128)
        model.train()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        loss_window.append(loss.item())
        if step % 200 == 0:
            print(f"  step {step:>5}/{steps}  loss={sum(loss_window[-200:])/200:.4f}  elapsed={time.time()-t0:.0f}s", flush=True)

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
                "exposure_bias_argmax_x": ppl_arg / tf_ppl,
                "exposure_bias_soft_x": ppl_soft / tf_ppl,
            })
            print(f"  >>> ckpt@{step}: TF={tf_ppl:.2f}  argmax={ppl_arg:.2f}  soft={ppl_soft:.2f}  soft-adv={soft_adv:+.2f}%  time={ckpt_t:.0f}s", flush=True)

    results["total_time_s"] = time.time() - t0
    return results


def main():
    train_ids, val_ids = load_data()
    configs = [
        ("XS_2M",  256, 4, 2),
        ("S_8M",   384, 6, 4),
        ("M_32M",  512, 8, 8),
    ]
    all_results = []
    for name, d, h, l in configs:
        r = run_model(name, d, h, l, train_ids, val_ids, steps=16000)
        all_results.append(r)
        # 增量保存
        out_path = "D:/CrystaLLM/experiments/v49_pre/exp27_boundary_results.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[saved] -> {out_path}")

    # 最终汇总
    print("\n" + "="*60)
    print("BOUNDARY SCAN SUMMARY")
    print("="*60)
    print(f"{'model':<10} {'params':>10} {'step':>6} {'TF':>8} {'argmax':>10} {'soft':>10} {'soft-adv':>10}")
    for r in all_results:
        for ck in r["checkpoints"]:
            print(f"{r['name']:<10} {r['params']:>10,} {ck['step']:>6} {ck['tf_ppl']:>8.2f} {ck['argmax_ppl']:>10.2f} {ck['soft_ppl']:>10.2f} {ck['soft_advantage_pct']:>+9.2f}%")
        print()


if __name__ == "__main__":
    main()