"""proto_v23_decoder.py — v23 BAD-DP decoder 训练 (Step 5)

warm-start from v22a decoder weights (24L×1280×20, ~475M params).
2 phase schedule: anneal (1e-4) 4K step + fine (5e-5) 4K step.
T expanded 128 → 512 (4x context length).
val 强制 v22a_val.parquet (PPL < 4.39 验收).
"""
import io
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
)
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def P(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


# === PATHS ===
DATA = Path("crystalllm/data/processed")
TRAIN_PATH = DATA / "extended_v23.parquet"
VAL_PATH = DATA / "v22a_val.parquet"  # 强制 anchor
VOCAB_PATH = DATA / "char_vocab.json"
INIT_FROM = Path("crystalllm/v22_decoder.pt")  # v22a decoder checkpoint
OUT_PATH = Path("crystalllm/v23_decoder.pt")

# === MODEL CONFIG (与 v22a 保持一致) ===
DEPTH = 24
D_MODEL = 1280
N_HEAD = 20
D_Z = 256
MAX_SEQ_LEN = 512  # 128 → 512 (v23 关键扩展)
VOCAB_SIZE = 2261
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === TRAINING SCHEDULE ===
PHASE1_STEPS = 4000  # anneal
PHASE2_STEPS = 4000  # fine
PHASE1_LR = 1.0e-4
PHASE2_LR = 5.0e-5
LOG_EVERY = 100
EVAL_EVERY = 1000
CHECKPOINT_EVERY = 500
W_RECON = 1.0
W_KL = 0.1
KL_ANNEAL_STEPS = 1000
FREE_BITS_NAT = 1.0
PPL_TARGET = 4.39  # v22a baseline: 必须低于此值


# === 模型定义 (与 v22a 对齐) ===
class BlockCausal(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.nh = n_head
        self.head_dim = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        B_, T_, C = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B_, T_, 3, self.nh, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).contiguous().view(B_, T_, C))
        x = x + self.mlp(self.ln2(x))
        return x


class Decoder(nn.Module):
    """v23 decoder: 24L×1280×20, z 256 dim, T=512."""

    def __init__(self, d_z=D_Z, max_seq_len=MAX_SEQ_LEN, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.d_z = d_z
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.z_to_emb = nn.Linear(d_z, D_MODEL)
        self.tok = nn.Embedding(vocab_size, D_MODEL)
        self.pos = nn.Embedding(max_seq_len + 2, D_MODEL)
        self.blocks = nn.ModuleList([BlockCausal(D_MODEL, N_HEAD) for _ in range(DEPTH)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab_size, bias=False)
        self.tok.weight = self.head.weight

    def forward(self, z, x):
        B_, T_ = x.shape
        z_emb = self.z_to_emb(z).unsqueeze(1)
        bos_emb = self.tok(torch.tensor([1], device=x.device)).expand(B_, 1, -1)
        x_emb = self.tok(x)
        inp = torch.cat([z_emb, bos_emb, x_emb], dim=1)
        inp = inp + self.pos(torch.arange(T_ + 2, device=x.device))
        for b in self.blocks:
            inp = b(inp)
        logits = self.head(self.ln_f(inp))
        return logits[:, 1:T_ + 1]


def build_model():
    """Build v23 decoder. Same arch as v22a but T=512."""
    return Decoder()


def load_v22a_state(model: Decoder) -> Decoder:
    """Warm-start from v22a decoder.

    v22a saves {"decoder": state, "config": {...}}.
    v22a pos embedding was T+2=130; v23 needs T+2=514.
    Other layers should be shape-compatible.
    """
    if not INIT_FROM.exists():
        P(f"[warn] {INIT_FROM} not found, training from scratch (cold start)")
        return model
    P(f"[warm-start] loading v22a state from {INIT_FROM}")
    ckpt = torch.load(INIT_FROM, map_location="cpu", weights_only=False)
    v22_state = ckpt.get("decoder", ckpt) if isinstance(ckpt, dict) else ckpt
    new_state = model.state_dict()
    n_loaded = 0
    for k, v in v22_state.items():
        if k in new_state:
            if v.shape == new_state[k].shape:
                new_state[k] = v
                n_loaded += 1
            elif k == "pos.weight":
                # v22a: pos (130, 1280) → v23: pos (514, 1280)
                # copy first 130 rows, zero-init the rest
                old_n = v.shape[0]
                new_state[k][:old_n] = v
                new_state[k][old_n:] = 0
                P(f"  {k}: 扩展 {old_n}→{new_state[k].shape[0]}, 旧行复制, 新行零")
                n_loaded += 1
            else:
                P(f"  跳过 {k}: 形状不匹配 {v.shape} vs {new_state[k].shape}")
        else:
            P(f"  警告: {k} 不在新 state 中")
    model.load_state_dict(new_state)
    P(f"[warm-start] loaded {n_loaded}/{len(new_state)} tensors from v22a")
    return model


# === 数据加载 (parquet → DataLoader) ===
def build_loader(parquet_path: Path, batch_size: int, shuffle: bool, max_seq_len: int = MAX_SEQ_LEN):
    """Build a DataLoader from a parquet with a 'text' column (char-level)."""
    pd = __import__("pandas")
    df = pd.read_parquet(parquet_path)
    if shuffle:
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    stoi = vocab["stoi"]
    pad_id = stoi.get("<pad>", 0)
    texts = df["text"].tolist()
    arr = np.full((len(texts), max_seq_len), pad_id, dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [stoi.get(c, 0) for c in str(t)[:max_seq_len]]
        n = len(ids)
        if n > 0:
            arr[i, :n] = ids

    class TextDS(torch.utils.data.Dataset):
        def __len__(self):
            return len(arr)

        def __getitem__(self, i):
            return arr[i]

    return DataLoader(
        TextDS(), batch_size=batch_size, shuffle=shuffle, num_workers=0
    )


def kl_loss(mu, logvar, free_bits=FREE_BITS_NAT):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = kl.mean(dim=0)
    return torch.clamp(kl_per_dim, min=free_bits).sum(), kl_per_dim.detach()


@torch.no_grad()
def eval_ppl(model: Decoder, val_loader, device, z_cache=None, val_items=None):
    """Eval val PPL. If z_cache is provided (numpy/tensor), use it for z; else sample N(0,1)."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i, batch in enumerate(val_loader):
        x = batch.to(device)
        B_ = x.size(0)
        if z_cache is not None:
            ix = np.random.randint(0, len(z_cache), B_)
            z = torch.as_tensor(z_cache[ix], dtype=torch.float32, device=device)
        else:
            z = torch.randn(B_, D_Z, device=device)
        logits = model(z, x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            x.reshape(-1),
            reduction="sum",
            ignore_index=0,  # <pad>
        )
        total_loss += loss.item()
        total_tokens += (x != 0).sum().item()
    model.train()
    if total_tokens == 0:
        return float("inf")
    return float(np.exp(total_loss / total_tokens))


def train():
    P("=== v23 Decoder Warm-Start (T=512, 8K step 2-phase) ===")
    model = build_model().to(DEVICE)
    model = load_v22a_state(model)
    n_params = sum(p.numel() for p in model.parameters())
    P(f"[init] model params: {n_params/1e6:.1f}M")

    # val loader (固定 anchor v22a)
    P(f"[data] val: {VAL_PATH}")
    val_loader = build_loader(VAL_PATH, BATCH_SIZE, shuffle=False)
    init_ppl = eval_ppl(model, val_loader, DEVICE)
    P(f"[init] val PPL (post warm-start): {init_ppl:.3f}")

    # train loader
    if not TRAIN_PATH.exists():
        P(f"[warn] {TRAIN_PATH} not found, will not actually train")
        return init_ppl
    P(f"[data] train: {TRAIN_PATH}")
    train_loader = build_loader(TRAIN_PATH, BATCH_SIZE, shuffle=True)

    optim = torch.optim.AdamW(model.parameters(), lr=PHASE1_LR, weight_decay=0.1, betas=(0.9, 0.95))
    total_steps = PHASE1_STEPS + PHASE2_STEPS
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps)

    phases = [(PHASE1_STEPS, PHASE1_LR), (PHASE2_STEPS, PHASE2_LR)]
    step = 0
    t0 = time.time()
    log = []
    train_iter = iter(train_loader)

    for phase_idx, (n_steps, lr) in enumerate(phases):
        for g in optim.param_groups:
            g["lr"] = lr
        P(f"\n[phase {phase_idx+1}] {n_steps} steps, lr={lr}")
        phase_start = step
        while step - phase_start < n_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            x = batch.to(DEVICE)
            B_ = x.size(0)
            z = torch.randn(B_, D_Z, device=DEVICE)  # BAD-DP: z 是 encoder 输出; 训练时采样
            logvar = torch.full_like(z, -3.0)
            logits = model(z, x)
            loss_recon = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), x.reshape(-1), ignore_index=0
            )
            loss_kl, _ = kl_loss(z, logvar, FREE_BITS_NAT)
            beta = min(1.0, step / KL_ANNEAL_STEPS)
            loss = W_RECON * loss_recon + W_KL * beta * loss_kl
            if not torch.isfinite(loss):
                P(f"[skip NaN] step={step}")
                continue
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            step += 1
            if step % LOG_EVERY == 0:
                P(
                    f"  step {step}/{total_steps} | recon {loss_recon.item():.3f} "
                    f"| KL {loss_kl.item():.2f} β={beta:.3f} "
                    f"| lr={optim.param_groups[0]['lr']:.2e} "
                    f"| t={time.time()-t0:.0f}s"
                )
            if step % EVAL_EVERY == 0 or step == total_steps:
                ppl = eval_ppl(model, val_loader, DEVICE)
                P(f"  step {step} val_ppl={ppl:.3f}")
                log.append({"step": step, "val_ppl": ppl, "phase": phase_idx + 1})
            if step % CHECKPOINT_EVERY == 0:
                ckpt_path = str(OUT_PATH).replace(".pt", f"_step{step}.pt")
                torch.save(model.state_dict(), ckpt_path)
                P(f"  [ckpt] {ckpt_path}")

    final_ppl = eval_ppl(model, val_loader, DEVICE)
    P(f"\n[final] val_ppl={final_ppl:.3f}")
    P(f"[done] {time.time()-t0:.0f}s total")
    torch.save(
        {
            "decoder": model.state_dict(),
            "config": {
                "V": VOCAB_SIZE,
                "T": MAX_SEQ_LEN,
                "D_Z": D_Z,
                "DEC_LAYER": DEPTH,
                "DEC_HEAD": N_HEAD,
                "DEC_EMBD": D_MODEL,
                "W_KL": W_KL,
                "W_RECON": W_RECON,
                "FREE_BITS_NAT": FREE_BITS_NAT,
                "warm_start_from": "v22a_decoder",
                "arch": "v23-500M-BAD-decoder-T512-warm-start",
            },
        },
        str(OUT_PATH),
    )
    P(f"[saved] {OUT_PATH}")
    log_path = str(OUT_PATH).replace(".pt", "_train_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "log": log,
                "config": {
                    "PHASE1_STEPS": PHASE1_STEPS,
                    "PHASE2_STEPS": PHASE2_STEPS,
                    "PHASE1_LR": PHASE1_LR,
                    "PHASE2_LR": PHASE2_LR,
                    "decoder_params_M": n_params / 1e6,
                    "warm_start_from": "v22a_decoder",
                    "arch": "v23-500M-BAD-decoder-T512-warm-start",
                },
                "final_ppl": final_ppl,
                "ppl_target": PPL_TARGET,
            },
            f,
            indent=2,
        )
    P(f"[log] {log_path}")
    return final_ppl


if __name__ == "__main__":
    final = train()
    if final >= PPL_TARGET:
        P(f"[FAIL] v23 PPL {final:.3f} >= {PPL_TARGET}, NOT IMPROVED")
        sys.exit(1)
    P(f"[PASS] v23 PPL {final:.3f} < {PPL_TARGET} ✓")
