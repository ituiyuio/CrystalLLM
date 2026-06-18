"""
train_v29_token_diff.py — v29 TokenDiffusionDrafter 训练

架构:
- 输入: z (256) + t_emb (128) + pos_emb (100, 512) + noise (100, 512)
- 模型: 30M, 6 ResBlock × 1024
- 输出: (100, 512) 去噪后的 token embedding

训练: CFM (Conditional Flow Matching)
- z_0 = target_emb (真实 token embedding)
- z_1 = noise
- z_t = (1-t)*z_1 + t*z_0
- v_target = z_0 - z_1
- loss = MSE(v_pred, v_target)

数据: cached_v29_outputs.npz (2000, 256) z + (2000, 100) tokens
"""
import json, sys, io, os, random, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); random.seed(42); np.random.seed(42)


def P(*a, **kw):
    print(*a, **kw); sys.stdout.flush()


P("=== v29 TokenDiffusionDrafter 训练 ===")
DATA = Path("data/processed")

# v25 vocab (与 v29 数据一致)
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]
P(f"Vocab {V}")

# ===== 加载数据 =====
P("Loading cached_v29_outputs.npz ...")
data = np.load("cached_v29_outputs.npz")
Z_ALL = torch.tensor(data["z"], dtype=torch.float32)       # (2000, 256)
TOKENS_ALL = torch.tensor(data["tokens"], dtype=torch.long)  # (2000, 100)
P(f"  z: {Z_ALL.shape}, range [{Z_ALL.min():.2f}, {Z_ALL.max():.2f}]")
P(f"  tokens: {TOKENS_ALL.shape}")

# ===== 配置 =====
D_Z = 256
N = 100          # token 序列长度
D_EMB = 512      # token embedding 维度 (与 v25 head 一致)
D_HID = 1024
D_T = 128        # time embedding 维度
N_LAYER = 6
LR = 2e-4
STEPS = 4000
B = 16
EVAL_EVERY = 500
WARMUP_STEPS = 400
DEVICE = "cuda"


class SinusoidalTimeEmbed(nn.Module):
    def __init__(s, dim):
        super().__init__()
        s.dim = dim
    def forward(s, t):
        # t: (B,) in [0, 1]
        half = s.dim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half))
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1) * (s.dim ** 0.5)


class ResBlock(nn.Module):
    def __init__(s, D_HID):
        super().__init__()
        s.ln1 = nn.LayerNorm(D_HID); s.fc1 = nn.Linear(D_HID, D_HID)
        s.ln2 = nn.LayerNorm(D_HID); s.fc2 = nn.Linear(D_HID, D_HID)
        s.film = nn.Linear(D_HID, 2 * D_HID)
    def forward(s, h, t_emb):
        gamma, beta = s.film(t_emb).unsqueeze(1).chunk(2, dim=-1)
        h_res = h
        h = s.fc1(F.gelu(s.ln1(h))) * (1 + gamma) + beta
        h = s.fc2(F.gelu(s.ln2(h)))
        return h_res + h


class TokenDiffusionDrafter(nn.Module):
    """
    输入: z (256), t (B,), noise (B, N, D_EMB)
    输出: v_pred (B, N, D_EMB) - CFM 速度预测

    每位置独立处理, 共享条件 z 和位置编码.
    """
    def __init__(s):
        super().__init__()
        s.pos_emb = nn.Embedding(N, D_EMB)

        # z (256) + t_emb (D_T) → (D_HID) 每位置广播
        s.z_proj = nn.Linear(D_Z, D_HID)
        s.t_proj = nn.Linear(D_T, D_HID)

        # 输入: z_cond (B, D_HID) + pos_emb (N, D_EMB) + noise (B, N, D_EMB)
        s.in_proj = nn.Linear(D_HID + 2 * D_EMB, D_HID)

        s.blocks = nn.ModuleList([ResBlock(D_HID) for _ in range(N_LAYER)])
        s.ln = nn.LayerNorm(D_HID)
        s.out = nn.Linear(D_HID, D_EMB)

        # head (推理用): D_EMB → V
        s.head = nn.Linear(D_EMB, V)

    def forward(s, z, t, noise):
        """
        z: (B, 256)
        t: (B,)
        noise: (B, N, D_EMB)
        返回: (B, N, D_EMB)
        """
        B_, N_, D_ = noise.shape
        # z 条件
        z_cond = s.z_proj(z)  # (B, D_HID)
        t_emb = s.t_proj(s.t_emb(t))  # (B, D_HID)
        cond = z_cond + t_emb  # (B, D_HID)
        cond = cond.unsqueeze(1).expand(-1, N_, -1)  # (B, N, D_HID)

        # 位置编码
        pos = s.pos_emb(torch.arange(N_, device=noise.device))  # (N, D_EMB)
        pos = pos.unsqueeze(0).expand(B_, -1, -1)  # (B, N, D_EMB)

        # 拼接
        x = torch.cat([cond, pos, noise], dim=-1)  # (B, N, D_HID + 2*D_EMB)
        x = s.in_proj(x)  # (B, N, D_HID)

        for blk in s.blocks:
            x = blk(x, t)  # (B, N, D_HID) - 注意 t broadcast

        # ResBlock 内部的 t_emb 需要单独计算
        # 修正: 改为 x = blk(x, t_emb_global)
        x = s.ln(x)
        v_pred = s.out(x)  # (B, N, D_EMB)
        return v_pred

    def t_emb(s, t):
        # 在外面计算
        half = D_T // 2
        freqs = torch.exp(-torch.arange(half, device=t.device, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half))
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1) * (D_T ** 0.5)


# 修正: 让 ResBlock 接收 (B, D_HID) 的 t_emb
class ResBlockV2(nn.Module):
    def __init__(s, D_HID):
        super().__init__()
        s.ln1 = nn.LayerNorm(D_HID); s.fc1 = nn.Linear(D_HID, D_HID)
        s.ln2 = nn.LayerNorm(D_HID); s.fc2 = nn.Linear(D_HID, D_HID)
        s.film = nn.Linear(D_HID, 2 * D_HID)
    def forward(s, h, t_emb):
        # h: (B, N, D_HID), t_emb: (B, D_HID)
        gamma, beta = s.film(t_emb).unsqueeze(1).chunk(2, dim=-1)
        h_res = h
        h = s.fc1(F.gelu(s.ln1(h))) * (1 + gamma) + beta
        h = s.fc2(F.gelu(s.ln2(h)))
        return h_res + h


class TokenDiffusionDrafterV2(nn.Module):
    def __init__(s):
        super().__init__()
        s.pos_emb = nn.Embedding(N, D_EMB)
        s.z_proj = nn.Linear(D_Z, D_HID)
        s.t_proj = nn.Linear(D_T, D_HID)
        s.in_proj = nn.Linear(D_HID + 2 * D_EMB, D_HID)
        s.blocks = nn.ModuleList([ResBlockV2(D_HID) for _ in range(N_LAYER)])
        s.ln = nn.LayerNorm(D_HID)
        s.out = nn.Linear(D_HID, D_EMB)
        s.head = nn.Linear(D_EMB, V)

    def forward(s, z, t, noise):
        B_, N_, D_ = noise.shape

        # 条件
        z_cond = s.z_proj(z)  # (B, D_HID)
        t_emb = _sinusoidal_t(t, D_T).to(z.device)  # (B, D_T)
        t_emb = s.t_proj(t_emb)  # (B, D_HID)
        cond = (z_cond + t_emb).unsqueeze(1).expand(-1, N_, -1)  # (B, N, D_HID)

        # 位置
        pos = s.pos_emb(torch.arange(N_, device=noise.device)).unsqueeze(0).expand(B_, -1, -1)

        x = torch.cat([cond, pos, noise], dim=-1)
        x = s.in_proj(x)

        for blk in s.blocks:
            x = blk(x, z_cond + t_emb)

        x = s.ln(x)
        v_pred = s.out(x)
        return v_pred


def _sinusoidal_t(t, dim):
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, device=t.device, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half))
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1) * (dim ** 0.5)


# ===== 模型 =====
model = TokenDiffusionDrafterV2().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
P(f"\nTokenDiffusionDrafter: {n_params/1e6:.2f}M params")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))


def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (STEPS - WARMUP_STEPS)
    return 0.5 * (1.0 + np.cos(np.pi * progress))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

# ===== Embedding & Head (Tied Weights!) =====
# 关键: tok_emb 和 head 必须 tied weights, 否则学不到正确的 token 映射
tok_emb = nn.Embedding(V, D_EMB).to(DEVICE)
# head.weight = tok_emb.weight.T 的语义: 给 embedding 算 logits
# 等价于 nn.Linear(D_EMB, V, bias=False) 且 weight = tok_emb.weight
# 但 Linear 是 (out, in), 所以 head.weight.shape = (V, D_EMB) = tok_emb.weight.shape
# 我们把 head 的 weight 直接设为 tok_emb.weight (共享!)
head = nn.Linear(D_EMB, V, bias=False).to(DEVICE)
head.weight = tok_emb.weight  # TIED WEIGHTS!
P(f"  tok_emb + tied head: {sum(p.numel() for p in tok_emb.parameters())/1e6:.2f}M params (TIED)")
opt.add_param_group({"params": tok_emb.parameters(), "lr": LR})

# 替换 drafter 中的 head 为 tied 版本
model.head = head

# ===== 训练 (CFM) =====
P(f"\n=== 训练 {STEPS} steps, B={B}, LR={LR} ===")
t0 = time.time()
log = []
for step in range(STEPS):
    model.train(); tok_emb.train()
    ix = np.random.randint(0, TOKENS_ALL.size(0), B)
    z = Z_ALL[ix].to(DEVICE)                # (B, 256)
    tokens = TOKENS_ALL[ix].to(DEVICE)      # (B, 100)

    # target embedding
    target_emb = tok_emb(tokens)            # (B, 100, D_EMB)

    # 随机 t
    t = torch.rand(B, device=DEVICE)         # (B,)

    # 加噪
    noise = torch.randn_like(target_emb)
    z_t = (1 - t[:, None, None]) * noise + t[:, None, None] * target_emb
    v_target = target_emb - noise

    # 预测
    v_pred = model(z, t, z_t)               # (B, 100, D_EMB)
    loss = F.mse_loss(v_pred, v_target)

    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(tok_emb.parameters()), 1.0)
    opt.step(); sched.step()

    if step % EVAL_EVERY == 0 or step == STEPS - 1:
        elapsed = time.time() - t0
        eta = elapsed / max(step, 1) * (STEPS - step)
        # 验证: 计算 token match 率 (用 head)
        with torch.no_grad():
            model.eval(); tok_emb.eval()
            # 跑 5 步 Euler ODE 推理
            z_test = Z_ALL[:4].to(DEVICE)
            tokens_test = TOKENS_ALL[:4].to(DEVICE)
            target_emb_test = tok_emb(tokens_test)
            x_t = torch.randn(4, N, D_EMB, device=DEVICE)
            dt = 1.0 / 5
            for k in range(5):
                t_val = k * dt
                t_tensor = torch.full((4,), t_val, device=DEVICE)
                v = model(z_test, t_tensor, x_t)
                x_t = x_t + dt * v
            pred_emb = x_t
            pred_logits = model.head(pred_emb)
            pred_tokens = pred_logits.argmax(dim=-1)
            match = (pred_tokens == tokens_test).float().mean().item()
            # 同时检查 target_emb 通过 head 是否能恢复 token (验证 tied weights)
            target_match = (model.head(target_emb_test).argmax(dim=-1) == tokens_test).float().mean().item()
        P(f"  step {step:4d}/{STEPS} | loss {loss.item():.4f} | pred_match {match*100:.1f}% | target_match {target_match*100:.1f}% "
          f"| LR {sched.get_last_lr()[0]:.2e} | {elapsed:.0f}s ETA {eta:.0f}s")
        log.append({"step": step, "loss": loss.item(), "pred_match": match, "target_match": target_match,
                    "lr": sched.get_last_lr()[0]})

SAVE = "v29_token_diff.pt"
torch.save({"model": model.state_dict(),  # model.head.weight 已经 = tok_emb.weight (tied)
            "tok_emb": tok_emb.state_dict(),
            "config": {"D_Z": D_Z, "N": N, "D_EMB": D_EMB, "D_HID": D_HID, "D_T": D_T,
                       "N_LAYER": N_LAYER, "STEPS": STEPS, "B": B, "LR": LR,
                       "arch": "v29-TokenDiffusionDrafter-CFM-N=100-TIED-WEIGHTS"}},
           SAVE)
P(f"\nModel saved: {SAVE} ({os.path.getsize(SAVE)/1e6:.1f} MB)")

with open("v29_token_diff_log.json", "w", encoding="utf-8") as f:
    json.dump({"log": log,
               "config": {"STEPS": STEPS, "B": B, "LR": LR,
                          "D_HID": D_HID, "N_LAYER": N_LAYER, "N": N, "D_EMB": D_EMB,
                          "model_params_M": n_params/1e6,
                          "n_train_samples": TOKENS_ALL.size(0),
                          "arch": "v29-TokenDiffusionDrafter-CFM-N=100"}}, f, indent=2)
P(f"\n=== 训练完成 ({time.time()-t0:.0f}s) ===")