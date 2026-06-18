"""
train_v30_diff_kv.py — v30 KV 扩散模型训练 (重做 v27)

核心改进 (vs v27):
- 模型: 13M → 100M (12 ResBlock × 1536)
- 训练: 直接回归 → CFM (与 v24 prior 一致)
- 数据: 200 → 500 样本
- 输入: z (256, prior 采样) + t_emb (128)
- 输出: KV latent (用 PCA 压缩到 128 维)

输出: v30_diff_kv.pt
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


P("=== v30 KV 扩散模型训练 (CFM, 100M) ===")

# ===== 加载 KV 数据 =====
P("Loading cached_v30_kv.npz ...")
data = np.load("cached_v30_kv.npz")
KV_ALL = data["kv"].astype(np.float32)  # (N, 28, 2, 20, 103, 64) float16 -> float32 (v28.5 28 层)
Z_ALL = torch.tensor(data["z"], dtype=torch.float32)  # (N, 256)
P(f"  kv: {KV_ALL.shape}, dtype={KV_ALL.dtype}")
P(f"  z: {Z_ALL.shape}")
P(f"  KV 范围: [{KV_ALL.min():.3f}, {KV_ALL.max():.3f}]")
N, L, S, H, T_kv, D = KV_ALL.shape
P(f"  N={N}, L={L} 层, S={S} (K/V), H={H} 头, T_kv={T_kv} tokens, D={D} dim/head")
KV_ALL = torch.tensor(KV_ALL.reshape(N, -1))  # (N, 7.4M)
P(f"  KV 扁平化: {KV_ALL.shape}")

# ===== PCA 降维 =====
P("\nFitting PCA (128 维) ...")
PCA_DIM = 128
# 转置: (N, D) -> mean subtraction
kv_mean = KV_ALL.mean(dim=0, keepdim=True)  # (1, D)
KV_CENTERED = KV_ALL - kv_mean

# SVD
U, S, Vt = torch.pca_lowrank(KV_CENTERED, q=PCA_DIM, center=False)
P(f"  U: {U.shape}, S[:5]: {S[:5].tolist()}")
P(f"  S 总和: {S.sum().item():.2f}")
P(f"  S 累计解释方差: {(S.cumsum(0)[-1] / S.sum()).item() * 100:.2f}%")

KV_LATENT = KV_CENTERED @ Vt[:, :PCA_DIM]  # (N, 128)
P(f"  KV_LATENT: {KV_LATENT.shape}")

# 标准化
LATENT_MEAN = KV_LATENT.mean(dim=0, keepdim=True)
LATENT_STD = KV_LATENT.std(dim=0, keepdim=True).clamp(min=1e-3)
KV_LATENT_NORM = (KV_LATENT - LATENT_MEAN) / LATENT_STD
P(f"  标准化: mean {KV_LATENT_NORM.mean():.3f}, std {KV_LATENT_NORM.std():.3f}")

# ===== 配置 =====
D_Z = 256
D_LATENT = PCA_DIM
D_HID = 1536      # 更大
D_T = 128
N_LAYER = 12      # 12 ResBlock (vs v27 6)
LR = 1e-4
STEPS = 4000
B = 8             # 数据少, batch 小
DEVICE = "cuda"


# ===== 模型 =====
class SinusoidalTimeEmbed(nn.Module):
    def __init__(s, dim):
        super().__init__()
        s.dim = dim
    def forward(s, t):
        half = s.dim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half))
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1) * (s.dim ** 0.5)


class ResBlock(nn.Module):
    """CFM ResBlock with FiLM (t 调制)"""
    def __init__(s, D_HID):
        super().__init__()
        s.ln1 = nn.LayerNorm(D_HID); s.fc1 = nn.Linear(D_HID, D_HID)
        s.ln2 = nn.LayerNorm(D_HID); s.fc2 = nn.Linear(D_HID, D_HID)
        s.film = nn.Linear(D_HID, 2 * D_HID)
    def forward(s, h, t_emb):
        gamma, beta = s.film(t_emb).chunk(2, dim=-1)
        h_res = h
        h = s.fc1(F.gelu(s.ln1(h))) * (1 + gamma) + beta
        h = s.fc2(F.gelu(s.ln2(h)))
        return h_res + h


class DiffKVGenerator(nn.Module):
    """
    v30 KV 扩散生成器 (vs v27 KVGenerator)

    输入: z (256), t (B,), noise (B, 128)
    输出: v_pred (B, 128) - CFM 速度预测
    """
    def __init__(s):
        super().__init__()
        s.z_proj = nn.Linear(D_Z, D_HID)
        s.t_proj = nn.Linear(D_T, D_HID)
        s.in_proj = nn.Linear(D_HID, D_HID)

        s.blocks = nn.ModuleList([ResBlock(D_HID) for _ in range(N_LAYER)])
        s.ln = nn.LayerNorm(D_HID)
        s.out = nn.Linear(D_HID, D_LATENT)

    def forward(s, z, t, noise):
        """
        z: (B, 256)
        t: (B,)
        noise: (B, 128)
        """
        z_cond = s.z_proj(z)  # (B, D_HID)

        half = D_T // 2
        freqs = torch.exp(-torch.arange(half, device=z.device, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half))
        args = t.float()[:, None] * freqs[None]
        t_emb_raw = torch.cat([torch.sin(args), torch.cos(args)], dim=-1) * (D_T ** 0.5)
        t_emb = s.t_proj(t_emb_raw)  # (B, D_HID)

        x = s.in_proj(z_cond + t_emb)  # (B, D_HID)

        for blk in s.blocks:
            x = blk(x, t_emb)

        x = s.ln(x)
        return s.out(x)


model = DiffKVGenerator().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
P(f"\nDiffKVGenerator: {n_params/1e6:.2f}M params (vs v27 13M)")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))


def lr_lambda(step):
    if step < 200:
        return step / 200
    progress = (step - 200) / (STEPS - 200)
    return 0.5 * (1.0 + np.cos(np.pi * progress))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

# ===== 训练 (CFM) =====
P(f"\n=== 训练 {STEPS} steps, B={B}, LR={LR} ===")
t0 = time.time()
log = []
for step in range(STEPS):
    model.train()
    ix = np.random.randint(0, KV_LATENT_NORM.size(0), B)
    z = Z_ALL[ix].to(DEVICE)                       # (B, 256)
    latent_true = KV_LATENT_NORM[ix].to(DEVICE)   # (B, 128) normalized

    # CFM: z_t = (1-t)*noise + t*target
    t = torch.rand(B, device=DEVICE)
    noise = torch.randn_like(latent_true)
    z_t = (1 - t[:, None]) * noise + t[:, None] * latent_true
    v_target = latent_true - noise  # CFM 速度目标

    v_pred = model(z, t, z_t)  # (B, 128)
    loss = F.mse_loss(v_pred, v_target)

    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()

    if step % 200 == 0 or step == STEPS - 1:
        elapsed = time.time() - t0
        eta = elapsed / max(step, 1) * (STEPS - step)
        # 验证: 用 5 步采样恢复 latent, 看与真实的 cosine 相似度
        with torch.no_grad():
            model.eval()
            z_test = Z_ALL[:8].to(DEVICE)
            latent_true_test = KV_LATENT_NORM[:8].to(DEVICE)

            x_t = torch.randn(8, D_LATENT, device=DEVICE)
            dt = 1.0 / 5
            for k in range(1, 6):
                t_val = (k - 1) * dt
                t_tensor = torch.full((8,), t_val, device=DEVICE)
                v = model(z_test, t_tensor, x_t)
                x_t = x_t + dt * v
            cos_sim = F.cosine_similarity(x_t, latent_true_test, dim=-1).mean().item()

            # 反归一化
            x_t_unnorm = x_t * LATENT_STD.to(DEVICE) + LATENT_MEAN.to(DEVICE)
            latent_true_unnorm = latent_true_test * LATENT_STD.to(DEVICE) + LATENT_MEAN.to(DEVICE)
            cos_sim_real = F.cosine_similarity(x_t_unnorm, latent_true_unnorm, dim=-1).mean().item()

        P(f"  step {step:4d}/{STEPS} | loss {loss.item():.5f} | "
          f"cos_sim_norm {cos_sim:.4f} | cos_sim_real {cos_sim_real:.4f} | "
          f"LR {sched.get_last_lr()[0]:.2e} | {elapsed:.0f}s ETA {eta:.0f}s")
        log.append({"step": step, "loss": loss.item(),
                    "cos_sim_norm": cos_sim, "cos_sim_real": cos_sim_real,
                    "lr": sched.get_last_lr()[0]})

SAVE = "v30_diff_kv.pt"
torch.save({"model": model.state_dict(),
            "config": {"D_Z": D_Z, "D_LATENT": D_LATENT, "D_HID": D_HID, "D_T": D_T,
                       "N_LAYER": N_LAYER, "STEPS": STEPS, "B": B, "LR": LR,
                       "arch": "v30-DiffKVGenerator-CFM-100M"}},
           SAVE)
P(f"\nModel saved: {SAVE} ({os.path.getsize(SAVE)/1e6:.1f} MB)")

# 保存 PCA basis 和归一化参数
np.savez("v30_pca_basis.npz",
         mean=kv_mean.numpy(),
         V=Vt[:, :PCA_DIM].numpy(),
         S=S.numpy(),
         latent_mean=LATENT_MEAN.numpy(),
         latent_std=LATENT_STD.numpy())
P(f"PCA basis saved: v30_pca_basis.npz")

with open("v30_diff_kv_log.json", "w", encoding="utf-8") as f:
    json.dump({"log": log,
               "config": {"STEPS": STEPS, "B": B, "LR": LR,
                          "D_HID": D_HID, "N_LAYER": N_LAYER, "D_LATENT": D_LATENT,
                          "model_params_M": n_params/1e6,
                          "n_train_samples": KV_LATENT.size(0),
                          "arch": "v30-DiffKVGenerator-CFM-100M"}}, f, indent=2)
P(f"\n=== 训练完成 ({time.time()-t0:.0f}s) ===")