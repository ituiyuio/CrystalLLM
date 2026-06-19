# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
proto_v15_xattn.py — v15: cross-attention z 注入 + 300M 规模 (修复版)

v15.1 修复:
- 去掉 z_proj+bias 路径 (与 cross-attn 双重注入冲突)
- encode 用纯 self-attn (无 cross-attn), decode 用 self-attn + cross-attn
- 增加 steps 1500 → 2000, lr 2e-4 → 3e-4, B 16 → 24

核心变化 (vs v14):
1. z 注入路径: v14 仅在 prefix 第 0 位置注入 z_emb
   → v15 每个 decode block 都做 cross-attn, 让 z 作为 K/V 被 AR token 查询
2. 规模: 52M → ~240M (N_LAYER=16, N_EMBD=1024, N_HEAD=16)
3. W_THEME: 0.1 → 0.3 (强化主题监督)

架构:
  Encode (纯 self-attn):
    tok_emb + pos_emb → 16 × BlockPure(self-attn + MLP) → mean pool → z_enc → z
  Decode (self-attn + cross-attn to z):
    tok_emb + pos_emb → 16 × BlockXattn(self-attn + cross-attn z + MLP) → ln_f → head

主题损失: L = L_pred + 0.4·L_recon + 0.05·L_diff + 0.3·L_theme
"""
import json, math, time, random, sys, io
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
torch.manual_seed(42); random.seed(42); np.random.seed(42)

DATA = Path("crystalllm/data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]
itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]

THEMES = ["UE_CPP", "JS_REACT"]
THEME_TO_ID = {"D--UnrealEngine-CODEO": 0, "D--long-running-harness": 1}
N_THEMES = 2

df = pd.read_parquet(DATA / "subset_2000.parquet")
df = df[df["project"].isin(THEME_TO_ID.keys())].reset_index(drop=True)
df["theme_id"] = df["project"].map(THEME_TO_ID)
print(f"主题分布: {df['theme_id'].value_counts().to_dict()}")
print(f"总 sessions: {len(df)}")

texts_themes = [(row["text"], int(row["theme_id"])) for _, row in df.iterrows()]
all_text = "\n".join(t for t, _ in texts_themes)
print(f"Vocab {V}  |  text {len(all_text):,} chars")

train_items, val_items = [], []
for theme_id in [0, 1]:
    items_t = [it for it in texts_themes if it[1] == theme_id]
    random.shuffle(items_t)
    n_val = max(int(0.1 * len(items_t)), 5)
    val_items.extend(items_t[:n_val])
    train_items.extend(items_t[n_val:])
random.shuffle(train_items); random.shuffle(val_items)
print(f"train sessions: {len(train_items)}  |  val sessions: {len(val_items)}")
print(f"  val 主题分布: {pd.Series([t for _, t in val_items]).value_counts().to_dict()}")

# ====== v15.2 配置: 440M ======
B, T, D_Z      = 24, 256, 64
T_HALF         = T // 2
N_LAYER, N_HEAD, N_EMBD = 16, 16, 1024  # ~440M (独立 enc + dec)
LR, STEPS      = 3e-4, 2000
EVAL_EVERY     = 250
W_PRED, W_RECON, W_DIFF, W_THEME = 1.0, 0.4, 0.05, 0.05  # W_THEME 大幅降低
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
PAD_ID         = stoi.get(' ', 0)
# v15.2: UE_CPP 样本权重 (反频率), 解决 class imbalance
THEME_WEIGHTS = torch.tensor([800/481, 481/800], device=DEVICE)  # UE_CPP=1.66, JS_REACT=0.60


def get_batch(items, B_local):
    ix = np.random.randint(0, len(items), B_local)
    fulls, themes = [], []
    for i in ix:
        text, theme = items[i]
        if len(text) < T + 2:
            text = text + "\n" * (T + 2 - len(text))
        start = random.randint(0, len(text) - T - 2)
        chunk = text[start:start + T + 2]
        fulls.append([stoi[c] for c in chunk])
        themes.append(theme)
    full = torch.tensor(fulls, dtype=torch.long).to(DEVICE)
    theme = torch.tensor(themes, dtype=torch.long).to(DEVICE)
    return full[:, :T_HALF], full[:, T_HALF:], theme


class BlockPure(nn.Module):
    """v15.1 Encode block: 纯 self-attn + MLP. 没有 cross-attn (z 是从 mean pool 抽的)."""
    def __init__(s, N_EMBD, N_HEAD):
        super().__init__()
        s.nh = N_HEAD
        s.head_dim = N_EMBD // N_HEAD
        s.ln1 = nn.LayerNorm(N_EMBD)
        s.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        s.proj = nn.Linear(N_EMBD, N_EMBD)
        s.ln2 = nn.LayerNorm(N_EMBD)
        s.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD)
        )

    def forward(s, x):
        B_, T_, C = x.shape
        h = s.ln1(x)
        qkv = s.qkv(h).reshape(B_, T_, 3, s.nh, s.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(y.transpose(1, 2).contiguous().view(B_, T_, C))
        x = x + s.mlp(s.ln2(x))
        return x


class BlockXattn(nn.Module):
    """v15.1 Decode block: self-attn + cross-attn to z + MLP. (无 z_proj 重复注入)"""
    def __init__(s, N_EMBD, N_HEAD, D_Z):
        super().__init__()
        s.nh = N_HEAD
        s.head_dim = N_EMBD // N_HEAD
        # Self-attention
        s.ln1 = nn.LayerNorm(N_EMBD)
        s.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        s.proj = nn.Linear(N_EMBD, N_EMBD)
        # Cross-attention to z
        s.ln_q = nn.LayerNorm(N_EMBD)
        s.q_cross = nn.Linear(N_EMBD, N_EMBD)
        s.kv_cross = nn.Linear(D_Z, 2 * N_EMBD)
        s.proj_cross = nn.Linear(N_EMBD, N_EMBD)
        # MLP
        s.ln2 = nn.LayerNorm(N_EMBD)
        s.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD)
        )

    def forward(s, x, z):
        B_, T_, C = x.shape
        h = s.ln1(x)
        qkv = s.qkv(h).reshape(B_, T_, 3, s.nh, s.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(y.transpose(1, 2).contiguous().view(B_, T_, C))
        # Cross-attention to z
        q_c = s.q_cross(s.ln_q(x)).reshape(B_, T_, s.nh, s.head_dim).transpose(1, 2)
        kv = s.kv_cross(z).unsqueeze(1)
        k_c, v_c = kv.chunk(2, dim=-1)
        k_c = k_c.reshape(B_, 1, s.nh, s.head_dim).transpose(1, 2)
        v_c = v_c.reshape(B_, 1, s.nh, s.head_dim).transpose(1, 2)
        y_c = F.scaled_dot_product_attention(q_c, k_c, v_c)
        x = x + s.proj_cross(y_c.transpose(1, 2).contiguous().view(B_, T_, C))
        x = x + s.mlp(s.ln2(x))
        return x


class Diffusion(nn.Module):
    def __init__(s, D_Z, N_EMBD):
        super().__init__()
        s.net = nn.Sequential(
            nn.Linear(D_Z * 2, N_EMBD), nn.SiLU(),
            nn.Linear(N_EMBD, N_EMBD), nn.SiLU(),
            nn.Linear(N_EMBD, D_Z)
        )
    def step(s, z, t):
        return z - 0.3 * s.net(torch.cat([z, t.view(1, 1).expand(z.size(0), z.size(1))], dim=-1))
    def denoise(s, z, K=5):
        for i in range(K - 1, -1, -1):
            z = s.step(z, torch.tensor(i / K, device=z.device))
        return z


class ControllableHybridV2(nn.Module):
    """v15.1: 独立 encoder (BlockPure) + decoder (BlockXattn)."""
    def __init__(s):
        super().__init__()
        s.tok = nn.Embedding(V, N_EMBD)
        s.pos = nn.Embedding(T, N_EMBD)
        # 独立 encoder blocks (纯 self-attn)
        s.enc_blocks = nn.ModuleList([BlockPure(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        # 独立 decoder blocks (self-attn + cross-attn to z)
        s.dec_blocks = nn.ModuleList([BlockXattn(N_EMBD, N_HEAD, D_Z) for _ in range(N_LAYER)])
        s.ln_f_enc = nn.LayerNorm(N_EMBD)
        s.ln_f_dec = nn.LayerNorm(N_EMBD)
        s.head = nn.Linear(N_EMBD, V, bias=False)
        s.tok.weight = s.head.weight  # tied
        s.z_enc = nn.Linear(N_EMBD, D_Z)
        s.z_to_chars = nn.Linear(D_Z, V)
        s.diff = Diffusion(D_Z, N_EMBD)
        s.theme_classifier = nn.Sequential(
            nn.Linear(D_Z, D_Z), nn.SiLU(),
            nn.Linear(D_Z, N_THEMES)
        )

    def encode(s, prefix):
        """Encode prefix → z (D_Z=64). 纯 self-attn."""
        B_, T_p = prefix.shape
        h = s.tok(prefix) + s.pos(torch.arange(T_p, device=prefix.device))
        for block in s.enc_blocks:
            h = block(h)
        h = s.ln_f_enc(h)
        return s.z_enc(h.mean(dim=1))

    def decode(s, z, suffix):
        """Decode z + suffix → logits. self-attn + cross-attn to z."""
        B_, T_s = suffix.shape
        sfx_emb = s.tok(suffix) + s.pos(torch.arange(1, T_s + 1, device=suffix.device))
        x = sfx_emb
        for block in s.dec_blocks:
            x = block(x, z)
        h = s.ln_f_dec(x)
        return s.head(h)

    def forward(s, prefix, suffix):
        z = s.encode(prefix)
        logits = s.decode(z, suffix)
        recon = s.z_to_chars(z.unsqueeze(1).expand(-1, prefix.size(1), -1))
        theme_logits = s.theme_classifier(z)
        return logits, z, recon, theme_logits

    @torch.no_grad()
    def gen(s, seed, n=120, t=0.8, z_override=None, use_real_starter=True, K_diff=5, from_noise=False):
        s.eval()
        seed_ids = [stoi[c] for c in seed]
        if z_override is not None:
            z = z_override
        elif from_noise:
            z = torch.randn(1, D_Z, device=DEVICE)
            z = s.diff.denoise(z, K=K_diff)
        else:
            ids = torch.tensor([seed_ids[:T_HALF]], device=DEVICE, dtype=torch.long)
            if ids.size(1) < T_HALF:
                ids = F.pad(ids, (0, T_HALF - ids.size(1)), value=PAD_ID)
            z = s.encode(ids)
        if use_real_starter:
            n_seed = len(seed_ids)
            pos = random.randint(0, len(all_text) - T_HALF - 3)
            starter_text = all_text[pos:pos + T_HALF + 2]
            suffix = list(seed_ids) + [stoi[c] for c in starter_text[n_seed:T_HALF + 2]]
        else:
            suffix = list(seed_ids) + [PAD_ID] * (T_HALF + 2 - len(seed_ids))
            suffix = suffix[:T_HALF + 2]
        sfx_t = torch.tensor([suffix], device=DEVICE, dtype=torch.long)
        out = list(seed_ids)
        for _ in range(n):
            logits = s.decode(z, sfx_t)
            tok = min(int(torch.multinomial(F.softmax(logits[0, T_HALF + 1] / t, -1), 1)), V - 1)
            if tok == 1: break
            out.append(tok)
            suffix = suffix[1:] + [tok]
            sfx_t = torch.tensor([suffix], device=DEVICE, dtype=torch.long)
        s.train()
        return "".join(itos[i] for i in out)


model = ControllableHybridV2().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel v15.1: {n_params/1e6:.2f}M params")
print(f"  Config: {N_LAYER}L × {N_EMBD} embd × {N_HEAD} head × D_Z={D_Z}")
print(f"  Encoder: {N_LAYER} × BlockPure (self-attn)")
print(f"  Decoder: {N_LAYER} × BlockXattn (self-attn + cross-attn to z)")
print(f"  W_PRED={W_PRED} W_RECON={W_RECON} W_DIFF={W_DIFF} W_THEME={W_THEME}")
print(f"  Steps={STEPS}, LR={LR}, B={B}, T={T}")
print(f"  Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB)")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

print(f"\n=== train {STEPS} steps ===")
t0 = time.time()
log = []
for step in range(STEPS):
    prefix, suffix, theme = get_batch(train_items, B)
    logits, z, recon, theme_logits = model(prefix, suffix)
    T_s = suffix.size(1)
    loss_pred = F.cross_entropy(logits[:, :T_s].reshape(-1, V), suffix.reshape(-1))
    loss_recon = F.cross_entropy(recon.reshape(-1, V), prefix.reshape(-1))
    z_noisy = z + 1.0 * torch.randn_like(z)
    z_denoised = model.diff.denoise(z_noisy)
    loss_diff = (z_denoised - z.detach()).pow(2).mean()
    # v15.2: 用 class weights 解决 UE_CPP 总是被预测为 JS_REACT 的问题
    loss_theme = F.cross_entropy(theme_logits, theme, weight=THEME_WEIGHTS)
    loss = W_PRED * loss_pred + W_RECON * loss_recon + W_DIFF * loss_diff + W_THEME * loss_theme
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % EVAL_EVERY == 0 or step == STEPS - 1:
        model.eval()
        with torch.no_grad():
            vp, vs, vt = get_batch(val_items, B)
            vlogits, vz, vrecon, vtheme_logits = model(vp, vs)
            vloss_pred = F.cross_entropy(vlogits[:, :vs.size(1)].reshape(-1, V), vs.reshape(-1))
            vloss_recon = F.cross_entropy(vrecon.reshape(-1, V), vp.reshape(-1))
            vloss_theme = F.cross_entropy(vtheme_logits, vt, weight=THEME_WEIGHTS)
            vtheme_acc = (vtheme_logits.argmax(-1) == vt).float().mean().item()
        model.train()
        log.append((step, loss_pred.item(), vloss_pred.item(), vloss_recon.item(),
                    loss_diff.item(), loss_theme.item(), vtheme_acc))
        elapsed = time.time() - t0
        eta = elapsed / max(step, 1) * (STEPS - step)
        print(f"  step {step:4d}/{STEPS} | pred {loss_pred.item():.3f} | val_pred {vloss_pred.item():.3f} "
              f"| val_theme {vloss_theme.item():.3f} | val_theme_acc {vtheme_acc:.3f} "
              f"| diff {loss_diff.item():.3f} | {elapsed:.0f}s ETA {eta:.0f}s")

# ===== 全集分类器 acc =====
print("\n=== 主题分类器全集 acc ===")
model.eval()
all_z, all_theme = [], []
with torch.no_grad():
    eval_items = train_items + val_items
    for _ in range(30):
        ix = np.random.randint(0, len(eval_items), B)
        fulls, themes = [], []
        for i in ix:
            text, theme = eval_items[i]
            if len(text) < T + 2: text = text + "\n" * (T + 2 - len(text))
            start = random.randint(0, len(text) - T - 2)
            fulls.append([stoi[c] for c in text[start:start + T + 2]])
            themes.append(theme)
        full = torch.tensor(fulls, dtype=torch.long).to(DEVICE)
        theme = torch.tensor(themes, dtype=torch.long).to(DEVICE)
        vz = model.encode(full[:, :T_HALF])
        all_z.append(vz.cpu().numpy()); all_theme.append(theme.cpu().numpy())
all_z = np.concatenate(all_z, axis=0)
all_theme = np.concatenate(all_theme, axis=0)
theme_logits = model.theme_classifier(torch.tensor(all_z, device=DEVICE))
preds = theme_logits.argmax(-1).cpu().numpy()
acc = (preds == all_theme).mean()
per_theme = {}
for t_id, t_name in enumerate(THEMES):
    mask = all_theme == t_id
    per_theme[t_name] = float((preds[mask] == t_id).mean()) if mask.sum() > 0 else 0.0
    print(f"  {t_name}: n={mask.sum()}, acc={per_theme[t_name]:.3f}")
print(f"  Overall: {acc:.3f}")

# ===== z_means =====
z_means = {}
for t_id, t_name in enumerate(THEMES):
    mask = all_theme == t_id
    z_means[t_name] = all_z[mask].mean(axis=0)

# ===== 可控性测试 =====
print("\n=== 可控性测试: 沿主题梯度编辑 z ===")
def edit_z_for_theme(z, target_theme, n_steps=10, lr=0.5):
    z = z.clone().detach().requires_grad_(True)
    target = torch.tensor([target_theme], device=DEVICE, dtype=torch.long)
    for _ in range(n_steps):
        theme_logits = model.theme_classifier(z)
        loss = F.cross_entropy(theme_logits, target)
        grad = torch.autograd.grad(loss, z)[0]
        z = (z - lr * grad).detach().requires_grad_(True)
    return z.detach()

for src_name, tgt_name in [("UE_CPP", "JS_REACT"), ("JS_REACT", "UE_CPP")]:
    src_id = THEMES.index(src_name); tgt_id = THEMES.index(tgt_name)
    z_src = torch.tensor(z_means[src_name], device=DEVICE, dtype=torch.float32).unsqueeze(0)
    pred_start = model.theme_classifier(z_src).argmax(-1).item()
    z_edited = edit_z_for_theme(z_src, target_theme=tgt_id, n_steps=30, lr=2.0)
    pred_end = model.theme_classifier(z_edited).argmax(-1).item()
    print(f"  [{src_name}→{tgt_name}] 起始={pred_start} (期望 {src_id}) → 编辑后={pred_end} (期望 {tgt_id})")

def safe(s): return ''.join(c if ord(c) < 128 else '?' for c in s)

# ===== 生成样例: 真实 prefix 编码 + 主题梯度编辑 =====
print("\n=== 生成样例 v15.2: 真实 prefix encode + 沿主题梯度编辑 ===")
seeds = ["def ", "void ", "class ", "const "]
# 收集带主题标签的真实 prefix 样本 (用真实 prefix → encode → 真实 z)
n_demo = 8
demo_items = []
for theme_id in [0, 1]:
    items_t = [it for it in train_items if it[1] == theme_id]
    random.shuffle(items_t)
    demo_items.extend(items_t[:n_demo // 2])

print(f"  Demo samples: {len(demo_items)} ({sum(1 for _, t in demo_items if t==0)} UE_CPP + {sum(1 for _, t in demo_items if t==1)} JS_REACT)")
real_z_by_theme = {0: [], 1: []}
with torch.no_grad():
    for text, theme in demo_items:
        if len(text) < T + 2: text = text + "\n" * (T + 2 - len(text))
        start = random.randint(0, len(text) - T - 2)
        prefix_ids = torch.tensor([[stoi[c] for c in text[start:start + T_HALF]]], device=DEVICE)
        z_real = model.encode(prefix_ids)
        real_z_by_theme[theme].append(z_real)
for t_id in [0, 1]:
    real_z_by_theme[t_id] = torch.cat(real_z_by_theme[t_id], dim=0)
print(f"  真实 UE_CPP z 范数: {real_z_by_theme[0].norm(dim=-1).mean().item():.2f} ± {real_z_by_theme[0].norm(dim=-1).std().item():.2f}")
print(f"  真实 JS_REACT z 范数: {real_z_by_theme[1].norm(dim=-1).mean().item():.2f} ± {real_z_by_theme[1].norm(dim=-1).std().item():.2f}")

def edit_real_z(z_real, target_theme, n_steps=30, lr=2.0):
    """对真实编码的 z 做主题梯度编辑."""
    z = z_real.clone().detach().requires_grad_(True)
    target = torch.tensor([target_theme], device=DEVICE, dtype=torch.long)
    for _ in range(n_steps):
        loss = F.cross_entropy(model.theme_classifier(z), target)
        grad = torch.autograd.grad(loss, z)[0]
        z = (z - lr * grad).detach().requires_grad_(True)
    return z.detach()

# 用真实 prefix 起始, 沿主题梯度编辑
all_demo = []
for i, (text, theme) in enumerate(demo_items):
    if len(text) < T + 2: text = text + "\n" * (T + 2 - len(text))
    start = random.randint(0, len(text) - T - 2)
    prefix_str = text[start:start + 20]  # 前 20 字符作为 seed
    z_real = real_z_by_theme[theme][i // (n_demo // 2) if n_demo // 2 > 0 else 0] if False else None
    # 重新 encode 真实 prefix
    prefix_ids = torch.tensor([[stoi[c] for c in text[start:start + T_HALF]]], device=DEVICE)
    with torch.no_grad():
        z_real = model.encode(prefix_ids)
    target_theme = 1 - theme  # 切换到另一主题
    z_edited = edit_real_z(z_real, target_theme=target_theme, n_steps=30, lr=2.0)
    pred_src = model.theme_classifier(z_real).argmax(-1).item()
    pred_edit = model.theme_classifier(z_edited).argmax(-1).item()
    torch.manual_seed(0)
    out_src = model.gen(prefix_str, n=60, z_override=z_real, use_real_starter=True, t=0.8)
    torch.manual_seed(0)
    out_edit = model.gen(prefix_str, n=60, z_override=z_edited, use_real_starter=True, t=0.8)
    src_name = THEMES[theme]; tgt_name = THEMES[target_theme]
    print(f"\n  [{src_name}→{tgt_name}] theme 预测 {pred_src}→{pred_edit}")
    print(f"    prefix:  {safe(prefix_str)}")
    print(f"    src:     {safe(out_src[20:80])}")
    print(f"    edited:  {safe(out_edit[20:80])}")
    all_demo.append({"theme_src": src_name, "theme_tgt": tgt_name,
                     "prefix": prefix_str,
                     "pred_src": pred_src, "pred_edit": pred_edit,
                     "src_text": out_src[20:80], "edit_text": out_edit[20:80]})

# ===== 保存 =====
SAVE_PATH = "crystalllm/proto_v15_xattn_model.pt"
torch.save({
    "model_state_dict": model.state_dict(),
    "config": {"V": V, "T": T, "D_Z": D_Z, "T_HALF": T_HALF,
               "N_LAYER": N_LAYER, "N_HEAD": N_HEAD, "N_EMBD": N_EMBD,
               "N_THEMES": N_THEMES, "THEMES": THEMES,
               "z_means": {k: v.tolist() for k, v in z_means.items()}}
}, SAVE_PATH)
print(f"\nModel saved: {SAVE_PATH}")

out_json = {
    "log": log,
    "val_theme_acc": float(acc),
    "per_theme_acc": per_theme,
    "demo_samples": all_demo,
    "config": {"STEPS": STEPS, "W_THEME": W_THEME, "N_LAYER": N_LAYER, "N_EMBD": N_EMBD,
               "N_HEAD": N_HEAD, "B": B, "T": T, "D_Z": D_Z, "LR": LR,
               "arch": "xattn-z-injection-v15.2"}
}
with open("crystalllm/v15_train_log.json", "w", encoding="utf-8") as f:
    json.dump(out_json, f, indent=2, ensure_ascii=False)
print(f"Log saved: crystalllm/v15_train_log.json")
print(f"\n=== v15 训练完成 ({time.time()-t0:.0f}s) ===")
