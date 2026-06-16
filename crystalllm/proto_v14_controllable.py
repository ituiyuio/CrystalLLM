"""
proto_v14_controllable.py — 监督可控 z 训练

v9 hybrid 架构 (52M) + 新增主题分类头.
训练时, z 必须能预测 prefix 主题 (UE_CPP / JS_REACT). 推理时沿主题梯度编辑 z, 主题应切换.

数据主题分布:
  - D--UnrealEngine-CODEO  (481 sessions, UE 5.7 C++)  → label=0 (UE_CPP)
  - D--long-running-harness (800 sessions, JS/React)   → label=1 (JS_REACT)

新加损失:
  L_theme = CE(theme_classifier(z), theme_label)  W=0.1

推理可控:
  - 给定 z, 计算 L_theme 对 z 的梯度 (向 target theme 方向)
  - 沿梯度方向更新 z (z_new = z - alpha * grad)
  - 用 z_new 生成, 看是否切到 target theme
"""
import json, math, time, random, sys, io
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
import numpy as np

# 修 stdout encoding for Windows gbk
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

torch.manual_seed(42); random.seed(42); np.random.seed(42)

DATA = Path("crystalllm/data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]
itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]

# 主题映射
THEMES = ["UE_CPP", "JS_REACT"]
THEME_TO_ID = {"D--UnrealEngine-CODEO": 0, "D--long-running-harness": 1}
N_THEMES = 2

df = pd.read_parquet(DATA / "subset_2000.parquet")
df = df[df["project"].isin(THEME_TO_ID.keys())].reset_index(drop=True)
df["theme_id"] = df["project"].map(THEME_TO_ID)
print(f"主题分布: {df['theme_id'].value_counts().to_dict()}")
print(f"总 sessions: {len(df)}")

# 构建 (text, theme_id) 列表
texts_themes = []
for _, row in df.iterrows():
    texts_themes.append((row["text"], int(row["theme_id"])))

all_text = "\n".join(t for t, _ in texts_themes)
data = torch.tensor([stoi[c] for c in all_text], dtype=torch.long)
print(f"Vocab {V}  |  text {len(all_text):,} chars")

# 训练/验证 split (按主题分层切, 保证 val 中两主题都有)
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

# 50M 配置 (与 v9 一致)
B, T, D_Z      = 32, 256, 64
T_HALF         = T // 2
N_LAYER, N_HEAD, N_EMBD = 16, 8, 512
LR, STEPS      = 3e-4, 3000
EVAL_EVERY     = 500
W_PRED, W_RECON, W_DIFF, W_THEME = 1.0, 0.4, 0.05, 0.1
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
PAD_ID         = stoi.get(' ', 0)

def get_batch(items, B_local):
    """从 items 采样 batch, 每个 item 切 T+2 长窗口."""
    ix = np.random.randint(0, len(items), B_local)
    fulls = []
    themes = []
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

def sample_real_starter(seed_ids, length):
    n_seed = len(seed_ids)
    pos = random.randint(0, len(all_text) - length - 1)
    starter_text = all_text[pos:pos + length]
    starter_ids = [stoi[c] for c in starter_text]
    return list(seed_ids) + starter_ids[n_seed:length]

class Block(nn.Module):
    def __init__(s, N_EMBD, N_HEAD):
        super().__init__()
        s.ln1 = nn.LayerNorm(N_EMBD); s.qkv = nn.Linear(N_EMBD, 3*N_EMBD)
        s.proj = nn.Linear(N_EMBD, N_EMBD); s.ln2 = nn.LayerNorm(N_EMBD)
        s.mlp = nn.Sequential(nn.Linear(N_EMBD, 4*N_EMBD), nn.GELU(),
                              nn.Linear(4*N_EMBD, N_EMBD))
        s.nh = N_HEAD
    def forward(s, x):
        B_, T_, C = x.shape
        h = s.ln1(x)
        qkv = s.qkv(h).reshape(B_, T_, 3, s.nh, C//s.nh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(y.transpose(1, 2).contiguous().view(B_, T_, C))
        x = x + s.mlp(s.ln2(x))
        return x

class Diffusion(nn.Module):
    def __init__(s, D_Z, N_EMBD):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(D_Z*2, N_EMBD), nn.SiLU(),
                              nn.Linear(N_EMBD, N_EMBD), nn.SiLU(), nn.Linear(N_EMBD, D_Z))
    def step(s, z, t):
        return z - 0.3 * s.net(torch.cat([z, t.view(1, 1).expand(z.size(0), z.size(1))], dim=-1))
    def denoise(s, z, K=5):
        for i in range(K-1, -1, -1):
            z = s.step(z, torch.tensor(i/K, device=z.device))
        return z

class ControllableHybrid(nn.Module):
    """v9 + 主题分类头."""
    def __init__(s):
        super().__init__()
        s.tok = nn.Embedding(V, N_EMBD); s.pos = nn.Embedding(T, N_EMBD)
        s.blocks = nn.Sequential(*[Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        s.ln_f = nn.LayerNorm(N_EMBD); s.head = nn.Linear(N_EMBD, V, bias=False)
        s.tok.weight = s.head.weight
        s.z_enc = nn.Linear(N_EMBD, D_Z)
        s.z_dec = nn.Linear(D_Z, N_EMBD)
        s.z_to_chars = nn.Linear(D_Z, V)
        s.diff = Diffusion(D_Z, N_EMBD)
        s.theme_classifier = nn.Sequential(
            nn.Linear(D_Z, D_Z), nn.SiLU(),
            nn.Linear(D_Z, N_THEMES)
        )
    def encode(s, prefix):
        h = s.tok(prefix) + s.pos(torch.arange(prefix.size(1), device=prefix.device))
        h = s.blocks(h); h = s.ln_f(h)
        return s.z_enc(h.mean(dim=1))
    def decode(s, z, suffix):
        B_, T_s = suffix.shape
        z_emb = s.z_dec(z).unsqueeze(1)
        sfx_emb = s.tok(suffix) + s.pos(torch.arange(1, T_s+1, device=suffix.device))
        x = torch.cat([z_emb, sfx_emb], dim=1)
        h = s.blocks(x); h = s.ln_f(h)
        return s.head(h)
    def forward(s, prefix, suffix):
        z = s.encode(prefix)
        logits = s.decode(z, suffix)
        recon = s.z_to_chars(z.unsqueeze(1).expand(-1, prefix.size(1), -1))
        theme_logits = s.theme_classifier(z)
        return logits, z, recon, theme_logits
    @torch.no_grad()
    def gen(s, seed, n=150, t=0.8, z_override=None, use_real_starter=True, K_diff=5, from_noise=False):
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
            tok = min(int(torch.multinomial(F.softmax(logits[0, T_HALF+1]/t, -1), 1)), V-1)
            if tok == 1: break
            out.append(tok)
            suffix = suffix[1:] + [tok]
            sfx_t = torch.tensor([suffix], device=DEVICE, dtype=torch.long)
        s.train()
        return "".join(itos[i] for i in out)

model = ControllableHybrid().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params/1e6:.2f}M params  |  {N_LAYER}L × {N_EMBD} embd × {N_HEAD} head  |  device: {DEVICE}")
print(f"  Config: W_PRED={W_PRED} W_RECON={W_RECON} W_DIFF={W_DIFF} W_THEME={W_THEME}")

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
    loss_theme = F.cross_entropy(theme_logits, theme)
    loss = W_PRED * loss_pred + W_RECON * loss_recon + W_DIFF * loss_diff + W_THEME * loss_theme
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % EVAL_EVERY == 0 or step == STEPS - 1:
        model.eval()
        with torch.no_grad():
            vp, vs, vt = get_batch(val_items, B)
            vlogits, vz, vrecon, vtheme_logits = model(vp, vs)
            vloss_pred = F.cross_entropy(vlogits[:, :vs.size(1)].reshape(-1, V), vs.reshape(-1))
            vloss_recon = F.cross_entropy(vrecon.reshape(-1, V), vp.reshape(-1))
            vloss_theme = F.cross_entropy(vtheme_logits, vt)
            vtheme_acc = (vtheme_logits.argmax(-1) == vt).float().mean().item()
        model.train()
        log.append((step, loss_pred.item(), vloss_pred.item(), vloss_recon.item(),
                    loss_diff.item(), loss_theme.item(), vtheme_acc))
        print(f"  step {step:4d} | pred {loss_pred.item():.3f} | val_pred {vloss_pred.item():.3f} "
              f"| val_theme {vloss_theme.item():.3f} | val_theme_acc {vtheme_acc:.3f} "
              f"| diff {loss_diff.item():.3f} | {time.time()-t0:.0f}s")

# ===== 主题分类器准确率详细评估 =====
print("\n=== 主题分类器在 train+val 上的详细指标 ===")
model.eval()
all_z, all_theme = [], []
with torch.no_grad():
    # 从 train+val 都采, 确保每主题都有
    eval_items = train_items + val_items
    for _ in range(20):
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

# ===== 可控性测试: z 沿主题梯度方向编辑 =====
print("\n=== 可控性测试: 沿主题梯度方向编辑 z, 看主题是否切换 ===")

def edit_z_for_theme(model, z, target_theme, n_steps=10, lr=0.5):
    """沿主题梯度方向 (最大化 target theme 概率) 走 n_steps, 编辑 z."""
    z = z.clone().detach().requires_grad_(True)
    target = torch.tensor([target_theme], device=DEVICE, dtype=torch.long)
    for _ in range(n_steps):
        theme_logits = model.theme_classifier(z)
        loss = F.cross_entropy(theme_logits, target)
        grad = torch.autograd.grad(loss, z)[0]
        z = (z - lr * grad).detach().requires_grad_(True)
    return z.detach()

# 收集 val 中每个主题的"代表 z" (均值)
z_means = {}
for t_id, t_name in enumerate(THEMES):
    mask = all_theme == t_id
    z_means[t_name] = all_z[mask].mean(axis=0)
print(f"  z 范数: UE_CPP={np.linalg.norm(z_means['UE_CPP']):.2f}, JS_REACT={np.linalg.norm(z_means['JS_REACT']):.2f}")

# 用 v9 一样方式生成 30 个 z 样本, 编辑到另一主题, 测主题是否切换
print("\n  [Test 1] 从 UE_CPP 的 z 出发, 编辑到 JS_REACT")
n_test = 20
correct = 0
z_src = torch.tensor(z_means["UE_CPP"], device=DEVICE, dtype=torch.float32).unsqueeze(0)
print(f"    起始 theme 预测: {model.theme_classifier(z_src).argmax(-1).item()} (期望 0=UE_CPP)")
for trial in range(n_test):
    torch.manual_seed(trial)
    out = model.gen("def ", n=80, z_override=z_src, use_real_starter=False, t=0.8)
    prefix_text = out[4:50]  # 取生成的 prefix 段
    # 用 v9 训练时的 vocab 直接 encode, 再用 z_enc 取 z
    prefix_ids = torch.tensor([stoi[c] for c in prefix_text], device=DEVICE, dtype=torch.long).unsqueeze(0)
    if prefix_ids.size(1) < T_HALF:
        prefix_ids = F.pad(prefix_ids, (0, T_HALF - prefix_ids.size(1)), value=PAD_ID)
    z_test = model.encode(prefix_ids[:, :T_HALF])
    pred_theme = model.theme_classifier(z_test).argmax(-1).item()
    if pred_theme == 0:
        correct += 1
print(f"    起始 z 主题预测准确率 (UE_CPP): {correct}/{n_test} = {correct/n_test:.2f}")

correct = 0
z_target = torch.tensor(z_means["JS_REACT"], device=DEVICE, dtype=torch.float32).unsqueeze(0)
z_edited = edit_z_for_theme(model, z_src, target_theme=1, n_steps=10, lr=0.5)
print(f"    编辑后 theme 预测: {model.theme_classifier(z_edited).argmax(-1).item()} (期望 1=JS_REACT)")
for trial in range(n_test):
    torch.manual_seed(trial)
    out = model.gen("def ", n=80, z_override=z_edited, use_real_starter=False, t=0.8)
    prefix_text = out[4:50]
    prefix_ids = torch.tensor([stoi[c] for c in prefix_text], device=DEVICE, dtype=torch.long).unsqueeze(0)
    if prefix_ids.size(1) < T_HALF:
        prefix_ids = F.pad(prefix_ids, (0, T_HALF - prefix_ids.size(1)), value=PAD_ID)
    z_test = model.encode(prefix_ids[:, :T_HALF])
    pred_theme = model.theme_classifier(z_test).argmax(-1).item()
    if pred_theme == 1:
        correct += 1
print(f"    编辑后 z 主题预测准确率 (JS_REACT): {correct}/{n_test} = {correct/n_test:.2f}")

# 抽样显示
print("\n  [样例] 起始 z (UE_CPP) vs 编辑 z (→JS_REACT) 生成对比:")
torch.manual_seed(0)
out1 = model.gen("def ", n=100, z_override=z_src, use_real_starter=False, t=0.8)
torch.manual_seed(0)
out2 = model.gen("def ", n=100, z_override=z_edited, use_real_starter=False, t=0.8)
def safe(s): return ''.join(c if ord(c) < 128 else '?' for c in s)
print(f"    UE_CPP z: {safe(out1[4:100])}")
print(f"    →JS_REACT: {safe(out2[4:100])}")

# 反向也测一下
print("\n  [Test 2] 从 JS_REACT 的 z 出发, 编辑到 UE_CPP")
z_src2 = torch.tensor(z_means["JS_REACT"], device=DEVICE, dtype=torch.float32).unsqueeze(0)
z_edited2 = edit_z_for_theme(model, z_src2, target_theme=0, n_steps=10, lr=0.5)
print(f"    起始 theme 预测: {model.theme_classifier(z_src2).argmax(-1).item()} (期望 1)")
print(f"    编辑后 theme 预测: {model.theme_classifier(z_edited2).argmax(-1).item()} (期望 0)")
torch.manual_seed(0)
out1 = model.gen("def ", n=100, z_override=z_src2, use_real_starter=False, t=0.8)
torch.manual_seed(0)
out2 = model.gen("def ", n=100, z_override=z_edited2, use_real_starter=False, t=0.8)
print(f"    JS_REACT z: {safe(out1[4:100])}")
print(f"    →UE_CPP:    {safe(out2[4:100])}")

# 保存
SAVE_PATH = "crystalllm/proto_v14_controllable_model.pt"
torch.save({"model_state_dict": model.state_dict(),
            "config": {"V": V, "T": T, "D_Z": D_Z, "T_HALF": T_HALF,
                       "N_LAYER": N_LAYER, "N_HEAD": N_HEAD, "N_EMBD": N_EMBD,
                       "N_THEMES": N_THEMES, "THEMES": THEMES,
                       "z_means": {k: v.tolist() for k, v in z_means.items()}}},
           SAVE_PATH)
print(f"\nModel saved: {SAVE_PATH}")

# 训练日志
out_json = {
    "log": log,
    "val_theme_acc": float(acc),
    "per_theme_acc": per_theme,
    "edit_UE_to_JS_acc": correct / n_test,
    "config": {"STEPS": STEPS, "W_THEME": W_THEME, "N_THEMES": N_THEMES}
}
with open("crystalllm/v14_train_log.json", "w", encoding="utf-8") as f:
    json.dump(out_json, f, indent=2, ensure_ascii=False)
print(f"Log saved: crystalllm/v14_train_log.json")
print(f"\n=== v14 训练完成 ({time.time()-t0:.0f}s) ===")
