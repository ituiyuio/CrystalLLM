"""
proto_v14_test.py — 加载已训练的 v14, 跑可控性测试

跳过训练, 直接从 v14_controllable_model.pt 加载并评估.
"""
import json, math, time, random
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
import numpy as np

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
all_text = "\n".join(df["text"].tolist())

B, T, D_Z = 32, 256, 64
T_HALF = T // 2
N_LAYER, N_HEAD, N_EMBD = 16, 8, 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAD_ID = stoi.get(' ', 0)

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
        return s.z_enc(s.ln_f(s.blocks(h)).mean(dim=1))
    def decode(s, z, suffix):
        B_, T_s = suffix.shape
        z_emb = s.z_dec(z).unsqueeze(1)
        sfx_emb = s.tok(suffix) + s.pos(torch.arange(1, T_s+1, device=suffix.device))
        x = torch.cat([z_emb, sfx_emb], dim=1)
        return s.head(s.ln_f(s.blocks(x)))
    def forward(s, prefix, suffix):
        z = s.encode(prefix)
        logits = s.decode(z, suffix)
        recon = s.z_to_chars(z.unsqueeze(1).expand(-1, prefix.size(1), -1))
        theme_logits = s.theme_classifier(z)
        return logits, z, recon, theme_logits
    @torch.no_grad()
    def gen(s, seed, n=150, t=0.8, z_override=None, use_real_starter=False, K_diff=5, from_noise=False):
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

# 加载模型
print("[load] v14_controllable_model.pt")
ck = torch.load("crystalllm/proto_v14_controllable_model.pt", map_location=DEVICE, weights_only=False)
model = ControllableHybrid().to(DEVICE)
model.load_state_dict(ck["model_state_dict"])
model.eval()
print(f"  loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

# 在训练数据上重新收集 z, 测主题分类器 acc + 重新算 z_means
print("\n[eval] 主题分类器在 train+val 上的 acc, 并重算 z_means")
items = list(zip(df["text"].tolist(), df["theme_id"].tolist()))
np.random.seed(0)
all_z, all_theme = [], []
with torch.no_grad():
    for _ in range(40):
        ix = np.random.randint(0, len(items), B)
        fulls, themes = [], []
        for i in ix:
            text, theme = items[i]
            if len(text) < T + 2: text = text + "\n" * (T + 2 - len(text))
            start = random.randint(0, len(text) - T - 2)
            fulls.append([stoi[c] for c in text[start:start + T + 2]])
            themes.append(theme)
        full = torch.tensor(fulls, dtype=torch.long).to(DEVICE)
        theme = torch.tensor(themes, dtype=torch.long).to(DEVICE)
        vz = model.encode(full[:, :T_HALF])
        all_z.append(vz.cpu().numpy()); all_theme.append(theme.cpu().numpy())
all_z = np.concatenate(all_z); all_theme = np.concatenate(all_theme)
theme_logits = model.theme_classifier(torch.tensor(all_z, device=DEVICE))
preds = theme_logits.argmax(-1).cpu().numpy()
acc = (preds == all_theme).mean()
per_theme = {}
z_means = {}
for t_id, t_name in enumerate(THEMES):
    mask = all_theme == t_id
    if mask.sum() > 0:
        per_theme[t_name] = float((preds[mask] == t_id).mean())
        z_means[t_name] = torch.tensor(all_z[mask].mean(axis=0), device=DEVICE, dtype=torch.float32).unsqueeze(0)
    print(f"  {t_name}: n={mask.sum()}, acc={per_theme.get(t_name, 0):.3f}, |z_mean|={torch.norm(z_means[t_name]).item():.2f}")
print(f"  Overall: {acc:.3f}")

# 可控性测试: 沿 L_theme 对 z 梯度编辑 z
def edit_z_for_theme(z, target_theme, n_steps=20, lr=1.0):
    z = z.clone().detach().requires_grad_(True)
    target = torch.tensor([target_theme], device=DEVICE, dtype=torch.long)
    for _ in range(n_steps):
        theme_logits = model.theme_classifier(z)
        loss = F.cross_entropy(theme_logits, target)
        grad = torch.autograd.grad(loss, z)[0]
        z = (z - lr * grad).detach().requires_grad_(True)
    return z.detach()

def safe(s): return ''.join(c if ord(c) < 128 else '?' for c in s)

print("\n[test] 沿主题梯度编辑 z, 看生成文本是否切换主题")
n_test = 16
results = {}
for src_name, src_id in [("UE_CPP", 0), ("JS_REACT", 1)]:
    for tgt_name, tgt_id in [("UE_CPP", 0), ("JS_REACT", 1)]:
        if src_name == tgt_name: continue
        z_src = z_means[src_name].clone()
        theme_logits_start = model.theme_classifier(z_src).argmax(-1).item()
        z_edited = edit_z_for_theme(z_src, target_theme=tgt_id, n_steps=30, lr=2.0)
        theme_logits_end = model.theme_classifier(z_edited).argmax(-1).item()
        samples_src, samples_edited = [], []
        for trial in range(n_test):
            torch.manual_seed(trial)
            samples_src.append(model.gen("def ", n=80, z_override=z_src, use_real_starter=False, t=0.8))
            torch.manual_seed(trial)
            samples_edited.append(model.gen("def ", n=80, z_override=z_edited, use_real_starter=False, t=0.8))
        results[(src_name, tgt_name)] = {
            "start_pred": theme_logits_start,
            "end_pred": theme_logits_end,
            "samples_src": samples_src[:3],
            "samples_edited": samples_edited[:3]
        }
        print(f"\n  [{src_name} -> {tgt_name}] 起始预测={theme_logits_start}, 编辑后={theme_logits_end}")
        for i in range(min(2, n_test)):
            print(f"    src#{i}:    {safe(results[(src_name, tgt_name)]['samples_src'][i][4:80])}")
            print(f"    edit#{i}:  {safe(results[(src_name, tgt_name)]['samples_edited'][i][4:80])}")

# 保存
out = {
    "theme_classifier_acc_overall": float(acc),
    "per_theme_acc": per_theme,
    "edit_results": {f"{k[0]}_to_{k[1]}": {"start_pred": v["start_pred"],
                                            "end_pred": v["end_pred"],
                                            "samples_src": v["samples_src"],
                                            "samples_edited": v["samples_edited"]}
                     for k, v in results.items()}
}
with open("crystalllm/v14_test_results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\n[done] saved to crystalllm/v14_test_results.json")
