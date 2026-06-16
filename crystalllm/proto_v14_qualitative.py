"""
proto_v14_qualitative.py — v14 主题切换定性测试

不再单独训练, 直接加载 v14_controllable_model.pt, 测多个 seeds 下主题切换.
"""
import json, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
import numpy as np
import random

torch.manual_seed(42); random.seed(42); np.random.seed(42)

DATA = Path("crystalllm/data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]
itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]

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
            nn.Linear(D_Z, 2)
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
    @torch.no_grad()
    def gen(s, seed, n=200, t=0.7, z_override=None, use_real_starter=False):
        s.eval()
        seed_ids = [stoi[c] for c in seed]
        if z_override is None:
            ids = torch.tensor([seed_ids[:T_HALF]], device=DEVICE, dtype=torch.long)
            if ids.size(1) < T_HALF:
                ids = F.pad(ids, (0, T_HALF - ids.size(1)), value=PAD_ID)
            z = s.encode(ids)
        else:
            z = z_override
        if use_real_starter:
            n_seed = len(seed_ids)
            # 找一段与 seed 主题相符的 starter — 这里就用随机
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

# Load
print("[load] v14")
ck = torch.load("crystalllm/proto_v14_controllable_model.pt", map_location=DEVICE, weights_only=False)
model = ControllableHybrid().to(DEVICE)
model.load_state_dict(ck["model_state_dict"])
model.eval()

# 收集 z 计算 z_means
df = pd.read_parquet(DATA / "subset_2000.parquet")
df = df[df["project"].isin({"D--UnrealEngine-CODEO", "D--long-running-harness"})].reset_index(drop=True)
all_text = "\n".join(df["text"].tolist())
THEME_TO_ID = {"D--UnrealEngine-CODEO": 0, "D--long-running-harness": 1}
df["theme_id"] = df["project"].map(THEME_TO_ID)

print("[collect z]")
items = list(zip(df["text"].tolist(), df["theme_id"].tolist()))
all_z, all_theme = [], []
with torch.no_grad():
    np.random.seed(0)
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
z_means = {0: torch.tensor(all_z[all_theme==0].mean(0), device=DEVICE, dtype=torch.float32).unsqueeze(0),
           1: torch.tensor(all_z[all_theme==1].mean(0), device=DEVICE, dtype=torch.float32).unsqueeze(0)}

def edit_z(z, target, n_steps=30, lr=2.0):
    z = z.clone().detach().requires_grad_(True)
    target_t = torch.tensor([target], device=DEVICE, dtype=torch.long)
    for _ in range(n_steps):
        loss = F.cross_entropy(model.theme_classifier(z), target_t)
        grad = torch.autograd.grad(loss, z)[0]
        z = (z - lr * grad).detach().requires_grad_(True)
    return z.detach()

def safe(s): return ''.join(c if ord(c) < 128 else '?' for c in s)

# 多 seed × 多 trial 的定性对比
# 关键: 不用 z_means 直接当 z_override (会主导整个生成)
# 改用: encode(seed_prefix) + 沿主题方向小步编辑
print("\n=== 定性对比: 真实 prefix z (起始) vs prefix z+主题梯度 (编辑) ===")
seeds = ["def ", "void ", "class ", "const ", "function "]
n_trials = 4
results = {}
for seed in seeds:
    print(f"\n  seed = {seed!r}")
    # encode 一次, 起始 z 来自真实 prefix
    seed_ids = [stoi[c] for c in seed]
    prefix_ids = torch.tensor([seed_ids[:T_HALF]], device=DEVICE, dtype=torch.long)
    if prefix_ids.size(1) < T_HALF:
        prefix_ids = F.pad(prefix_ids, (0, T_HALF - prefix_ids.size(1)), value=PAD_ID)
    with torch.no_grad():
        z_real = model.encode(prefix_ids[:, :T_HALF])
    for trial in range(n_trials):
        torch.manual_seed(trial * 7)
        # 起始: 真实 z
        z_src = z_real.clone()
        # 编辑: 沿主题梯度走小步 (避免破坏 z 结构)
        z_edited = edit_z(z_src, target=1, n_steps=10, lr=0.1)
        torch.manual_seed(trial * 7)
        out_src = model.gen(seed, n=120, z_override=z_src, use_real_starter=True, t=0.7)
        torch.manual_seed(trial * 7)
        out_edit = model.gen(seed, n=120, z_override=z_edited, use_real_starter=True, t=0.7)
        # 统计
        src_text = out_src[len(seed):120]
        edit_text = out_edit[len(seed):120]
        # 主题指标
        def stats(text):
            return {
                "brace_open_frac": text.count("{") / max(len(text), 1),
                "brace_close_frac": text.count("}") / max(len(text), 1),
                "arrow_frac": text.count("->") / max(len(text), 1),
                "semi_frac": text.count(";") / max(len(text), 1),
                "const_frac": sum(1 for i in range(len(text)-5) if text[i:i+5]=="const") / max(len(text), 1),
                "void_frac": sum(1 for i in range(len(text)-4) if text[i:i+4]=="void") / max(len(text), 1),
                "function_frac": sum(1 for i in range(len(text)-8) if text[i:i+8]=="function") / max(len(text), 1),
            }
        s_src = stats(src_text); s_edit = stats(edit_text)
        results.setdefault(seed, {"src": [], "edit": [], "src_stats": [], "edit_stats": []})
        results[seed]["src"].append(src_text)
        results[seed]["edit"].append(edit_text)
        results[seed]["src_stats"].append(s_src)
        results[seed]["edit_stats"].append(s_edit)
        # 打印前 2 个
        if trial < 2:
            print(f"    trial {trial}:")
            print(f"      src:    {safe(src_text[:80])}")
            print(f"      edited: {safe(edit_text[:80])}")

# 汇总指标
print("\n=== 主题特征变化汇总 (UE_CPP→JS_REACT 编辑) ===")
metrics = ["brace_open_frac", "brace_close_frac", "arrow_frac", "semi_frac", "const_frac", "function_frac"]
for seed in seeds:
    print(f"\n  seed={seed!r}:")
    for m in metrics:
        v_src = np.mean([s[m] for s in results[seed]["src_stats"]])
        v_edit = np.mean([s[m] for s in results[seed]["edit_stats"]])
        print(f"    {m:20s}: src={v_src:.4f} -> edit={v_edit:.4f} (delta={v_edit-v_src:+.4f})")

# 保存
out = {seed: {"src_samples": results[seed]["src"][:2], "edit_samples": results[seed]["edit"][:2]} for seed in seeds}
with open("crystalllm/v14_qualitative.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\n[done] saved to crystalllm/v14_qualitative.json")
