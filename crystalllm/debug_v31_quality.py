"""Reproduce v31 eval results directly"""
import json, sys, io, os, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"
torch.manual_seed(42); np.random.seed(42)


def P(*a, **kw):
    print(*a, **kw); sys.stdout.flush()


DATA = Path("data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1)

# Load verifier
ckpt_v = torch.load("v28_5_decoder.pt", map_location="cuda", weights_only=False)
vcfg = ckpt_v["config"]
T_v28 = vcfg["T"]; D_Z = vcfg["D_Z"]
DEC_LAYER = vcfg["DEC_LAYER"]; DEC_HEAD = vcfg["DEC_HEAD"]; DEC_EMBD = vcfg["DEC_EMBD"]


class BlockCausal(nn.Module):
    def __init__(self, N_EMBD, N_HEAD):
        super().__init__()
        self.nh = N_HEAD; self.head_dim = N_EMBD // N_HEAD
        self.ln1 = nn.LayerNorm(N_EMBD); self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(), nn.Linear(4 * N_EMBD, N_EMBD))
    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x); qkv = self.qkv(h).reshape(B, T, 3, self.nh, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).contiguous().view(B, T, C))
        x = x + self.mlp(self.ln2(x)); return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.z_to_emb = nn.Linear(D_Z, DEC_EMBD)
        self.tok = nn.Embedding(V, DEC_EMBD)
        self.pos = nn.Embedding(T_v28 + 2, DEC_EMBD)
        self.blocks = nn.ModuleList([BlockCausal(DEC_EMBD, DEC_HEAD) for _ in range(DEC_LAYER)])
        self.ln_f = nn.LayerNorm(DEC_EMBD)
        self.head = nn.Linear(DEC_EMBD, V, bias=False)
        self.tok.weight = self.head.weight
    def forward(self, z, x):
        B, T = x.shape
        z_emb = self.z_to_emb(z).unsqueeze(1)
        bos_emb = self.tok(torch.tensor([BOS_ID], device=x.device)).expand(B, 1, -1)
        x_emb = self.tok(x)
        inp = torch.cat([z_emb, bos_emb, x_emb], dim=1)
        inp = inp + self.pos(torch.arange(T + 2, device=x.device))
        for b in self.blocks: inp = b(inp)
        return self.head(self.ln_f(inp))[:, 1:T + 1]


verifier = Decoder().to("cuda")
verifier.load_state_dict(ckpt_v["decoder"])
verifier.eval()

# Load v31 drafter
ckpt_d = torch.load("v31_diff_drafter.pt", map_location="cuda", weights_only=False)
dcfg = ckpt_d["config"]
N = dcfg["N"]; D_EMB = dcfg["D_EMB"]; D_HID = dcfg["D_HID"]; D_T = dcfg["D_T"]
N_LAYER = dcfg["N_LAYER"]


class ResBlockV2(nn.Module):
    def __init__(self, D_HID):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_HID); self.fc1 = nn.Linear(D_HID, D_HID)
        self.ln2 = nn.LayerNorm(D_HID); self.fc2 = nn.Linear(D_HID, D_HID)
        self.film = nn.Linear(D_HID, 2 * D_HID)
    def forward(self, h, t_emb):
        gamma, beta = self.film(t_emb).unsqueeze(1).chunk(2, dim=-1)
        h_res = h
        h = self.fc1(F.gelu(self.ln1(h))) * (1 + gamma) + beta
        h = self.fc2(F.gelu(self.ln2(h)))
        return h_res + h


class TokenDiffusionDrafter(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_emb = nn.Embedding(N, D_EMB)
        self.z_proj = nn.Linear(D_Z, D_HID)
        self.t_proj = nn.Linear(D_T, D_HID)
        self.in_proj = nn.Linear(D_HID + 2 * D_EMB, D_HID)
        self.blocks = nn.ModuleList([ResBlockV2(D_HID) for _ in range(N_LAYER)])
        self.ln = nn.LayerNorm(D_HID)
        self.out = nn.Linear(D_HID, D_EMB)
    def forward(self, z, t, noise):
        B, N_, D_ = noise.shape
        z_cond = self.z_proj(z)
        half = D_T // 2
        import math
        freqs = torch.exp(-torch.arange(half, device=z.device, dtype=torch.float32) * (math.log(10000.0) / half))
        args = t.float()[:, None] * freqs[None]
        t_emb_raw = torch.cat([torch.sin(args), torch.cos(args)], dim=-1) * (D_T ** 0.5)
        t_emb = self.t_proj(t_emb_raw)
        cond = (z_cond + t_emb).unsqueeze(1).expand(-1, N_, -1)
        pos = self.pos_emb(torch.arange(N_, device=noise.device)).unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([cond, pos, noise], dim=-1)
        x = self.in_proj(x)
        for blk in self.blocks: x = blk(x, z_cond + t_emb)
        return self.out(self.ln(x))


drafter = TokenDiffusionDrafter().to("cuda")
drafter.load_state_dict(ckpt_d["model"])
drafter.eval()
tok_emb = nn.Embedding(V, D_EMB).to("cuda")
tok_emb.load_state_dict(ckpt_d["tok_emb"])
tok_emb.eval()

# Load prior
prior_ckpt = torch.load("v24_diffusion_prior.pt", map_location="cuda", weights_only=False)
pcfg = prior_ckpt["config"]
D_ZP = pcfg["D_Z"]; D_HIDP = pcfg["D_HID"]; N_LAYER_P = pcfg["N_LAYER"]
N_SAMPLE_STEPS = pcfg["N_SAMPLE_STEPS"]


class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        import math
        half = self.dim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device, dtype=torch.float32) * (math.log(10000.0) / half))
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1) * (self.dim ** 0.5)


class ResBlockP(nn.Module):
    def __init__(self, D_HID):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_HID); self.fc1 = nn.Linear(D_HID, D_HID)
        self.ln2 = nn.LayerNorm(D_HID); self.fc2 = nn.Linear(D_HID, D_HID)
        self.film = nn.Linear(D_HID, 2 * D_HID)
    def forward(self, h, t_emb):
        gamma, beta = self.film(t_emb).chunk(2, dim=-1)
        h_res = h
        h = self.fc1(F.gelu(self.ln1(h))) * (1 + gamma) + beta
        h = self.fc2(F.gelu(self.ln2(h)))
        return h_res + h


class DiffusionPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_emb = SinusoidalTimeEmbed(D_HIDP)
        self.in_proj = nn.Linear(D_ZP, D_HIDP)
        self.blocks = nn.ModuleList([ResBlockP(D_HIDP) for _ in range(N_LAYER_P)])
        self.ln = nn.LayerNorm(D_HIDP)
        self.out = nn.Linear(D_HIDP, D_ZP)
    def forward(self, z_t, t):
        h = self.in_proj(z_t)
        t_emb = self.t_emb(t)
        for blk in self.blocks: h = blk(h, t_emb)
        return self.out(self.ln(h))


prior = DiffusionPrior().to("cuda")
prior.load_state_dict(prior_ckpt["model"])
prior.eval()


@torch.no_grad()
def sample_prior(n):
    z = torch.randn(n, D_ZP, device="cuda")
    dt = 1.0 / N_SAMPLE_STEPS
    for k in range(1, N_SAMPLE_STEPS + 1):
        t_val = (k - 1) * dt
        t = torch.full((n,), t_val, device="cuda")
        v = prior(z, t)
        z = z + dt * v
    return z


@torch.no_grad()
def sample_K_tokens(z, K=8, n_steps=5):
    x_t = torch.randn(1, K, D_EMB, device="cuda")
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t_val = k * dt
        t = torch.full((1,), t_val, device="cuda")
        v = drafter(z, t, x_t)
        x_t = x_t + dt * v
    logits = F.linear(x_t, tok_emb.weight)
    return logits.argmax(dim=-1)[0]


@torch.no_grad()
def gen_v31_sps_full(n_ar=100, K=8):
    z = sample_prior(1)
    cur = [BOS_ID]
    n_rounds = 0
    n_drafted = 0
    n_accepted = 0
    round_log = []  # 记录每轮的 draft 和 accepted 数

    while len(cur) - 1 < n_ar:
        n_rounds += 1
        draft = sample_K_tokens(z, K=K)
        x = draft.unsqueeze(0)
        v_logits = verifier(z, x)
        v_tokens = v_logits.argmax(dim=-1)[0]

        n_acc = 0
        for j in range(K):
            n_drafted += 1
            if draft[j].item() == v_tokens[j].item():
                cur.append(draft[j].item())
                n_acc += 1
                n_accepted += 1
            else:
                cur.append(v_tokens[j].item())
                break
        round_log.append({
            "round": n_rounds,
            "draft": draft.tolist(),
            "verifier": v_tokens.tolist(),
            "accepted": n_acc,
            "draft_text": "".join([itos.get(int(t), "?") for t in draft]),
            "verifier_text": "".join([itos.get(int(t), "?") for t in v_tokens])
        })
        if n_rounds <= 3 or n_rounds % 5 == 0:
            P(f"  round {n_rounds}: draft=\"{round_log[-1]['draft_text']}\", "
              f"verifier=\"{round_log[-1]['verifier_text']}\", accepted={n_acc}/{K}")

    return cur, n_rounds, n_drafted, n_accepted, round_log


P("=== v31 SpS - 详细 trace ===")
cur, n_rounds, n_drafted, n_accepted, round_log = gen_v31_sps_full(n_ar=100, K=8)
P(f"\n总接受: {n_accepted}/{n_drafted} = {n_accepted/n_drafted*100:.1f}%")
P(f"Rounds: {n_rounds}")
text = "".join([itos.get(t, "?") for t in cur[:80]])
P(f"生成: {repr(text)}")

# 测试: 给 verifier 一个有意义 prefix, 看它续写什么
P("\n=== Verifier 续写 (有 prefix) ===")
z_test = sample_prior(1)
prefix_text = "user: Build a scalable web app using React and Node"
prefix_ids = [BOS_ID]
for ch in prefix_text:
    prefix_ids.append(stoi.get(ch, 0))

with torch.no_grad():
    for n_test in [1, 5, 20]:
        # 给 prefix, 让 verifier 续写 n_test 个 token
        cur_test = list(prefix_ids)
        for _ in range(n_test):
            x = torch.tensor([cur_test], dtype=torch.long, device="cuda")
            logits = verifier(z_test, x)
            nxt = logits[0, -1].argmax().item()
            cur_test.append(nxt)
        new_text = "".join([itos.get(t, "?") for t in cur_test[len(prefix_ids):]])
        full_text = "".join([itos.get(t, "?") for t in cur_test])
        P(f"  prefix \"{prefix_text[:30]}...\" + {n_test} tokens: \"{full_text}\"")

# 测试: 真实 AR 生成 (v28.5 baseline)
P("\n=== v28.5 verifier AR baseline (从头生成 50 token) ===")
cur_v = [BOS_ID]
with torch.no_grad():
    for _ in range(50):
        x = torch.tensor([cur_v[-1:]], dtype=torch.long, device="cuda")
        logits = verifier(z_test, x)
        nxt = logits[0, -1].argmax().item()
        cur_v.append(nxt)
text_v = "".join([itos.get(t, "?") for t in cur_v])
P(f"  AR baseline: {repr(text_v)}")

# 关键测试: verifier 看到 drafter 草稿后, 每个位置预测什么?
P("\n=== verifier 看 drafter 草稿后逐位置预测 ===")
# 用 v35 drafter 的真实草稿 (有意义的字母)
with torch.no_grad():
    # 取一个训练数据 z
    data = np.load("cached_v35_outputs.npz")
    z_train = torch.tensor(data["z"][0], dtype=torch.float32).unsqueeze(0).to("cuda")
    # v35 drafter 5步生成 8 tokens
    x_t = torch.randn(1, 8, D_EMB, device="cuda")
    dt = 1.0 / 5
    for k in range(5):
        tt = torch.full((1,), k * dt, device="cuda")
        # 用 v35 drafter
        ckpt35 = torch.load("v35_diff_drafter.pt", map_location="cuda", weights_only=False)
        d35 = TokenDiffusionDrafter().to("cuda")
        d35.load_state_dict(ckpt35["model"])
        te35 = nn.Embedding(V, D_EMB).to("cuda")
        te35.load_state_dict(ckpt35["tok_emb"])
        v = d35(z_train, tt, x_t)
        x_t = x_t + dt * v
    draft_logits = F.linear(x_t, te35.weight)
    draft = draft_logits.argmax(-1)[0]
    draft_text = "".join([itos.get(int(t), "?") for t in draft])
    P(f"  v35 drafter output: \"{draft_text}\"")

    # verifier 看这个 draft, 每个位置预测什么?
    x = draft.unsqueeze(0)
    v_logits = verifier(z_train, x)
    v_pred = v_logits.argmax(-1)[0]
    v_text = "".join([itos.get(int(t), "?") for t in v_pred])
    P(f"  verifier 看 draft 后逐位置预测: \"{v_text}\"")

    # 位置 0: verifier 看到 draft[0], 它想要什么?
    v_pos0 = v_logits[0, 0].argmax().item()
    v_pos0_text = itos.get(v_pos0, "?")
    P(f"  verifier 位置 0 想要: {repr(v_pos0_text)} (draft[0]={repr(itos.get(int(draft[0]), '?'))})")

    # 比较每一位置
    for i in range(8):
        match = "✓" if int(v_pred[i]) == int(draft[i]) else "✗"
        P(f"    pos {i}: draft={repr(itos.get(int(draft[i]), '?'))}, verifier={repr(itos.get(int(v_pred[i]), '?'))} {match}")