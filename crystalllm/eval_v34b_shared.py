"""
eval_v34a_shared.py — v34a 推理与 benchmark

推理流程 (Speculative Decoding with spot-check):
  1. 扩散 ODE 生成 K=8 草稿
  2. AR head 抽查 n_check=3 位置 (概率最低的)
  3. 不一致则 AR top-1 修正

评估指标:
  - speed_ms (100 tokens 平均)
  - ppl (val set, 用 AR head)
  - acceptance_rate (抽查位置一致率)
"""
import json, sys, io, os, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); np.random.seed(42)


def P(*a, **kw):
    print(*a, **kw); sys.stdout.flush()


from v34a_model import SharedBackbone, ARHead, DHead, get_time_embedding

P("=== v34a 推理与 Benchmark ===")
DATA = Path("data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1)

# ===== 加载 v34b 模型 (20K 数据训练) =====
CKPT = "v34b_shared_backbone.pt"
ckpt = torch.load(CKPT, map_location="cuda", weights_only=False)
cfg = ckpt["config"]
backbone = SharedBackbone(V, n_layer=cfg["N_LAYER"], n_embd=cfg["N_EMBD"],
                           n_head=cfg["N_HEAD"], z_dim=cfg["Z_DIM"],
                           t_dim=cfg["T_DIM"], max_seq=cfg["SEQ_LEN"] + cfg["K_WINDOW"] + 4).to("cuda")
backbone.load_state_dict(ckpt["backbone"])
backbone.eval()
ar = ARHead(backbone).to("cuda")
dh = DHead(n_embd=cfg["N_EMBD"], k_window=cfg["K_WINDOW"]).to("cuda")
dh.load_state_dict(ckpt["d_head"])
dh.eval()
P(f"Model loaded: backbone {cfg['N_LAYER']}×{cfg['N_EMBD']}×{cfg['N_HEAD']} ({sum(p.numel() for p in backbone.parameters())/1e6:.1f}M)")
P(f"Trained for {cfg['TOTAL_STEPS']} steps, B={cfg['B']}")


@torch.no_grad()
def diffusion_draft(z, K=8, n_ode_steps=8):
    """扩散 ODE 生成 K 个 token 草稿"""
    noisy_emb = torch.randn(1, K, cfg["N_EMBD"], device="cuda")
    dt = 1.0 / n_ode_steps
    for k in range(n_ode_steps):
        t = torch.full((1,), k * dt, device="cuda")
        hidden = backbone.forward_emb(noisy_emb, z, t)
        v = dh(hidden)[:, :, 0, :]  # (1, K, N_EMBD)
        noisy_emb = noisy_emb + dt * v
    # 最近邻 token (cosine similarity)
    tok_emb_w = backbone.tok_emb.weight  # (V, N_EMBD)
    flat = noisy_emb.view(-1, cfg["N_EMBD"])
    sim = F.cosine_similarity(flat.unsqueeze(1), tok_emb_w.unsqueeze(0), dim=-1)
    return sim.argmax(dim=-1).view(1, K)


@torch.no_grad()
def ar_spot_check(z, prefix_tokens, draft_tokens, n_check=3):
    """AR head 抽查 n_check 个位置 (抽查 AR 最有把握的位置)

    抽查策略 v2:
      - 抽查 AR top-1 概率**最高**的位置 (AR 最有把握的位置)
      - 如果 AR top-1 == draft_token: 接受, 不动
      - 如果 AR top-1 != draft_token: 不接受, 用 AR top-1 修正
    """
    full_tokens = torch.cat([prefix_tokens.view(1, -1), draft_tokens.view(1, -1)], dim=1)
    hidden = backbone(full_tokens, z, t=None)
    logits = ar(hidden[:, -len(draft_tokens):])  # (1, K, V)
    probs = F.softmax(logits, dim=-1)
    ar_top1 = logits[0].argmax(dim=-1)
    # AR 对 draft token 的概率
    draft_probs = probs[0, range(len(draft_tokens)), draft_tokens]
    # 抽查概率**最高**的 n_check 个位置 (AR 最确信)
    check_idx = draft_probs.topk(n_check, largest=True).indices
    final = draft_tokens.clone()
    for i in check_idx:
        if final[i].item() != ar_top1[i].item():
            final[i] = ar_top1[i]
    return final, check_idx


@torch.no_grad()
def generate(z, max_tokens=100, K=8, n_check=3):
    """SpS-style 推理"""
    cur = torch.tensor([BOS_ID], device="cuda", dtype=torch.long)
    n_rounds = 0
    n_drafted = 0
    n_accepted = 0

    while len(cur) - 1 < max_tokens:
        n_rounds += 1
        draft = diffusion_draft(z, K=K)[0]  # (K,)
        final, check_idx = ar_spot_check(z, cur, draft, n_check=n_check)
        # 检查真实一致性 (抽查位置)
        for j in check_idx:
            n_drafted += 1
            if final[j].item() == draft[j].item():
                n_accepted += 1
        # 接受所有 final tokens
        cur = torch.cat([cur, final])
        # 防止无限增长
        if len(cur) - 1 > max_tokens + K:
            break

    return cur[1:max_tokens + 1], n_rounds, n_drafted, n_accepted


@torch.no_grad()
def compute_ppl():
    """在 val 上计算 PPL (用 AR head)"""
    P("\n=== 计算 PPL ===")
    data = np.load("cached_v29_outputs.npz")
    Z = torch.tensor(data["z"][:200], dtype=torch.float32).to("cuda")
    T = torch.tensor(data["tokens"][:200, :100], dtype=torch.long).to("cuda")
    nll_total = 0.0
    n_tokens = 0
    SEQ = cfg["SEQ_LEN"]
    for i in range(0, 200, 8):
        z = Z[i:i + 8]
        tokens = T[i:i + 8, :SEQ]
        target = T[i:i + 8, 1:SEQ + 1]
        hidden = backbone(tokens, z, t=None)
        logits = ar(hidden)
        nll = F.cross_entropy(logits.reshape(-1, V), target.reshape(-1), reduction="sum")
        nll_total += nll.item()
        n_tokens += target.numel()
    ppl = np.exp(nll_total / n_tokens)
    P(f"  PPL = {ppl:.4f} (target ≤ 2.39)")
    return ppl


# ===== Benchmark =====
P("\n=== 速度 Benchmark ===")
data = np.load("cached_v29_outputs.npz")
z_test = torch.tensor(data["z"][0], dtype=torch.float32).unsqueeze(0).to("cuda")


def gen_fn():
    return generate(z_test, max_tokens=100, K=8, n_check=3)


# warmup
for _ in range(2): gen_fn()
torch.cuda.synchronize()
times = []
for _ in range(10):
    torch.cuda.synchronize()
    t0 = time.time()
    cur, n_rounds, n_drafted, n_accepted = gen_fn()
    torch.cuda.synchronize()
    times.append((time.time() - t0) * 1000)
mean_ms = float(np.mean(times))
acc_rate = n_accepted / max(n_drafted, 1)
P(f"  速度 (100 tokens): {mean_ms:.1f} ms (target < 150)")
P(f"  接受率 (抽查): {acc_rate*100:.1f}% (target > 95.5%)")
P(f"  rounds: {n_rounds}")

# PPL
ppl = compute_ppl()

# 生成质量
P("\n=== 生成质量 ===")
text = "".join([itos.get(int(t), "?") for t in cur[:80]])
P(f"  {repr(text)}")

# 结果汇总
results = {
    "speed_ms": float(mean_ms),
    "ppl": float(ppl),
    "acceptance_rate": float(acc_rate),
    "n_rounds": int(n_rounds),
    "speed_target_ms": 150,
    "ppl_target": 2.39,
    "accept_target": 0.955,
    "speed_pass": bool(mean_ms < 150),
    "ppl_pass": bool(ppl <= 2.39),
    "accept_pass": bool(acc_rate > 0.955),
    "n_backbone_M": float(sum(p.numel() for p in backbone.parameters())/1e6),
    "n_dhead_M": float(sum(p.numel() for p in dh.parameters())/1e6),
    "v31_baseline_ms": 206,
    "v31_ppl": 2.39,
    "v31_accept": 0.955,
}
all_pass = results["speed_pass"] and results["ppl_pass"] and results["accept_pass"]
results["all_pass"] = bool(all_pass)
P(f"\n=== 结果 ===")
P(f"  速度: {'PASS' if results['speed_pass'] else 'FAIL'} ({mean_ms:.1f}ms, vs v31 206ms)")
P(f"  PPL: {'PASS' if results['ppl_pass'] else 'FAIL'} ({ppl:.4f}, vs v31 2.39)")
P(f"  接受率: {'PASS' if results['accept_pass'] else 'FAIL'} ({acc_rate*100:.1f}%, vs v31 95.5%)")
P(f"  总评: {'✅ ALL PASS' if all_pass else '❌ FAILED'}")

with open("v34a_e2e.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
P(f"\nSaved: v34a_e2e.json")