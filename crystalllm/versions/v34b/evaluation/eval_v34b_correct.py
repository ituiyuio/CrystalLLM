# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
eval_v34b_correct.py — 用**正确的 SpS 接受率逻辑**重测 v34b

关键修正:
- 不抽查, 全部位置 AR 验证 (因为只有 1 backbone, AR forward 与 draft 同一模型)
- 接受率 = AR 同意 draft 的连续位置 / K
- 第一个不一致就 break, 用 AR 修正
"""
import json, sys, io, os, time
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); np.random.seed(42)


def P(*a, **kw):
    print(*a, **kw); sys.stdout.flush()


from v34a_model import SharedBackbone, ARHead, DHead

P("=== v34b 正确接受率测试 ===")
DATA = Path("data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1)

# 加载 v34b 模型
ckpt = torch.load("v34b_shared_backbone.pt", map_location="cuda", weights_only=False)
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
P(f"v34b loaded")


@torch.no_grad()
def diffusion_draft(z, K=8, n_ode_steps=8):
    """扩散 ODE 生成 K 个 token 草稿"""
    noisy_emb = torch.randn(1, K, cfg["N_EMBD"], device="cuda")
    dt = 1.0 / n_ode_steps
    for k in range(n_ode_steps):
        t = torch.full((1,), k * dt, device="cuda")
        hidden = backbone.forward_emb(noisy_emb, z, t)
        v = dh(hidden)[:, :, 0, :]
        noisy_emb = noisy_emb + dt * v
    # 最近邻 token
    tok_emb_w = backbone.tok_emb.weight
    flat = noisy_emb.view(-1, cfg["N_EMBD"])
    sim = F.cosine_similarity(flat.unsqueeze(1), tok_emb_w.unsqueeze(0), dim=-1)
    return sim.argmax(dim=-1).view(1, K)[0]  # (K,)


@torch.no_grad()
def ar_verify(z, prefix_tokens, draft_tokens):
    """AR head 验证 draft, 返回 (accepted_count, ar_top1_per_pos)"""
    # 1 次 forward, AR 看到完整 draft
    full_tokens = torch.cat([prefix_tokens.view(1, -1), draft_tokens.view(1, -1)], dim=1)
    hidden = backbone(full_tokens, z, t=None)
    logits = ar(hidden[:, -len(draft_tokens):])  # (1, K, V)
    ar_top1 = logits[0].argmax(dim=-1)  # (K,)
    # 接受率 = 第一个不一致位置之前的连续一致数
    accepted = 0
    for j in range(len(draft_tokens)):
        if ar_top1[j].item() == draft_tokens[j].item():
            accepted += 1
        else:
            break
    return accepted, ar_top1


@torch.no_grad()
def generate(z, max_tokens=100, K=8):
    """正确 SpS: AR 逐位置验证, 接受 draft 直到第一个不一致"""
    cur = torch.tensor([BOS_ID], device="cuda", dtype=torch.long)
    n_rounds = 0
    n_drafted = 0
    n_accepted = 0
    generated_text = []

    while len(cur) - 1 < max_tokens:
        n_rounds += 1
        draft = diffusion_draft(z, K=K)
        accepted, ar_top1 = ar_verify(z, cur, draft)
        # 接受 accepted 个 draft token
        for j in range(accepted):
            cur = torch.cat([cur, draft[j].view(1)])
            generated_text.append(int(draft[j].item()))
        # 如果有修正位置, 用 AR top-1 替代
        if accepted < K:
            cur = torch.cat([cur, ar_top1[accepted].view(1)])
            generated_text.append(int(ar_top1[accepted].item()))
        n_drafted += K
        n_accepted += accepted
        if len(cur) - 1 > max_tokens + K:
            break

    return cur[1:max_tokens + 1], n_rounds, n_drafted, n_accepted, generated_text[:max_tokens]


# ===== 测试 =====
P("\n=== 速度 + 接受率 Benchmark ===")
data = np.load("cached_v34b_outputs.npz")
z_test = torch.tensor(data["z"][0], dtype=torch.float32).unsqueeze(0).to("cuda")


def gen_fn():
    return generate(z_test, max_tokens=100, K=8)


# warmup
for _ in range(2): gen_fn()
torch.cuda.synchronize()
times = []
results_list = []
for _ in range(10):
    torch.cuda.synchronize()
    t0 = time.time()
    cur, n_rounds, n_drafted, n_accepted, gen_text = gen_fn()
    torch.cuda.synchronize()
    times.append((time.time() - t0) * 1000)
    results_list.append((n_rounds, n_drafted, n_accepted))

mean_ms = float(np.mean(times))
# 平均接受率 (跨多次 run)
avg_accept_rate = np.mean([a/max(d,1) for _, d, a in results_list])
avg_rounds = np.mean([r for r, _, _ in results_list])
P(f"  速度 (100 tokens): {mean_ms:.1f} ms")
P(f"  接受率 (SpS 真实): {avg_accept_rate*100:.1f}%")
P(f"  rounds (avg): {avg_rounds:.1f}")

# PPL on val
P("\n=== 计算 PPL ===")
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
P(f"  PPL = {ppl:.4f}")

# 生成样本
P("\n=== 生成样本 ===")
text = "".join([itos.get(t, "?") for t in gen_text[:80]])
P(f"  {repr(text)}")

# 结果
results = {
    "speed_ms": mean_ms,
    "ppl": float(ppl),
    "acceptance_rate_spS": float(avg_accept_rate),
    "n_rounds_avg": float(avg_rounds),
    "v31_baseline_ms": 206,
    "v31_accept": 0.955,
    "v31_ppl": 2.39,
}
results["speed_pass"] = bool(mean_ms < 150)
results["accept_pass"] = bool(avg_accept_rate > 0.955)
results["ppl_pass"] = bool(ppl <= 2.39)
results["all_pass"] = bool(results["speed_pass"] and results["accept_pass"] and results["ppl_pass"])

P(f"\n=== 结果 (正确 SpS 接受率) ===")
P(f"  速度: {'PASS' if results['speed_pass'] else 'FAIL'} ({mean_ms:.1f}ms)")
P(f"  PPL: {'PASS' if results['ppl_pass'] else 'FAIL'} ({ppl:.4f})")
P(f"  接受率 (SpS): {'PASS' if results['accept_pass'] else 'FAIL'} ({avg_accept_rate*100:.1f}%)")
P(f"  总评: {'✅ ALL PASS' if results['all_pass'] else '❌ FAILED'}")

with open("v34b_correct_e2e.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
P(f"\nSaved: v34b_correct_e2e.json")