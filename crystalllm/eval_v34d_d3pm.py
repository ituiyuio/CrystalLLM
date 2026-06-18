"""
eval_v34d_d3pm.py — v34d 评测: D3PM 投机生成 + AR 验证

推理流程 (SpS-style):
  1. AR 续写 prefix (标准 LM)
  2. D3PM 去噪生成 K 个 token 草稿
  3. AR 验证 D3PM draft, 接受率测量

关键评估: D3PM 生成的 draft, AR 接受率
  - 期望: > 0% (vs v34b 0%, v31 95.5%)
  - 如果 > 50% 说明"共享 token 空间"有效
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


from v34d_model import SharedBackbone, ARHead, DHead

P("=== v34d D3PM 评测 ===")
DATA = Path("data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; MASK_ID = V
BOS_ID = stoi.get("<bos>", 1)

# 加载模型
ckpt = torch.load("v34d_d3pm.pt", map_location="cuda", weights_only=False)
cfg = ckpt["config"]
backbone = SharedBackbone(V, n_layer=cfg["N_LAYER"], n_embd=cfg["N_EMBD"],
                           n_head=cfg["N_HEAD"], z_dim=cfg["Z_DIM"],
                           t_dim=cfg["T_DIM"], max_seq=cfg["SEQ_LEN"] + cfg["K_WINDOW"] + 4).to("cuda")
backbone.load_state_dict(ckpt["backbone"])
backbone.eval()
ar = ARHead(backbone).to("cuda")
dh = DHead(backbone, n_embd=cfg["N_EMBD"]).to("cuda")
dh.load_state_dict(ckpt["d_head"])
dh.eval()
P(f"v34d loaded")


@torch.no_grad()
def d3pm_draft(z, K=8, n_diff_steps=8):
    """D3PM K 步去噪, 从全 MASK 还原 K tokens"""
    # 起点: 全 MASK
    tokens = torch.full((1, K), MASK_ID, device="cuda", dtype=torch.long)
    for k in range(n_diff_steps):
        t = torch.full((1,), 1.0 - (k + 1) / n_diff_steps, device="cuda")  # 1 → 0
        hidden = backbone(tokens, z, t=t)
        d_logits = dh(hidden)  # (1, K, V+1)
        d_probs = F.softmax(d_logits, dim=-1)
        # 排除 MASK 维度, 只采真实 token
        d_probs_real = d_probs[..., :V]  # (1, K, V)
        # 选 top-1 作为预测
        pred_tokens = d_probs_real.argmax(dim=-1)  # (1, K)
        # 随机选择部分位置 unmask (按 t 调度)
        unmasK_prob = (1.0 - t).item()  # 当前 t 越小, 越接近全 unmask
        rand = torch.rand(1, K, device="cuda")
        unmask = rand < unmasK_prob
        tokens = torch.where(unmask, pred_tokens, tokens)
    return tokens[0]  # (K,)


@torch.no_grad()
def ar_verify(z, prefix_tokens, draft_tokens):
    """AR head 验证 draft"""
    full_tokens = torch.cat([prefix_tokens.view(1, -1), draft_tokens.view(1, -1)], dim=1)
    hidden = backbone(full_tokens, z, t=None)
    logits = ar(hidden[:, -len(draft_tokens):])  # (1, K, V)
    ar_top1 = logits[0].argmax(dim=-1)  # (K,)
    accepted = 0
    for j in range(len(draft_tokens)):
        if ar_top1[j].item() == draft_tokens[j].item():
            accepted += 1
        else:
            break
    return accepted, ar_top1


@torch.no_grad()
def generate(z, max_tokens=100, K=8, n_diff_steps=8):
    """SpS: D3PM 草稿 + AR 验证"""
    cur = torch.tensor([BOS_ID], device="cuda", dtype=torch.long)
    n_rounds = 0
    n_drafted = 0
    n_accepted = 0
    generated_text = []

    while len(cur) - 1 < max_tokens:
        n_rounds += 1
        draft = d3pm_draft(z, K=K, n_diff_steps=n_diff_steps)
        accepted, ar_top1 = ar_verify(z, cur, draft)
        for j in range(accepted):
            cur = torch.cat([cur, draft[j].view(1)])
            generated_text.append(int(draft[j].item()))
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
    return generate(z_test, max_tokens=100, K=8, n_diff_steps=8)


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
avg_accept_rate = np.mean([a/max(d,1) for _, d, a in results_list])
avg_rounds = np.mean([r for r, _, _ in results_list])
P(f"  速度 (100 tokens): {mean_ms:.1f} ms")
P(f"  接受率 (SpS): {avg_accept_rate*100:.1f}%")
P(f"  rounds (avg): {avg_rounds:.1f}")

# PPL
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

# D 单独生成质量 (绕过 AR)
P("\n=== D 单独生成 K tokens 质量 ===")
draft_only = d3pm_draft(z_test, K=8, n_diff_steps=8)
draft_text = "".join([itos.get(int(t.item()), "?") for t in draft_only])
P(f"  D 草稿: {repr(draft_text)}")

# 结果
results = {
    "speed_ms": mean_ms,
    "ppl": float(ppl),
    "acceptance_rate": float(avg_accept_rate),
    "n_rounds_avg": float(avg_rounds),
    "v31_baseline_ms": 206,
    "v31_accept": 0.955,
    "v31_ppl": 2.39,
    "v34b_acc": 0.0,
    "v34b_ms": 484,
}
results["speed_pass"] = bool(mean_ms < 150)
results["accept_pass"] = bool(avg_accept_rate > 0.10)  # 任何 > 0% 都算"共享 token 有效"
results["ppl_pass"] = bool(ppl <= 2.39)
results["all_pass"] = bool(results["speed_pass"] and results["accept_pass"] and results["ppl_pass"])

P(f"\n=== 结果 ===")
P(f"  速度: {'PASS' if results['speed_pass'] else 'FAIL'} ({mean_ms:.1f}ms, vs v31 206ms, v34b 484ms)")
P(f"  PPL: {'PASS' if results['ppl_pass'] else 'FAIL'} ({ppl:.4f}, vs v31 2.39)")
P(f"  接受率: {'PASS' if results['accept_pass'] else 'FAIL'} ({avg_accept_rate*100:.1f}%, vs v31 95.5%, v34b 0%)")
P(f"  总评: {'✅ ALL PASS' if results['all_pass'] else '⚠️ PARTIAL' if avg_accept_rate > 0 else '❌ FAILED'}")

with open("v34d_e2e.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
P(f"\nSaved: v34d_e2e.json")