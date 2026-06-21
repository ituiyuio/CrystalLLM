"""验证 CMT-Fixed 50M 模型的输出.

步骤:
  1. 构建 CMTFixed50M (用 cmt_v2 共享模块)
  2. 未训练 forward pass: 检查 shape, dtype, 参数量
  3. 短训练 (500 step): 验证训练链路
  4. 文本生成: 给定 prompt, 生成 100 字符, 与 baseline 对照
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.v49_pre.data_loader import build_subset_loader, _load_vocab
from experiments.v49_pre.exp_runner import build_50m_model, train_step, count_active_params


# ---------------------------------------------------------------------------
# CMT-Fixed 50M (复用 cmt_v2 共享模块 + Exp 14/15 已验证组件)
# ---------------------------------------------------------------------------
from experiments.v49_pre.cmt_v2 import LieRE_NoContext, ComplexKANFFN_TrueMul


class ComplexLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln_r = nn.LayerNorm(dim)
        self.ln_i = nn.LayerNorm(dim)
    def forward(self, z):
        d = z.size(-1) // 2
        return torch.cat([self.ln_r(z[..., :d]), self.ln_i(z[..., d:])], dim=-1)


class CMTBlockFixed(nn.Module):
    def __init__(self, d_model, n_heads=8, kan_dim=96, dropout=0.1):
        super().__init__()
        self.pe = LieRE_NoContext(d_model)
        self.ln1 = ComplexLayerNorm(d_model)
        self.attn = nn.MultiheadAttention(2 * d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = ComplexLayerNorm(d_model)
        self.ffn = ComplexKANFFN_TrueMul(d_model, kan_dim=kan_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, z):
        z = z + self.pe(z)
        z_norm = self.ln1(z)
        attn_out, _ = self.attn(z_norm, z_norm, z_norm, need_weights=False)
        z = z + self.dropout(attn_out)
        z = z + self.ffn(self.ln2(z))
        return z


class CMTFixed50M(nn.Module):
    def __init__(self, vocab_size, d_model=640, n_layers=8, n_heads=8,
                 kan_dim=96, max_seq_len=2048, dropout=0.1):
        super().__init__()
        self.config = type("Config", (), {
            "vocab_size": vocab_size, "d_model": d_model, "n_layers": n_layers,
            "n_heads": n_heads, "kan_dim": kan_dim,
            "max_seq_len": max_seq_len, "dropout": dropout,
        })()
        self.token_emb = nn.Embedding(vocab_size, 2 * d_model)
        self.pos_emb = nn.Embedding(max_seq_len, 2 * d_model)
        self.layers = nn.ModuleList([
            CMTBlockFixed(d_model, n_heads, kan_dim, dropout) for _ in range(n_layers)
        ])
        self.ln_f = ComplexLayerNorm(d_model)
        self.head = nn.Linear(2 * d_model, vocab_size, bias=False)
        self._init_weights()
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        z = self.token_emb(x) + self.pos_emb(pos)
        for layer in self.layers:
            z = layer(z)
        z = self.ln_f(z)
        return self.head(z)


# ---------------------------------------------------------------------------
# 文本生成函数
# ---------------------------------------------------------------------------
def generate_text(model, prompt_tokens, max_new_tokens=100, temperature=1.0, top_k=50):
    """Greedy / top-k sampling 生成."""
    model.eval()
    device = next(model.parameters()).device
    generated = list(prompt_tokens)
    for _ in range(max_new_tokens):
        # 取最后 max_seq_len 个 token
        ctx = torch.tensor([generated[-model.config.max_seq_len:]], device=device, dtype=torch.long)
        with torch.no_grad():
            logits = model(ctx)
        # 取最后一个位置的 logits
        logits = logits[0, -1, :] / temperature
        # top-k 过滤
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_token)
    return generated


def tokens_to_text(tokens, itos):
    """Token id 列表 → 字符串."""
    chars = []
    for t in tokens:
        if 0 <= t < len(itos):
            chars.append(itos[t])
        else:
            chars.append("?")
    return "".join(chars)


# ---------------------------------------------------------------------------
# 主验证流程
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    stoi, vocab_size = _load_vocab()
    itos = {v: k for k, v in stoi.items()}
    print(f"Vocab size: {vocab_size}\n")

    # ------------------------------------------------------------------
    # Step 1: 构建模型 + 未训练 forward
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Step 1: 构建 CMT-Fixed 50M 模型")
    print("=" * 70)
    torch.manual_seed(42)
    model = CMTFixed50M(vocab_size=vocab_size, d_model=640, n_layers=8, n_heads=8).to(device)
    n_params = count_active_params(model)
    print(f"参数量: {n_params:,} ({n_params/1e6:.1f}M)")

    # Forward pass 测试
    print("\n未训练 forward pass 测试:")
    test_input = torch.randint(0, vocab_size, (2, 64), device=device)
    with torch.no_grad():
        logits = model(test_input)
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Logits mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}")
    print(f"  Logits min: {logits.min().item():.4f}, max: {logits.max().item():.4f}")
    print(f"  [OK] Forward pass 工作正常\n")

    # ------------------------------------------------------------------
    # Step 2: 短训练 (500 step) — 验证训练链路
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Step 2: 短训练 500 step (验证训练链路)")
    print("=" * 70)
    model.train()
    loader = build_subset_loader(batch_size=8, seq_len=256, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    t0 = time.time()
    losses = []
    for step in range(1, 501):
        batch = next(iter(loader))[0].to(device)
        loss = train_step(model, batch, optimizer)
        losses.append(loss)
        if step % 100 == 0:
            recent = sum(losses[-100:]) / 100
            print(f"  Step {step:4d}: loss={recent:.4f}")
    elapsed = time.time() - t0
    print(f"\n  训练耗时: {elapsed:.1f}s")
    print(f"  Loss 下降: {losses[0]:.4f} → {losses[-1]:.4f} (delta={losses[-1]-losses[0]:+.4f})")
    print(f"  [OK] 训练链路工作正常\n")

    # ------------------------------------------------------------------
    # Step 3: 文本生成测试
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Step 3: 文本生成 (500-step 训练后)")
    print("=" * 70)

    # 准备 prompt: 用训练集的第一个 token sequence
    sample_text = "The quick brown fox jumps over the lazy dog"
    prompt_tokens = [stoi.get(c, 0) for c in sample_text]
    print(f"\nPrompt: {sample_text!r}")
    print(f"Prompt tokens: {prompt_tokens[:20]}...")

    for temp in [0.5, 1.0]:
        generated = generate_text(model, prompt_tokens, max_new_tokens=80,
                                   temperature=temp, top_k=50)
        gen_text = tokens_to_text(generated, itos)
        print(f"\n[Temperature={temp}] 生成 ({len(generated)-len(prompt_tokens)} chars):")
        print(f"  {gen_text!r}")

    # Top-5 预测 (单步)
    print(f"\n下一步 top-5 预测 (给定 prompt '{sample_text[:20]}...'):")
    model.eval()
    ctx = torch.tensor([prompt_tokens[:32]], device=device, dtype=torch.long)
    with torch.no_grad():
        logits = model(ctx)
    last_logits = logits[0, -1, :]
    probs = F.softmax(last_logits, dim=-1)
    top5 = torch.topk(probs, 5)
    for prob, idx in zip(top5.values, top5.indices):
        char = itos.get(idx.item(), "?")
        print(f"  P={prob.item():.4f}  token={idx.item():4d}  char={char!r}")

    # ------------------------------------------------------------------
    # Step 4: 对比 baseline (短训练同样 500 step)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 4: 对比 baseline (同样 500 step 训练)")
    print("=" * 70)

    torch.manual_seed(42)
    baseline = build_50m_model(vocab_size=vocab_size).to(device)
    n_base = count_active_params(baseline)
    print(f"\nBaseline 参数量: {n_base:,} ({n_base/1e6:.1f}M)")

    baseline.train()
    loader_b = build_subset_loader(batch_size=8, seq_len=256, shuffle=True)
    optimizer_b = torch.optim.AdamW(baseline.parameters(), lr=1e-4)
    losses_b = []
    for step in range(1, 501):
        batch = next(iter(loader_b))[0].to(device)
        loss = train_step(baseline, batch, optimizer_b)
        losses_b.append(loss)
        if step % 100 == 0:
            recent = sum(losses_b[-100:]) / 100
            print(f"  Step {step:4d}: loss={recent:.4f}")

    # Baseline 生成
    baseline.eval()
    print(f"\nBaseline 生成 (temp=1.0):")
    gen_b = generate_text(baseline, prompt_tokens, max_new_tokens=80, temperature=1.0, top_k=50)
    print(f"  {tokens_to_text(gen_b, itos)!r}")

    # ------------------------------------------------------------------
    # 总结
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print(f"CMT-Fixed:     {n_params/1e6:.1f}M params, 500-step loss={losses[-1]:.4f}")
    print(f"baseline:      {n_base/1e6:.1f}M params, 500-step loss={losses_b[-1]:.4f}")
    print(f"差距: CMT-Fixed 参数量是 baseline 的 {n_params/n_base:.2f}×, "
          f"500-step loss 差 {losses[-1]-losses_b[-1]:+.4f}")
    print(f"\n注: 500 step 不够得出 PPL 结论, 完整 10k step 训练结果见 Exp 14/15.")
    print(f"    CMT-Fixed @ 10k step: PPL 1.01 vs baseline 2.07")


if __name__ == "__main__":
    main()