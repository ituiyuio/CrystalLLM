"""共享 50M 模型 preset + 训练循环.

Self-contained 50M Transformer (no z injection, no v47 dependency).
用于 v49 前置实验 baseline.

DEVIATION FROM SPEC:
  - Spec suggests wrapping v47's `build_v47_model`. Strategy A chosen instead
    (self-contained) to keep experiment validation independent of v47 refactors.
  - Pos-emb is learned (max_seq_len=2048); exceeding it will raise IndexError,
    which is intentional — callers should slice inputs.
"""
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Path to vocab file
PROJECT_ROOT = Path(__file__).resolve().parents[2]
VOCAB_PATH = PROJECT_ROOT / "crystalllm" / "data" / "processed" / "char_vocab.json"

DEFAULT_VOCAB_SIZE = 2261  # From char_vocab.json (char-level)


def _load_vocab_size() -> int:
    """从 char_vocab.json 加载 vocab size."""
    if VOCAB_PATH.exists():
        with open(VOCAB_PATH, encoding="utf-8") as f:
            vocab = json.load(f)
        # vocab 是 {"stoi": {...}, "itos": [...]} 还是 just dict?
        if isinstance(vocab, dict) and "stoi" in vocab:
            return len(vocab["stoi"])
        return len(vocab)
    return DEFAULT_VOCAB_SIZE


VOCAB_SIZE = _load_vocab_size()


class TransformerBlock(nn.Module):
    """单层 Transformer block (no z injection)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        # x: (B, T, D)
        h = self.ln1(x)
        # Causal mask
        T = x.size(1)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        h_attn, _ = self.attn(h, h, h, attn_mask=causal_mask, is_causal=True, need_weights=False)
        x = x + h_attn
        x = x + self.ffn(self.ln2(x))
        return x


class Transformer50M(nn.Module):
    """~50M parameter Transformer (no z injection)."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = 640, n_layers: int = 10,
                 n_heads: int = 8, d_ff: int = 2560, max_seq_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.config = type("Config", (), {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "d_ff": d_ff,
            "max_seq_len": max_seq_len,
            "dropout": dropout,
        })()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Tie weights
        self.head.weight = self.token_emb.weight
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: (B, T) token ids
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.token_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h)


def build_50m_model(vocab_size: int = VOCAB_SIZE, **kwargs) -> Transformer50M:
    """构建 ~50M 参数的简化 Transformer (无 z 注入)."""
    return Transformer50M(vocab_size=vocab_size, **kwargs)


def count_active_params(model: nn.Module) -> int:
    """统计模型参数总数 (排除 tied weights)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_step(model, batch, optimizer, loss_fn=None):
    """单步 next-token prediction 训练. Returns loss."""
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    # batch 是 tuple (tensor,) 或直接 tensor
    if isinstance(batch, (tuple, list)):
        x = batch[0]
    else:
        x = batch
    x, y = x[:, :-1], x[:, 1:]  # next-token prediction
    logits = model(x)
    loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_ppl(model, val_loader, loss_fn=None):
    """在 val_loader 上计算 perplexity."""
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss(reduction="sum")

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch
            x, y = x[:, :-1], x[:, 1:]
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
            total_tokens += y.numel()
    model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)


if __name__ == "__main__":
    # 快速 sanity check
    model = build_50m_model()
    print(f"Active params: {count_active_params(model):,}")
    print(f"Vocab size: {VOCAB_SIZE}")
    x = torch.randint(0, VOCAB_SIZE, (2, 128))
    out = model(x)
    print(f"Output shape: {out.shape}")
