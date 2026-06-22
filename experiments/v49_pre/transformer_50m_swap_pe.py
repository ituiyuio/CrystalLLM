"""50M Transformer with swappable PE — Exp 24.

复用 exp_runner.TransformerBlock, 但用外部传入的 PE 模块替代 learned pos_emb.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v49_pre.exp_runner import TransformerBlock


class Transformer50MSwapPE(nn.Module):
    """~50M Transformer (复用 exp_runner 的 TransformerBlock)."""

    def __init__(self, vocab_size: int, d_model: int = 640, n_layers: int = 10,
                 n_heads: int = 8, d_ff: int = 2560, max_seq_len: int = 2048,
                 dropout: float = 0.1, pe_module: nn.Module = None):
        super().__init__()
        if pe_module is None:
            from experiments.v49_pre.pe_modules import StandardRoPE
            pe_module = StandardRoPE(d_model=d_model)
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pe = pe_module
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, max_seq_len=max_seq_len)
            for _ in range(n_layers)
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
        B, T = x.shape
        h = self.token_emb(x)
        h = h + self.pe(h)  # 残差注入 (与 cmt_clean 一致)
        for layer in self.layers:
            h = layer(h)
        h = self.ln_f(h)
        return self.head(h)