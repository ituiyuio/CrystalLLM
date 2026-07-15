"""
Exp 33: CWF × FSK Text-Wave Smoke
==================================

文本 → FSK baseband 调制 → 复数波 (S=64) → CWF 单 block 预测 → per-slot IFFT 解调 → char_ids

Nyquist-aware 阈值 (>80% GO, 88.4% 天花板), 8-char shift-by-1 任务.
复用 research/cwf/prototype/cwf_minimal.py::CWFSingleBlock.

Spec: docs/superpowers/specs/2026-07-15-脉冲波-text-wave-design.md
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
PROTOTYPE_DIR = HERE.parents[1] / "prototype"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROTOTYPE_DIR))

from cwf_minimal import CWFSingleBlock  # noqa: E402

# === 物理参数 (来自 spec §3.2) ===
VOCAB_SIZE = 8            # top-8 ASCII 频率 (空 e t a o i n s)
S = 64                    # CWF 网格点数 (= CWFSingleBlock d)
N_CHARS = 8               # 段内字符数
T_CHAR = 8                # 每字符占网格点数 (= S // N_CHARS = VOCAB_SIZE)
TOP8_ASCII = [' ', 'e', 't', 'a', 'o', 'i', 'n', 's']

# === 训练超参 (来自 spec §3.5, 与 exp31/32 一致) ===
TRAIN_STEPS = 1000
BATCH_SIZE = 32
LR = 3e-4
SEEDS = [42, 123, 2024, 7, 11]
N_EVAL = 1000
DEVICE = "cpu"

RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# FSK 编码: char_id → baseband 复数波
# ===========================================================================
def fsk_encode(char_ids: torch.Tensor) -> torch.Tensor:
    """
    FSK baseband 编码. 每字符 c 占 T_CHAR=8 网格点, 1 周期, 频率 = c/T_CHAR.

    Args:
        char_ids: (B, N_CHARS) int64, in [0, VOCAB_SIZE)
    Returns:
        waveform: (B, S=64) complex64
    """
    B, N = char_ids.shape
    waveform = torch.zeros(B, S, dtype=torch.complex64, device=char_ids.device)
    n = torch.arange(T_CHAR, dtype=torch.float32, device=char_ids.device)  # (8,)
    for c in range(N):
        freq = char_ids[:, c].float()  # (B,) in [0, VOCAB_SIZE)
        # 1 周期相位: 2π · freq · n / T_CHAR
        # Conjugate sign (-sin) so IFFT argmax directly yields the frequency bin.
        # (Using +sin gives reflected peak at 8-k due to IFFT convention.)
        phase = 2.0 * math.pi * freq.unsqueeze(-1) * n.unsqueeze(0) / T_CHAR  # (B, 8)
        waveform[:, c * T_CHAR : (c + 1) * T_CHAR] = torch.complex(
            torch.cos(phase), -torch.sin(phase)
        )
    return waveform


def fsk_decode(waveform: torch.Tensor) -> torch.Tensor:
    """
    FSK 解码: per-slot IFFT, argmax amplitude bin.

    Args:
        waveform: (B, S=64) complex64
    Returns:
        char_ids: (B, N_CHARS) int64, in [0, VOCAB_SIZE)
    """
    B, S_in = waveform.shape
    assert S_in == S, f"expected S={S}, got {S_in}"
    char_ids = torch.zeros(B, N_CHARS, dtype=torch.long, device=waveform.device)
    for c in range(N_CHARS):
        slot = waveform[:, c * T_CHAR : (c + 1) * T_CHAR]  # (B, 8) complex
        spec = torch.fft.ifft(slot, dim=-1).abs()  # (B, 8) amplitude
        # argmax 可能返回 [0, 7], 但 freq=7 是 Nyquist; 取 [0, 7) 范围取 [0, 7]
        char_ids[:, c] = spec.argmax(dim=-1)
    return char_ids


# ===========================================================================
# 数据生成: 内存随机, shift-by-1
# ===========================================================================
def sample_batch(batch_size: int, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    随机 8-char 序列, 目标 = input shift-by-1 (最后 1 位是新的随机).

    Args:
        batch_size: B
        seed: 随机种子 (None = 全随机)
    Returns:
        (input_char_ids, target_char_ids): 都是 (B, N_CHARS) int64
        target[:, :-1] == input[:, 1:], target[:, -1] 是新随机
    """
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    inp = torch.randint(0, VOCAB_SIZE, (batch_size, N_CHARS), generator=gen)
    # target[:, :-1] = inp[:, 1:], target[:, -1] = 新随机
    new_chars = torch.randint(0, VOCAB_SIZE, (batch_size, 1), generator=gen)
    tgt = torch.cat([inp[:, 1:], new_chars], dim=1)
    return inp, tgt


# ===========================================================================
# 模型: CWF 包装 (复用 CWFSingleBlock, d=S=64 skip projection)
# ===========================================================================
class CWFFSKPredictor(nn.Module):
    """CWF 单 block 包装: complex 波形 → complex 波形.

    复用 CWFSingleBlock(d=64, hidden_mult=2), 输入 (B, 1, 64, 2) 单 token 维度.
    """
    def __init__(self, d: int = S, hidden_mult: int = 2):
        super().__init__()
        self.block = CWFSingleBlock(d=d, hidden_mult=hidden_mult)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psi: (B, S=64) complex
        Returns:
            psi_out: (B, S=64) complex
        """
        # (B, S) complex → (B, 1, S, 2) for CWFSingleBlock API
        psi_ri = torch.stack([psi.real, psi.imag], dim=-1)  # (B, S, 2)
        psi_ri = psi_ri.unsqueeze(1)  # (B, 1, S, 2)
        psi_out_ri, _ = self.block(psi_ri)
        psi_out_ri = psi_out_ri.squeeze(1)  # (B, S, 2)
        return torch.complex(psi_out_ri[..., 0], psi_out_ri[..., 1])


# ===========================================================================
# 训练一个 (model, seed) 配置, 复用 exp31 风格
# ===========================================================================
def train_one(model: nn.Module, seed: int, steps: int = TRAIN_STEPS) -> dict:
    """
    训练 model 在 FSK shift-by-1 任务上.

    Args:
        model: nn.Module, 输入输出 (B, S) complex
        seed: 随机种子
        steps: 训练步数 (默认 1000)
    Returns:
        {"losses": [...], "elapsed_s": float, "final_mse": float}
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    losses = []
    t0 = time.time()
    for step in range(steps):
        inp_ids, tgt_ids = sample_batch(BATCH_SIZE)
        psi_in = fsk_encode(inp_ids).to(DEVICE)  # (B, S) complex
        psi_tgt = fsk_encode(tgt_ids).to(DEVICE)  # (B, S) complex
        psi_pred = model(psi_in)
        # MSE on real + imag
        loss = F.mse_loss(psi_pred.real, psi_tgt.real) + F.mse_loss(psi_pred.imag, psi_tgt.imag)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    elapsed = time.time() - t0

    # 评估: 重新生成 N_EVAL 个测试样本, 算 final MSE
    model.eval()
    with torch.no_grad():
        inp_ids, tgt_ids = sample_batch(N_EVAL, seed=seed + 999)
        psi_in = fsk_encode(inp_ids).to(DEVICE)
        psi_tgt = fsk_encode(tgt_ids).to(DEVICE)
        psi_pred = model(psi_in)
        final_mse = (
            F.mse_loss(psi_pred.real, psi_tgt.real) + F.mse_loss(psi_pred.imag, psi_tgt.imag)
        ).item()
    return {"losses": losses, "elapsed_s": elapsed, "final_mse": final_mse}


class TransformerFSKPredictor(nn.Module):
    """1D Transformer encoder baseline (实数, 处理 real/imag 2 通道).

    复用 exp31 TransformerPredictor 的结构, 改输入通道 1→2 (real+imag).
    """
    def __init__(self, d: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(2, d)  # 输入 2 通道 (real, imag)
        self.pos_embed = nn.Parameter(torch.randn(S, d) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=d * 4,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d, 2)  # 输出 2 通道 (real, imag)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psi: (B, S=64) complex
        Returns:
            psi_out: (B, S=64) complex
        """
        # (B, S) complex → (B, S, 2) real
        x = torch.stack([psi.real, psi.imag], dim=-1)  # (B, S, 2)
        # Transformer
        h = self.input_proj(x) + self.pos_embed.unsqueeze(0)  # (B, S, d)
        h = self.encoder(h)  # (B, S, d)
        out = self.output_proj(h)  # (B, S, 2)
        return torch.complex(out[..., 0], out[..., 1])


# ===========================================================================
# 评估: char accuracy + per-position accuracy (sanity check shift learned)
# ===========================================================================
def evaluate_model(model: nn.Module, n_samples: int = N_EVAL, seed: int = 0) -> dict:
    """
    评估 model 在 shift-by-1 任务上.

    Returns:
        {
            "char_acc": float,            # 全 8 位平均
            "per_pos_acc": list[float],   # 长度 8, 每个位置的 accuracy
            "waveform_mse": float,
        }
    """
    model.eval()
    inp_ids, tgt_ids = sample_batch(n_samples, seed=seed)
    psi_in = fsk_encode(inp_ids)
    with torch.no_grad():
        psi_pred = model(psi_in)
    pred_ids = fsk_decode(psi_pred)  # (B, N_CHARS) int64
    tgt_ids = tgt_ids.long()
    # 全局 accuracy
    char_acc = (pred_ids == tgt_ids).float().mean().item()
    # 位置级别 accuracy
    per_pos_acc = []
    for c in range(N_CHARS):
        pos_acc = (pred_ids[:, c] == tgt_ids[:, c]).float().mean().item()
        per_pos_acc.append(pos_acc)
    # waveform MSE
    waveform_mse = (
        F.mse_loss(psi_pred.real, psi_in.real) + F.mse_loss(psi_pred.imag, psi_in.imag)
    ).item() / 2.0
    return {
        "char_acc": char_acc,
        "per_pos_acc": per_pos_acc,
        "waveform_mse": waveform_mse,
    }


# ===========================================================================
# 判决 (必须在 run_main 之前定义, 否则脚本运行时 NameError)
# ===========================================================================
def compute_verdict(cwf_acc: float, trans_acc: float) -> str:
    """
    Nyquist-aware 判决 (spec §2):
      > 0.80          → GO
      0.50-0.80 + 优势 ≥ 2x → PARTIAL
      0.50-0.80 + 无优势 → NEUTRAL
      < 0.50          → DEAD
    """
    if cwf_acc >= 0.80:
        return "GO"
    if cwf_acc >= 0.50:
        # 优势 = trans_acc / cwf_acc (trans 是 cwf 的几倍) ; CWF 优势要求 ratio < 0.5
        ratio = cwf_acc / max(trans_acc, 1e-6)
        if ratio < 0.5:
            return "PARTIAL"
        return "NEUTRAL"
    return "DEAD"


# ===========================================================================
# 主程序: multi-seed 跑 CWF + Trans baseline, 输出 results.json
# ===========================================================================
def run_main(seeds: list[int] = None, steps: int = TRAIN_STEPS) -> dict:
    """
    跑 5 seeds × 2 模型 (CWF + Trans), 收集 char accuracy + per-position breakdown,
    计算 verdict, 写入 results/exp33_results.json.

    Args:
        seeds: 随机种子列表 (默认 SEEDS = [42, 123, 2024, 7, 11])
        steps: 训练步数 (默认 1000, 测试时可调小)
    Returns:
        results dict
    """
    if seeds is None:
        seeds = SEEDS
    print("=" * 70)
    print(f"Exp 33: CWF × FSK Text-Wave Smoke ({len(seeds)} seeds × {steps} steps)")
    print("=" * 70)
    print(f"Config: VOCAB={VOCAB_SIZE}, S={S}, N_CHARS={N_CHARS}, T_CHAR={T_CHAR}")
    print(f"        BATCH={BATCH_SIZE}, LR={LR}, DEVICE={DEVICE}")
    print()

    # 参数量统计
    cwf = CWFFSKPredictor()
    trans = TransformerFSKPredictor()
    cwf_params = sum(p.numel() for p in cwf.parameters())
    trans_params = sum(p.numel() for p in trans.parameters())
    print(f"  CWF   params: {cwf_params:,}")
    print(f"  Trans params: {trans_params:,}\n")

    per_seed = []
    cwf_accs, trans_accs, ratios = [], [], []

    for seed in seeds:
        print(f"[seed {seed}] CWF training ({steps} steps)...")
        t0 = time.time()
        cwf = CWFFSKPredictor()
        train_one(cwf, seed, steps=steps)
        cwf_eval = evaluate_model(cwf, n_samples=N_EVAL, seed=seed + 999)
        cwf_t = time.time() - t0
        print(f"  char_acc={cwf_eval['char_acc']:.3f}  ({cwf_t:.0f}s)")
        print(f"  per_pos={[f'{x:.2f}' for x in cwf_eval['per_pos_acc']]}")

        print(f"[seed {seed}] Trans training ({steps} steps)...")
        t0 = time.time()
        trans = TransformerFSKPredictor()
        train_one(trans, seed, steps=steps)
        trans_eval = evaluate_model(trans, n_samples=N_EVAL, seed=seed + 999)
        trans_t = time.time() - t0
        print(f"  char_acc={trans_eval['char_acc']:.3f}  ({trans_t:.0f}s)")
        print(f"  per_pos={[f'{x:.2f}' for x in trans_eval['per_pos_acc']]}\n")

        cwf_accs.append(cwf_eval['char_acc'])
        trans_accs.append(trans_eval['char_acc'])
        # ratio < 1 = CWF 更好
        ratio = cwf_eval['char_acc'] / max(trans_eval['char_acc'], 1e-6)
        ratios.append(ratio)

        per_seed.append({
            "seed": seed,
            "cwf_char_acc": cwf_eval['char_acc'],
            "trans_char_acc": trans_eval['char_acc'],
            "cwf_per_pos": cwf_eval['per_pos_acc'],
            "trans_per_pos": trans_eval['per_pos_acc'],
            "cwf_waveform_mse": cwf_eval['waveform_mse'],
            "trans_waveform_mse": trans_eval['waveform_mse'],
            "ratio": ratio,
        })

    # 总结
    cwf_arr = np.array(cwf_accs)
    trans_arr = np.array(trans_accs)
    ratios_arr = np.array(ratios)
    median_cwf = float(np.median(cwf_arr))
    median_trans = float(np.median(trans_arr))
    median_ratio = float(np.median(ratios_arr))

    # verdict: 用中位数对比
    verdict = compute_verdict(median_cwf, median_trans)
    verdict_messages = {
        "GO": f"CWF 达 80% GO 阈值 (median={median_cwf:.3f}), 88.4% 天花板的 {median_cwf/0.884*100:.1f}%",
        "PARTIAL": f"CWF 重建 {median_cwf:.3f}, 优势 {median_trans/max(median_cwf,1e-6):.1f}x ≥ 2x → PARTIAL",
        "NEUTRAL": f"CWF 重建 {median_cwf:.3f}, 但优势不足 2x → NEUTRAL",
        "DEAD": f"CWF 重建 {median_cwf:.3f} < 50% → DEAD, 归档",
    }
    verdict_msg = verdict_messages[verdict]

    print("=" * 70)
    print(f"Summary ({len(seeds)} seeds, {steps} steps):")
    print("=" * 70)
    print(f"  CWF  median acc: {median_cwf:.3f}  "
          f"min={cwf_arr.min():.3f}  max={cwf_arr.max():.3f}  std={cwf_arr.std():.3f}")
    print(f"  Trans median acc: {median_trans:.3f}  "
          f"min={trans_arr.min():.3f}  max={trans_arr.max():.3f}  std={trans_arr.std():.3f}")
    print(f"  Ratio (CWF/Trans): median={median_ratio:.3f}  "
          f"min={ratios_arr.min():.3f}  max={ratios_arr.max():.3f}")
    print(f"\nVerdict: {verdict_msg}")

    results = {
        "config": {
            "vocab_size": VOCAB_SIZE,
            "s": S,
            "n_chars": N_CHARS,
            "t_char": T_CHAR,
            "train_steps": steps,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "n_seeds": len(seeds),
            "seeds": seeds,
            "cwf_params": cwf_params,
            "trans_params": trans_params,
            "device": DEVICE,
        },
        "per_seed": per_seed,
        "summary": {
            "cwf_median": median_cwf,
            "cwf_min": float(cwf_arr.min()),
            "cwf_max": float(cwf_arr.max()),
            "cwf_std": float(cwf_arr.std()),
            "trans_median": median_trans,
            "trans_min": float(trans_arr.min()),
            "trans_max": float(trans_arr.max()),
            "trans_std": float(trans_arr.std()),
            "ratio_median": median_ratio,
            "ratio_min": float(ratios_arr.min()),
            "ratio_max": float(ratios_arr.max()),
            "ratio_std": float(ratios_arr.std()),
        },
        "verdict": verdict,
        "verdict_message": verdict_msg,
    }
    out_path = RESULTS_DIR / "exp33_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Exp 33: CWF × FSK Text-Wave Smoke")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS,
                        help=f"训练步数 (default {TRAIN_STEPS})")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                        help=f"随机种子列表 (default {SEEDS})")
    args = parser.parse_args()
    run_main(seeds=args.seeds, steps=args.steps)
