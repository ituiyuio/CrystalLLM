"""
v40 decoder 注入位置诊断
6 个推理变体, 看哪个能让 v25 用上 z
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import sys
import math
from pathlib import Path

V40_DIR = Path(__file__).resolve().parents[1]
DATA = Path("D:/CrystaLLM/crystalllm/data/processed")

# 复用 v37 的 decoder + data loading
sys.path.insert(0, str(V40_DIR.parent / "v37" / "pipeline"))
from zero_z_eval import BlockCausal, DecoderV25, load_decoder, load_val_data

torch.manual_seed(42); np.random.seed(42)


# ============================================================
# 6 个变体的 forward 函数
# ============================================================
def forward_baseline(decoder, z, x, **kwargs):
    """V1: baseline (z at pos 0)"""
    return decoder(z, x)


def forward_broadcast_z(decoder, z, x, **kwargs):
    """V2: z at pos 0 (as trained, 1x) + z residual to OTHER positions only (calibrated)"""
    B_, T_ = x.shape
    z_emb = decoder.z_to_emb(z).unsqueeze(1)  # (B, 1, D)
    bos_emb = decoder.tok(torch.tensor([decoder.BOS_ID], device=x.device)).expand(B_, 1, -1)
    x_emb = decoder.tok(x)
    # 同 V1 格式: [z, bos, x] (T+2 positions)
    inp = torch.cat([z_emb, bos_emb, x_emb], dim=1)
    inp = inp + decoder.pos(torch.arange(T_ + 2, device=x.device))
    # ★ 在 V1 基础上, 把 z_emb 作为 residual 加到 [1, T+1] 位置 (跳过 pos 0, 避免 2x z)
    z_residual = z_emb.expand(B_, T_ + 2, -1).clone()
    z_residual[:, 0, :] = 0  # pos 0 不再加
    inp = inp + z_residual
    for b in decoder.blocks:
        inp = b(inp)
    logits = decoder.head(decoder.ln_f(inp))
    return logits[:, 1:T_ + 1]  # 同 V1


def forward_z_scaled(decoder, z, x, scale=2.0, **kwargs):
    """V3: z scaled by 2.0 (or V4: 0.5)"""
    return decoder(z * scale, x)


def forward_z_projection(decoder, z, x, **kwargs):
    """V5: z linear projection (random init, frozen)"""
    # 一次性创建 (避免每 batch 新建)
    if not hasattr(forward_z_projection, 'proj'):
        torch.manual_seed(42)
        forward_z_projection.proj = nn.Linear(z.shape[-1], z.shape[-1]).to(z.device)
    z_proj = forward_z_projection.proj(z)
    return decoder(z_proj, x)


def forward_z_at_end(decoder, z, x, **kwargs):
    """V6: z at end of sequence (after x)"""
    B_, T_ = x.shape
    z_emb = decoder.z_to_emb(z).unsqueeze(1)  # (B, 1, D)
    bos_emb = decoder.tok(torch.tensor([decoder.BOS_ID], device=x.device)).expand(B_, 1, -1)
    x_emb = decoder.tok(x)
    # ★ z 移到末尾
    inp = torch.cat([bos_emb, x_emb, z_emb], dim=1)
    inp = inp + decoder.pos(torch.arange(T_ + 2, device=x.device))
    for b in decoder.blocks:
        inp = b(inp)
    logits = decoder.head(decoder.ln_f(inp))
    # logits[:, :T_] 是 x 位置的预测
    return logits[:, :T_]


def forward_broadcast_random(decoder, z, x, **kwargs):
    """V7 (control): random noise broadcast (same shape as V2)"""
    B_, T_ = x.shape
    # 用 random noise 替代 z, 同样的形状和 magnitude
    z_random = torch.randn_like(z)
    z_emb = decoder.z_to_emb(z_random).unsqueeze(1)  # (B, 1, D)
    bos_emb = decoder.tok(torch.tensor([decoder.BOS_ID], device=x.device)).expand(B_, 1, -1)
    x_emb = decoder.tok(x)
    inp = torch.cat([z_emb, bos_emb, x_emb], dim=1)
    inp = inp + decoder.pos(torch.arange(T_ + 2, device=x.device))
    # 同 V2: 把 z_emb 作为 residual 加到 [1, T+1] 位置 (跳过 pos 0)
    z_residual = z_emb.expand(B_, T_ + 2, -1).clone()
    z_residual[:, 0, :] = 0
    inp = inp + z_residual
    for b in decoder.blocks:
        inp = b(inp)
    logits = decoder.head(decoder.ln_f(inp))
    return logits[:, 1:T_ + 1]


# ============================================================
# PPL 评估
# ============================================================
@torch.no_grad()
def eval_ppl(decoder, val_batches, val_z, forward_fn, device="cuda",
             T=512, batch_size=4, max_batches=None, **fwd_kwargs):
    """
    评估 decoder 在给定 forward 函数下的 PPL
    val_batches: list of (x_tensor, index) from v37 get_val_batches
    """
    decoder.eval()
    total_loss = 0.0
    total_tokens = 0

    for bi, (x, i) in enumerate(val_batches):
        if max_batches is not None and bi >= max_batches:
            break
        B_ = x.size(0)
        batch_z = val_z[i:i + B_]
        logits = forward_fn(decoder, batch_z, x, **fwd_kwargs)  # (B, T, V)
        # CE loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            x.reshape(-1),
            reduction='sum'
        )
        total_loss += loss.item()
        total_tokens += x.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return ppl, avg_loss


# ============================================================
# main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=None,
                        help="限制 batch 数 (smoke test)")
    parser.add_argument("--variants", nargs="+",
                        default=["V1", "V2", "V3", "V4", "V5", "V6", "V7"])
    parser.add_argument("--output", default=str(V40_DIR / "decoder_injection_ppl.json"))
    args = parser.parse_args()

    print(f"v40 decoder injection diagnostic | device={args.device}")
    print(f"variants: {args.variants}")

    # 加载 v25 decoder + data
    decoder, V, D_Z, T = load_decoder("v25", args.device)
    val_texts, val_z_all, stoi, itos, V_check = load_val_data(args.device)
    assert V_check == V, f"vocab mismatch: ckpt V={V}, data V={V_check}"
    n_samples = len(val_texts)
    print(f"decoder: V={V}, D_Z={D_Z}, T={T}")
    print(f"val: {n_samples} samples, z shape: {val_z_all.shape}")

    # 使用 v37 的 get_val_batches 复用其随机 chunk 策略 (与 v37 PPL 对齐)
    sys.path.insert(0, str(V40_DIR.parent / "v37" / "pipeline"))
    from zero_z_eval import get_val_batches
    val_batches = get_val_batches(val_texts, stoi, T, B=args.batch_size)
    if args.max_batches is not None:
        val_batches = val_batches[:args.max_batches]
    print(f"val_batches: {len(val_batches)}")

    # 6 个变体
    variants = {
        "V1": ("baseline (z at pos 0)", forward_baseline, {}),
        "V2": ("broadcast z to all positions", forward_broadcast_z, {}),
        "V3": ("z scale x2.0", forward_z_scaled, {"scale": 2.0}),
        "V4": ("z scale x0.5", forward_z_scaled, {"scale": 0.5}),
        "V5": ("z linear projection", forward_z_projection, {}),
        "V6": ("z at end of sequence", forward_z_at_end, {}),
        "V7": ("broadcast random noise (control)", forward_broadcast_random, {}),
    }

    results = {}
    baseline_ppl = None

    for vid in args.variants:
        name, fwd_fn, kwargs = variants[vid]
        print(f"\n--- {vid}: {name} ---")
        ppl, avg_loss = eval_ppl(
            decoder, val_batches, val_z_all, fwd_fn,
            device=args.device, T=T, batch_size=args.batch_size,
            max_batches=args.max_batches, **kwargs
        )
        delta = ""
        if vid == "V1":
            baseline_ppl = ppl
        elif baseline_ppl is not None:
            delta_pct = (ppl - baseline_ppl) / baseline_ppl * 100
            delta = f" (delta = {delta_pct:+.3f}%)"
        print(f"{vid} PPL: {ppl:.4f}{delta}")
        results[vid] = {
            "name": name,
            "ppl": float(ppl),
            "avg_loss": float(avg_loss)
        }
        if baseline_ppl is not None and baseline_ppl > 0:
            results[vid]["delta_vs_V1_pct"] = float((ppl - baseline_ppl) / baseline_ppl * 100)

    # 决策
    print("\n--- Decision ---")
    if baseline_ppl is None:
        decision = "no_baseline"
        action = "rerun_with_V1"
    else:
        # 找最佳变体
        best_vid = min(results.keys(), key=lambda k: results[k]["ppl"])
        best_ppl = results[best_vid]["ppl"]
        best_delta = (best_ppl - baseline_ppl) / baseline_ppl * 100
        if best_vid == "V1":
            decision = "all_neutral"
            action = "decoder_truly_ignores_z -> block-diffusion PoC (v41)"
        elif best_delta < -0.1:
            decision = f"{best_vid}_wins"
            action = f"use {best_vid} injection for v41 PoC"
        else:
            decision = "marginal_difference"
            action = "all variants similar -> block-diffusion PoC (v41)"
        print(f"Best variant: {best_vid} ({best_ppl:.4f}, delta = {best_delta:+.3f}%)")
        print(f"Decision: {decision}")
        print(f"Action: {action}")

    # 写 JSON
    report = {
        "experiment": "v40 decoder injection diagnostic",
        "n_samples": n_samples,
        "n_batches": len(val_batches),
        "max_batches": args.max_batches,
        "results": results,
        "decision": decision,
        "action": action,
        "note": "V1 baseline = v25 BAD-DP (PPL 2.47 per v37)"
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()