"""
Exp 26b: 同一模型, 加测 teacher-forcing PPL 作为"模型真实能力"基线
"""
import math, json, torch
import torch.nn.functional as F
import numpy as np
from exp26_knife4_minimal import MiniGPT, load_data, get_batch, DEVICE

torch.manual_seed(42)

def eval_teacher_forcing(model, val_ids, num_seqs=80, seq_len=128):
    """用 ground truth 作为输入, 标准 PPL (没有自回归暴露偏差)."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    n = len(val_ids) - seq_len - 1
    starts = torch.randint(0, n, (num_seqs,))
    with torch.no_grad():
        for s in starts:
            ids = val_ids[s:s+seq_len+1].to(DEVICE).unsqueeze(0)
            x = ids[:, :-1]
            y = ids[:, 1:]
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += y.numel()
    ppl = math.exp(total_loss / total_tokens)
    return total_loss / total_tokens, ppl


def main():
    train_ids, val_ids = load_data()
    model = MiniGPT().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params: {n_params:,}")

    print("\n[TRAIN] 2000 steps teacher forcing + CE")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    model.train()
    import time
    t0 = time.time()
    for step in range(1, 2001):
        x, y = get_batch(train_ids, 32, 128)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 500 == 0:
            print(f"  step {step}  loss={loss.item():.4f}  elapsed={time.time()-t0:.1f}s")

    print("\n[EVAL] teacher-forcing PPL on val (true model capability)")
    tf_loss, tf_ppl = eval_teacher_forcing(model, val_ids)

    # Re-load the autoregressive results from the prior run
    with open("D:/CrystaLLM/experiments/v49_pre/exp26_knife4_results.json") as f:
        prior = json.load(f)

    print("\n" + "="*60)
    print("FULL PICTURE")
    print("="*60)
    print(f"  Teacher-forcing PPL      = {tf_ppl:>10.2f}     ← 模型的真实能力 (无暴露偏差)")
    print(f"  Autoregressive argmax    = {prior['ppl_argmax']:>10.2f}     ← 离散反馈的代价")
    print(f"  Autoregressive Soft-Exp  = {prior['ppl_soft']:>10.2f}     ← 连续反馈 (刀4)")
    print(f"  Autoregressive Hard-Exp  = {prior['ppl_hard']:>10.2f}     ← Gumbel 采样")
    print()
    print(f"  Soft-Exp vs Argmax       = {prior['delta_soft_pct']:+.2f}%   (连续反馈显著更稳)")
    print(f"  Soft-Exp vs Teacher-F    = {(prior['ppl_soft']/tf_ppl - 1)*100:+.2f}%   (仍有 ~{prior['ppl_soft']/tf_ppl:.1f}x 暴露偏差)")
    print()
    print("INTERPRETATION:")
    print(f"  • 模型真实 PPL ~{tf_ppl:.0f}, 但 argmax 自回归飙到 {prior['ppl_argmax']:.0f} ({prior['ppl_argmax']/tf_ppl:.0f}x 膨胀)")
    print(f"  • Soft-Exp 把暴露偏差压到 {prior['ppl_soft']/tf_ppl:.0f}x ({(1-prior['ppl_soft']/prior['ppl_argmax'])*100:.0f}% 缩减)")
    print(f"  • 这不是\"波函数理论胜利\", 而是: 训练不充分时, 连续期望 = 内置 soft beam search")
    print()
    print("  ⚠️ 重要警告:")
    print("     此优势是'模型不确定时的稳定性增益'.")
    print("     训练充分后 argmax ≈ expected_embed, 优势会消失.")
    print("     这不能证明 CMT 的复数/KAN/Cayley 链, 仅证明连续反馈本身有'低置信时稳定'的工程价值.")
    print("     刀4 在'最小反例'上是 FALSIFIED (你预测) → 实际是 VERIFIED (有条件).")

    out = {
        "tf_ppl": tf_ppl,
        "ppl_argmax": prior['ppl_argmax'],
        "ppl_soft": prior['ppl_soft'],
        "ppl_hard": prior['ppl_hard'],
        "exposure_bias_argmax_factor": prior['ppl_argmax']/tf_ppl,
        "exposure_bias_soft_factor": prior['ppl_soft']/tf_ppl,
        "soft_advantage_pct": prior['delta_soft_pct'],
    }
    with open("D:/CrystaLLM/experiments/v49_pre/exp26b_results.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()