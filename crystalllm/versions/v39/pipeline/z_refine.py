"""
v39 z 细化诊断 - 方向 B (better MI) + 方向 C (维度子集)
"""
import torch
import torch.nn as nn
import numpy as np
import json
import sys
from pathlib import Path

V39_DIR = Path(__file__).resolve().parents[1]
V38_DIR = V39_DIR.parent / "v38"
DATA = Path("D:/CrystaLLM/crystalllm/data/processed")

# 复用 v38 的函数
sys.path.insert(0, str(V38_DIR / "pipeline"))
from z_health_check import (
    load_val_data, MINE, compute_mi_lower_bound,
    compute_collapse_ratio, compute_class_separability,
    load_val_labels, js_divergence_gaussian
)


# ============================================================
# 方向 B: 更好的 text features
# ============================================================
def make_better_text_features(val_texts, embed_dim=64, seed=42):
    """
    用 frozen random embedding + mean pooling 替代 6 维弱特征.

    步骤:
      1. 加载 char_vocab (vocab_size=2261)
      2. 创建 frozen random embedding matrix (2261, embed_dim), seed=42
      3. 把每个 text 转成 token IDs
      4. mean-pool embeddings → (N, embed_dim) tensor
    """
    vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
    stoi = vocab["stoi"]
    V = vocab["vocab_size"]

    torch.manual_seed(seed)
    embed = nn.Embedding(V, embed_dim)
    embed.weight.requires_grad = False  # frozen

    features = []
    for text in val_texts[:1016]:
        ids = [stoi.get(c, 0) for c in text[:256]]  # 限长 256
        if len(ids) == 0:
            ids = [0]
        ids_tensor = torch.tensor(ids, dtype=torch.long)
        emb = embed(ids_tensor)  # (L, embed_dim)
        mean_emb = emb.mean(dim=0)  # (embed_dim,)
        features.append(mean_emb)

    return torch.stack(features)  # (N, embed_dim)


def make_char_trigram_features(val_texts, top_k=64):
    """
    备选特征: 字符 trigrams 的 one-hot (top-K 高频)
    """
    from collections import Counter
    # 统计全语料 trigram 频率
    counter = Counter()
    for text in val_texts[:1016]:
        for i in range(len(text) - 2):
            counter[text[i:i+3]] += 1
    top_trigrams = [t for t, _ in counter.most_common(top_k)]
    trigram_to_idx = {t: i for i, t in enumerate(top_trigrams)}

    features = []
    for text in val_texts[:1016]:
        text_trigrams = set(text[i:i+3] for i in range(len(text) - 2))
        feat = [1.0 if t in text_trigrams else 0.0 for t in top_trigrams]
        features.append(feat)

    return torch.tensor(features, dtype=torch.float32)


# ============================================================
# 方向 C: 维度子集分析
# ============================================================
def per_dim_domain_discrimination(z: np.ndarray, labels: np.ndarray):
    """
    对每维 z 计算 code vs agentic 的 JS 散度 (1-D 高斯闭式解)
    返回: (256,) numpy array of JS scores per dim
    """
    classes = np.unique(labels)
    if len(classes) < 2:
        return np.zeros(z.shape[1])

    D = z.shape[1]
    js_scores = np.zeros(D)

    for d in range(D):
        z_d = z[:, d:d+1]  # (N, 1)
        class_stats = {}
        for c in classes:
            z_c = z_d[labels == c]
            if len(z_c) < 5:
                continue
            class_stats[c] = (z_c.mean(axis=0), np.cov(z_c.T) + 1e-6 * np.eye(1))
        if len(class_stats) < 2:
            continue
        classes_list = list(class_stats.keys())
        js_d = []
        for i, c1 in enumerate(classes_list):
            for c2 in classes_list[i+1:]:
                js_d.append(max(0, js_divergence_gaussian(
                    class_stats[c1][0], class_stats[c1][1],
                    class_stats[c2][0], class_stats[c2][1])))
        js_scores[d] = np.mean(js_d) if js_d else 0.0

    return js_scores


# ============================================================
# main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_samples", type=int, default=1016)
    parser.add_argument("--mine_steps", type=int, default=2000)
    parser.add_argument("--mine_runs", type=int, default=5)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--output", default=str(V39_DIR / "v39_refine_report.json"))
    args = parser.parse_args()

    print(f"v39 z refine | device={args.device} | n_samples={args.n_samples}")

    # 加载数据
    val_texts, val_z_np = load_val_data()
    val_z = torch.tensor(val_z_np[:args.n_samples], dtype=torch.float32, device=args.device)
    val_texts = val_texts[:args.n_samples]
    labels = load_val_labels()[:args.n_samples]

    print(f"z shape: {val_z.shape}, labels: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # ============================================================
    # 方向 B: 重测 MI with better features
    # ============================================================
    print("\n=== 方向 B: 重测 MI with token embeddings ===")

    # B1: random embedding + mean pool
    print("\n--- B1: random embedding + mean pooling (64 dim) ---")
    text_emb_64 = make_better_text_features(val_texts, embed_dim=args.embed_dim, seed=42)
    text_emb_64 = text_emb_64.to(args.device)
    mi_b1_mean, mi_b1_std = compute_mi_lower_bound(text_emb_64, val_z,
                                                     n_steps=args.mine_steps,
                                                     n_runs=args.mine_runs)
    print(f"MI (64-dim random embed + mean pool): {mi_b1_mean:.4f} ± {mi_b1_std:.4f}")

    # B2: char trigrams (top-64)
    print("\n--- B2: char trigrams (top-64) ---")
    text_emb_tri = make_char_trigram_features(val_texts, top_k=64)
    text_emb_tri = text_emb_tri.to(args.device)
    mi_b2_mean, mi_b2_std = compute_mi_lower_bound(text_emb_tri, val_z,
                                                     n_steps=args.mine_steps,
                                                     n_runs=args.mine_runs)
    print(f"MI (64-dim char trigrams): {mi_b2_mean:.4f} ± {mi_b2_std:.4f}")

    # 对比 v38 的 6 维特征
    print(f"\n--- v38 baseline (6-dim weak features): MI = 0.0598 ---")
    v38_baseline = 0.0598
    improvement_b1 = mi_b1_mean / v38_baseline if v38_baseline > 0 else 0
    improvement_b2 = mi_b2_mean / v38_baseline if v38_baseline > 0 else 0
    print(f"B1 improvement: {improvement_b1:.2f}x, B2 improvement: {improvement_b2:.2f}x")

    # ============================================================
    # 方向 C: 维度子集分析
    # ============================================================
    print("\n=== 方向 C: 维度子集分析 ===")
    z_np = val_z.cpu().numpy()
    print("Computing per-dim domain discrimination (256 dims)...")
    js_per_dim = per_dim_domain_discrimination(z_np, labels)
    print(f"Per-dim JS: max={js_per_dim.max():.4f}, min={js_per_dim.min():.4f}, "
          f"mean={js_per_dim.mean():.4f}, std={js_per_dim.std():.4f}")

    # 排序
    sorted_dims = np.argsort(js_per_dim)[::-1]
    top_8_dims = sorted_dims[:8].tolist()
    top_16_dims = sorted_dims[:16].tolist()
    top_32_dims = sorted_dims[:32].tolist()

    print(f"\nTop-8 dims by per-dim JS: {top_8_dims}")
    print(f"Top-16 dims: {top_16_dims}")
    print(f"Top-32 dims: {top_32_dims}")

    # 累计 JS 占比
    cum_js_total = js_per_dim.sum()
    cum_js_top_8 = js_per_dim[sorted_dims[:8]].sum() / cum_js_total if cum_js_total > 0 else 0
    cum_js_top_16 = js_per_dim[sorted_dims[:16]].sum() / cum_js_total if cum_js_total > 0 else 0
    cum_js_top_32 = js_per_dim[sorted_dims[:32]].sum() / cum_js_total if cum_js_total > 0 else 0
    cum_js_top_64 = js_per_dim[sorted_dims[:64]].sum() / cum_js_total if cum_js_total > 0 else 0
    cum_js_top_128 = js_per_dim[sorted_dims[:128]].sum() / cum_js_total if cum_js_total > 0 else 0

    print(f"\nCumulative JS coverage:")
    print(f"  top-8 dims:  {cum_js_top_8*100:.1f}% of total JS")
    print(f"  top-16 dims: {cum_js_top_16*100:.1f}%")
    print(f"  top-32 dims: {cum_js_top_32*100:.1f}%")
    print(f"  top-64 dims: {cum_js_top_64*100:.1f}%")
    print(f"  top-128 dims: {cum_js_top_128*100:.1f}%")

    # 稀疏性判定
    if cum_js_top_32 > 0.5:
        sparsity = "highly sparse (top-32 dims cover >50% JS)"
    elif cum_js_top_64 > 0.5:
        sparsity = "moderately sparse"
    else:
        sparsity = "distributed (JS spread across many dims)"
    print(f"\nz sparsity verdict: {sparsity}")

    # 写 JSON
    report = {
        "experiment": "v39 z refine",
        "n_samples": int(val_z.shape[0]),
        "direction_b_mi": {
            "v38_baseline_6dim": {"mean": v38_baseline, "std": 0.0174, "note": "weak features"},
            "b1_random_embed_64dim": {"mean": float(mi_b1_mean), "std": float(mi_b1_std)},
            "b2_char_trigrams_64": {"mean": float(mi_b2_mean), "std": float(mi_b2_std)},
            "improvement_b1_x": float(improvement_b1),
            "improvement_b2_x": float(improvement_b2),
            "interpretation": (
                "如果 B1/B2 MI 显著提升 (>0.5), 说明 z 与文本内容强相关, v38 MI=0.06 是特征太弱"
                "如果 B1/B2 MI 仍 <0.2, 说明 z 真的与文本内容弱相关, decoder 不消费 z 是合理的"
            )
        },
        "direction_c_dim_analysis": {
            "per_dim_js_max": float(js_per_dim.max()),
            "per_dim_js_min": float(js_per_dim.min()),
            "per_dim_js_mean": float(js_per_dim.mean()),
            "per_dim_js_std": float(js_per_dim.std()),
            "top_8_dims": top_8_dims,
            "top_16_dims": top_16_dims,
            "top_32_dims": top_32_dims,
            "cum_js_top_8": float(cum_js_top_8),
            "cum_js_top_16": float(cum_js_top_16),
            "cum_js_top_32": float(cum_js_top_32),
            "cum_js_top_64": float(cum_js_top_64),
            "cum_js_top_128": float(cum_js_top_128),
            "sparsity_verdict": sparsity,
            "interpretation": (
                "如果 sparsity=highly sparse, z 是 sparse-coding-like, 可能值得做 top-K 注入 (只注入 top-K dims)"
                "如果 sparsity=distributed, z 信息均匀分布, 整体注入更合理 (但 v37 已证 decoder 不消费)"
            )
        },
        "decisions_recommendation": {
            "if_mi_high_and_sparse": "走 block-diffusion PoC with top-K dim injection",
            "if_mi_high_and_distributed": "走 block-diffusion PoC with full z injection",
            "if_mi_low_and_sparse": "走修 z (encoder 微调) + top-K dim 注入",
            "if_mi_low_and_distributed": "战略重定位, 放弃 z 路径 (与 v37 一致)"
        }
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
