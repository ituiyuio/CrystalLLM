"""
v38 z 健康度诊断 - 4 个独立指标
"""
import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path

V38_DIR = Path(__file__).resolve().parents[1]
DATA = Path("D:/CrystaLLM/crystalllm/data/processed")


# ============================================================
# 指标 1: KL 散度
# ============================================================
def compute_kl(z: torch.Tensor) -> float:
    """
    假设 q(z|x) ~ N(mu_x, sigma_x^2 I)
    KL(q || N(0, I)) per sample, summed over dims, mean over N
    z: (N, D) tensor
    return: KL in nats (scalar)
    """
    mu = z.mean(dim=0)            # (D,)
    sigma = z.std(dim=0)          # (D,)
    # KL(N(mu, sigma^2) || N(0, 1))
    # = 0.5 * sum(mu^2 + sigma^2 - 1 - log(sigma^2))
    kl_per_dim = 0.5 * (mu ** 2 + sigma ** 2 - 1 - torch.log(sigma ** 2 + 1e-8))
    return kl_per_dim.sum().item()


# ============================================================
# 指标 2: MINE 互信息下界
# ============================================================
class MINE(nn.Module):
    def __init__(self, x_dim: int, z_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # joint: x_i with z_i
        joint = self.net(torch.cat([x, z], dim=-1)).mean()
        # marginal: x_i with shuffled z
        z_perm = z[torch.randperm(B)]
        marginal = torch.log(torch.exp(self.net(torch.cat([x, z_perm], dim=-1))).mean() + 1e-8)
        return joint - marginal


def compute_mi_lower_bound(x_emb: torch.Tensor, z: torch.Tensor,
                            n_steps: int = 1000, n_runs: int = 5):
    """
    MINE 估计 I(z; x_emb) 下界
    x_emb: (N, hidden) - text embedding
    z: (N, D)
    return: (mean, std) over n_runs
    """
    estimates = []
    device = x_emb.device
    for run in range(n_runs):
        torch.manual_seed(42 + run)
        mine = MINE(x_emb.shape[1], z.shape[1]).to(device)
        opt = torch.optim.Adam(mine.parameters(), lr=1e-3)
        for step in range(n_steps):
            mi_est = mine(x_emb, z)
            loss = -mi_est
            opt.zero_grad(); loss.backward(); opt.step()
        estimates.append(mine(x_emb, z).item())
    return float(np.mean(estimates)), float(np.std(estimates))


# ============================================================
# 指标 3: 维度塌缩比例
# ============================================================
def compute_collapse_ratio(z: torch.Tensor, threshold: float = 0.01) -> float:
    """
    z: (N, D)
    return: std < threshold 的维度比例
    """
    std_per_dim = z.std(dim=0)
    return (std_per_dim < threshold).float().mean().item()


# ============================================================
# 指标 4: JS 类别可分性
# ============================================================
def js_divergence_gaussian(mu1, cov1, mu2, cov2):
    """
    两个高斯的 JS 散度解析式 (使用 mid-point mixture)
    """
    D = len(mu1)
    mu_mid = 0.5 * (mu1 + mu2)
    cov_mid = 0.5 * (cov1 + cov2)
    eps = 1e-6
    cov_mid_reg = cov_mid + eps * np.eye(D)
    inv_cov_mid = np.linalg.inv(cov_mid_reg)

    def kl_gaussian(mu, cov):
        diff = mu_mid - mu
        return 0.5 * (np.trace(inv_cov_mid @ cov)
                      + diff @ inv_cov_mid @ diff
                      - D
                      + np.log(np.linalg.det(cov_mid_reg) + 1e-12)
                      - np.log(np.linalg.det(cov + eps * np.eye(D)) + 1e-12))

    return 0.5 * (kl_gaussian(mu1, cov1) + kl_gaussian(mu2, cov2))


def compute_class_separability(z: np.ndarray, labels: np.ndarray, n_pca: int = 32) -> float:
    """
    按类别分组, 计算组间 JS 散度均值
    z: (N, D) numpy
    labels: (N,) 类别索引
    return: mean JS over all pairs (≥ 0)
    """
    from sklearn.decomposition import PCA
    D = min(n_pca, z.shape[1])
    z_pca = PCA(n_components=D).fit_transform(z)

    classes = np.unique(labels)
    if len(classes) < 2:
        return 0.0

    class_stats = {}
    for c in classes:
        z_c = z_pca[labels == c]
        if len(z_c) < 10:
            continue
        class_stats[c] = (z_c.mean(axis=0), np.cov(z_c.T) + 1e-6 * np.eye(D))

    if len(class_stats) < 2:
        return 0.0

    js_scores = []
    classes_list = list(class_stats.keys())
    for i, c1 in enumerate(classes_list):
        for c2 in classes_list[i + 1:]:
            js = js_divergence_gaussian(class_stats[c1][0], class_stats[c1][1],
                                         class_stats[c2][0], class_stats[c2][1])
            js_scores.append(max(0.0, js))  # 截断负值
    return float(np.mean(js_scores))


# ============================================================
# 类别标注启发式
# ============================================================
def classify_text(text: str) -> str:
    """
    启发式分类: code / comment / dialog / plain
    """
    # code: 包含常见代码关键字
    if any(kw in text for kw in ["def ", "class ", "import ", "function ", "return ", "if __name__"]):
        return "code"
    # dialog: 包含对话标记
    if any(kw in text for kw in ["User:", "Human:", "Assistant:", "Q:", "A:"]):
        return "dialog"
    # comment: 大量 # 或 // 开头行
    lines = text.split("\n")[:10]
    if len(lines) > 0:
        comment_lines = sum(1 for l in lines if l.strip().startswith(("#", "//")))
        if comment_lines > len(lines) * 0.5:
            return "comment"
    return "plain"


# ============================================================
# v24 encoder + 数据加载
# ============================================================
def load_v24_encoder(device: str = "cuda"):
    """
    加载 v24 encoder.
    v25 和 v36 都消费这个 encoder 输出的 z (cached_v24_z.npz).
    """
    ckpt_path = V38_DIR.parent / "v24" / "v24_encoder.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    return ckpt


def load_val_data():
    """
    加载 val 文本 + v24 cached z
    """
    import pandas as pd
    df_val = pd.read_parquet(DATA / "v24_val.parquet")
    val_texts = df_val["text"].tolist()
    cache = np.load(DATA / "cached_v24_z.npz")
    val_z = cache["val_z"]  # (1016, 256)
    return val_texts, val_z


def encode_val_with_encoder(encoder_ckpt, device: str = "cuda"):
    """
    复用 cached_v24_z.npz 的 val_z (v24 encoder 已训练完成, 输出已 cache)
    不重新 inference, 直接用 cache.
    """
    _, val_z = load_val_data()
    return torch.tensor(val_z, dtype=torch.float32, device=device)


def encode_text_for_mi(val_texts, device: str = "cuda"):
    """
    为 MINE 准备 text features (简化: 长度 + 首字符 one-hot)
    返回: (N, hidden_dim) tensor
    """
    vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
    stoi = vocab["stoi"]
    V = vocab["vocab_size"]

    # 取 vocab 中前 5 个最常见字符作为 one-hot 特征
    common_chars = list(stoi.keys())[:5]

    features = []
    for text in val_texts[:1016]:
        length = min(len(text), 1000) / 1000.0
        first_char = text[0] if text else " "
        one_hot = [1.0 if first_char == c else 0.0 for c in common_chars]
        features.append([length] + one_hot)

    return torch.tensor(features, dtype=torch.float32, device=device)


# ============================================================
# 主函数
# ============================================================
def main():
    """
    v38 z 健康度诊断主函数.
    测量 v24 encoder 的 z 在 4 个维度上的健康度.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_samples", type=int, default=1016)
    parser.add_argument("--mine_steps", type=int, default=1000)
    parser.add_argument("--mine_runs", type=int, default=5)
    parser.add_argument("--output", default=str(V38_DIR / "z_health_report.json"))
    args = parser.parse_args()

    print(f"v38 z health check | device={args.device} | n_samples={args.n_samples}")

    # 加载 encoder + 数据
    encoder_ckpt = load_v24_encoder(args.device)
    val_texts, val_z_np = load_val_data()
    val_z = torch.tensor(val_z_np[:args.n_samples], dtype=torch.float32, device=args.device)
    val_texts = val_texts[:args.n_samples]

    print(f"z shape: {val_z.shape}, mean={val_z.mean():.4f}, std={val_z.std():.4f}")

    # 指标 1: KL
    print("\n--- Metric 1: KL divergence ---")
    kl = compute_kl(val_z)
    print(f"KL: {kl:.4f} nats (healthy < 50, unhealthy > 100)")
    kl_healthy = kl < 50

    # 指标 3: 维度塌缩
    print("\n--- Metric 3: Dimension collapse ---")
    collapse = compute_collapse_ratio(val_z)
    print(f"Collapse ratio: {collapse:.4f} (healthy < 0.5, unhealthy > 0.7)")
    collapse_healthy = collapse < 0.5

    # 指标 4: 类别可分性
    print("\n--- Metric 4: Class separability (JS divergence) ---")
    labels = np.array([hash(classify_text(t)) % 4 for t in val_texts])
    class_counts = {int(c): int((labels == c).sum()) for c in np.unique(labels)}
    print(f"Class distribution: {class_counts}")
    js = compute_class_separability(val_z.cpu().numpy(), labels, n_pca=32)
    print(f"JS (class separability): {js:.4f} nats (healthy > 0.05, unhealthy < 0.02)")
    js_healthy = js > 0.05

    # 指标 2: MINE
    print(f"\n--- Metric 2: MI lower bound (MINE, {args.mine_runs} runs x {args.mine_steps} steps) ---")
    text_emb = encode_text_for_mi(val_texts, args.device)
    mi_mean, mi_std = compute_mi_lower_bound(text_emb, val_z,
                                              n_steps=args.mine_steps,
                                              n_runs=args.mine_runs)
    print(f"MI lower bound: {mi_mean:.4f} +/- {mi_std:.4f} nats (healthy > 0.10, unhealthy < 0.05)")
    mi_healthy = mi_mean > 0.10

    # 决策矩阵
    print("\n--- Decision matrix ---")
    n_healthy = sum([kl_healthy, mi_healthy, collapse_healthy, js_healthy])
    if n_healthy == 4:
        scenario = "A"
        action = "block-diffusion PoC (v39)"
    elif n_healthy >= 2:
        if not kl_healthy and mi_healthy and collapse_healthy and js_healthy:
            scenario = "B"
            action = "修 z (free_bits ++, 或换 encoder)"
        else:
            scenario = "C"
            action = "二次 brainstorm"
    else:
        scenario = "F"
        action = "战略重定位, 放弃 z 路径"

    print(f"Scenario: {scenario} ({n_healthy}/4 healthy)")
    print(f"Action: {action}")

    # 写 JSON
    report = {
        "model": "v24_encoder",
        "n_samples": int(val_z.shape[0]),
        "z_shape": list(val_z.shape),
        "z_mean": float(val_z.mean()),
        "z_std": float(val_z.std()),
        "metrics": {
            "kl_nats": {"value": float(kl), "healthy": bool(kl_healthy),
                        "threshold_healthy": 50, "threshold_unhealthy": 100},
            "mi_lower_bound_nats": {"mean": float(mi_mean), "std": float(mi_std),
                                     "healthy": bool(mi_healthy),
                                     "threshold_healthy": 0.10, "threshold_unhealthy": 0.05},
            "collapse_ratio": {"value": float(collapse), "healthy": bool(collapse_healthy),
                                "threshold_healthy": 0.5, "threshold_unhealthy": 0.7},
            "js_class_separability_nats": {"value": float(js), "healthy": bool(js_healthy),
                                             "threshold_healthy": 0.05, "threshold_unhealthy": 0.02,
                                             "class_distribution": class_counts}
        },
        "decision": {
            "n_healthy": n_healthy,
            "scenario": scenario,
            "action": action
        },
        "note": "v25 和 v36 都使用 v24 encoder 的 z (cached_v24_z.npz), 所以本测量对两者都适用"
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()