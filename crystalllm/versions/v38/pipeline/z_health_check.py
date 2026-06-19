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
    for run in range(n_runs):
        torch.manual_seed(42 + run)
        mine = MINE(x_emb.shape[1], z.shape[1])
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