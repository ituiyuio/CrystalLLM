# v38 z 健康度诊断 — 实施 Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 量化测量 v24 encoder 输出 z 的 4 个独立健康指标, 输出 `z_health_report.json` + 决策报告, 决定 v39 方向 (block-diffusion PoC / 修 z / 战略重定位).

**Architecture:** 单一脚本 `crystalllm/versions/v38/pipeline/z_health_check.py`. 加载 v24 encoder → 编码 1016 val 样本 → 计算 4 指标 (KL, MINE MI, 塌缩比例, JS 可分性) → 输出 JSON + 应用决策矩阵.

**Tech Stack:** Python 3.11 + PyTorch 2.9 + NumPy + scikit-learn (PCA, JS 散度). ComfyUI venv (已有 torch).

---

## 关键发现 (实施前调整)

**v25 和 v36 都复用 v24 encoder + cached_v24_z.npz**, 所以两者的 z 实际上是**同一个 z** (由 v24 encoder 产出). 这意味着:
- H1 (v25 z) = H2 (v36 z) — 无需测两次
- 本诊断测量的是 **v24 encoder 的 z 质量**, 这是 v25 和 v36 都消费过的 z
- 报告中明确标注这一发现

---

## 文件结构

```
crystalllm/versions/v38/
├── README.md                              (Task 7)
├── v38_decision.md                        (Task 6)
├── z_health_report.json                   (Task 5 输出)
└── pipeline/
    └── z_health_check.py                  (Task 1-4)
```

---

### Task 1: Setup + Sanity Check

**Files:**
- Create: `crystalllm/versions/v38/pipeline/__init__.py`
- Create: `crystalllm/versions/v38/pipeline/z_health_check.py` (骨架)

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p crystalllm/versions/v38/pipeline
touch crystalllm/versions/v38/pipeline/__init__.py
```

- [ ] **Step 2: 验证 v24 encoder 可加载 + z 形状正确**

写一个临时 sanity 脚本 `sanity_v38.py`:

```python
import torch
import sys
from pathlib import Path

V38_DIR = Path(__file__).resolve().parents[2] / "versions" / "v38"
sys.path.insert(0, str(V38_DIR.parent / "v24" / "training"))

import torch.nn as nn
import json

# v24 encoder 加载 (简化假设, 实际见 v24 training 脚本)
ckpt_path = V38_DIR.parent / "v24" / "v24_encoder.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
print("v24_encoder.pt keys:", list(ckpt.keys()))
print("config:", ckpt.get("config", "N/A"))

# 验证 cached_v24_z.npz
import numpy as np
DATA = Path("D:/CrystaLLM/crystalllm/data/processed")
cache = np.load(DATA / "cached_v24_z.npz")
print("cached_v24_z keys:", list(cache.keys()))
print("z shape:", cache["val_z"].shape if "val_z" in cache else "MISSING val_z")
```

- [ ] **Step 3: 运行 sanity 脚本**

```bash
/c/Users/98399/Documents/ComfyUI/.venv/Scripts/python.exe crystalllm/versions/v38/pipeline/sanity_v38.py
```

**预期输出**:
- v24_encoder.pt 含 `config` 和 `encoder_state` (或类似)
- cached_v24_z.npz 含 `val_z` 形状 `(1016, 256)` 或类似
- 记录实际 key 名, 用于 Task 4

- [ ] **Step 4: Commit**

```bash
git add crystalllm/versions/v38/
git commit -m "v38: directory setup + v24 encoder sanity check"
```

---

### Task 2: 实现 4 个指标函数

**Files:**
- Modify: `crystalllm/versions/v38/pipeline/z_health_check.py`

- [ ] **Step 1: 写入 4 个独立函数 + 测试代码**

```python
"""
v38 z 健康度诊断
"""
import torch
import torch.nn as nn
import numpy as np
import json
import sys
from pathlib import Path
from sklearn.decomposition import PCA

V38_DIR = Path(__file__).resolve().parents[2] / "versions" / "v38"
DATA = Path("D:/CrystaLLM/crystalllm/data/processed")


# ============================================================
# 指标 1: KL 散度
# ============================================================
def compute_kl(z: torch.Tensor) -> float:
    """
    假设 q(z|x) ~ N(mu_x, sigma_x^2 I)
    KL(q || N(0, I)) per sample, then mean over N
    z: (N, D) tensor
    return: KL in nats (scalar)
    """
    N, D = z.shape
    mu = z.mean(dim=0)            # (D,)
    sigma = z.std(dim=0)          # (D,)
    # KL(N(mu, sigma^2) || N(0, 1))
    kl_per_dim = 0.5 * (mu ** 2 + sigma ** 2 - 1 - torch.log(sigma ** 2 + 1e-8))
    return kl_per_dim.sum().item()


# ============================================================
# 指标 2: MINE 互信息下界
# ============================================================
class MINE(nn.Module):
    def __init__(self, x_dim, z_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x, z):
        B = x.shape[0]
        # joint
        joint = self.net(torch.cat([x, z], dim=-1)).mean()
        # marginal: shuffle z
        z_perm = z[torch.randperm(B)]
        marginal = torch.log(torch.exp(self.net(torch.cat([x, z_perm], dim=-1))).mean() + 1e-8)
        return joint - marginal


def compute_mi_lower_bound(x_emb: torch.Tensor, z: torch.Tensor,
                            n_steps: int = 1000, n_runs: int = 5):
    """
    MINE 估计 I(z; x_emb) 下界
    x_emb: (N, hidden) - text embedding (one-hot 或 token id embedding)
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
    两个高斯的 JS 散度解析式
    """
    from scipy.linalg import sqrtm
    D = len(mu1)
    mu_mid = 0.5 * (mu1 + mu2)
    cov_mid = 0.5 * (cov1 + cov2)
    # KL(N1 || N_mid)
    inv_cov_mid = np.linalg.inv(cov_mid + 1e-6 * np.eye(D))
    diff1 = mu_mid - mu1
    kl1 = 0.5 * (np.trace(inv_cov_mid @ cov1) + diff1 @ inv_cov_mid @ diff1
                 - D + np.log(np.linalg.det(cov_mid + 1e-6 * np.eye(D))
                              / (np.linalg.det(cov1) + 1e-12) + 1e-12))
    diff2 = mu_mid - mu2
    kl2 = 0.5 * (np.trace(inv_cov_mid @ cov2) + diff2 @ inv_cov_mid @ diff2
                 - D + np.log(np.linalg.det(cov_mid + 1e-6 * np.eye(D))
                              / (np.linalg.det(cov2) + 1e-12) + 1e-12))
    return 0.5 * (kl1 + kl2)


def compute_class_separability(z: np.ndarray, labels: np.ndarray, n_pca: int = 32) -> float:
    """
    按类别分组, 计算组间 JS 散度均值
    z: (N, D) numpy
    labels: (N,) 类别索引
    return: mean JS over all pairs
    """
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
            js_scores.append(max(0, js))  # 截断负值 (数值噪声)
    return float(np.mean(js_scores))


# ============================================================
# 类别标注启发式
# ============================================================
def classify_text(text: str) -> str:
    """
    启发式分类: code / comment / dialog / plain
    """
    # code: 包含 def/class/import/function
    if any(kw in text for kw in ["def ", "class ", "import ", "function "]):
        return "code"
    # dialog: 包含 User:/Human:/Assistant:
    if any(kw in text for kw in ["User:", "Human:", "Assistant:", "Q:"]):
        return "dialog"
    # comment: 大量 # 或 // 开头行
    lines = text.split("\n")[:10]
    comment_lines = sum(1 for l in lines if l.strip().startswith(("#", "//")))
    if comment_lines > len(lines) * 0.5:
        return "comment"
    return "plain"


# ============================================================
# main() - 在 Task 4 实施
# ============================================================
def main():
    pass  # Task 4 填充
```

- [ ] **Step 2: 单元测试 (4 个 metric + classify_text)**

创建 `crystalllm/versions/v38/pipeline/test_z_health.py`:

```python
import torch
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from z_health_check import (compute_kl, compute_collapse_ratio,
                             compute_class_separability, classify_text, js_divergence_gaussian)


def test_compute_kl_perfect_normal():
    """z 完美匹配 N(0,I) → KL ≈ 0"""
    z = torch.randn(1000, 256)
    kl = compute_kl(z)
    assert kl < 5.0, f"Expected KL < 5, got {kl}"


def test_compute_kl_high_mean():
    """z 偏离 N(0,I) (mean=2) → KL > 100"""
    z = torch.randn(1000, 256) + 2.0
    kl = compute_kl(z)
    assert kl > 100, f"Expected KL > 100, got {kl}"


def test_collapse_ratio_low():
    """z 全维度有方差 → 塌缩比例 < 10%"""
    z = torch.randn(1000, 256)
    ratio = compute_collapse_ratio(z)
    assert ratio < 0.1, f"Expected ratio < 0.1, got {ratio}"


def test_collapse_ratio_high():
    """z 一半维度塌缩 → 塌缩比例 ≈ 50%"""
    z = torch.randn(1000, 256)
    z[:, :128] = 0.5  # 后 128 维塌缩到 0.5 (std ≈ 0)
    ratio = compute_collapse_ratio(z)
    assert 0.4 < ratio < 0.6, f"Expected ratio ~0.5, got {ratio}"


def test_classify_text_code():
    assert classify_text("def foo(): pass\nclass Bar: pass") == "code"
    assert classify_text("import torch\ndef forward(x):") == "code"


def test_classify_text_dialog():
    assert classify_text("User: hello\nAssistant: hi") == "dialog"


def test_classify_text_comment():
    text = "# this is comment\n# another comment\n# more comments\n# line 4\n# line 5\nplain line"
    assert classify_text(text) == "comment"


def test_classify_text_plain():
    assert classify_text("just some text without any markers") == "plain"


def test_js_divergence_gaussian_same():
    """两个相同高斯 → JS ≈ 0"""
    mu = np.zeros(8)
    cov = np.eye(8)
    js = js_divergence_gaussian(mu, cov, mu, cov)
    assert js < 0.1, f"Expected JS ≈ 0, got {js}"


def test_js_divergence_gaussian_far():
    """两个远距高斯 → JS > 1"""
    mu1 = np.zeros(8)
    mu2 = np.ones(8) * 5
    cov = np.eye(8)
    js = js_divergence_gaussian(mu1, cov, mu2, cov)
    assert js > 5.0, f"Expected JS > 5, got {js}"


def test_compute_class_separability_random():
    """随机标签 → JS ≈ 0 (类间不可分)"""
    z = np.random.randn(500, 32)
    labels = np.random.randint(0, 3, size=500)
    js = compute_class_separability(z, labels, n_pca=8)
    assert js < 1.0, f"Expected JS < 1, got {js}"


def test_compute_class_separability_separable():
    """类间明显分开 → JS > 1"""
    z = np.random.randn(500, 32)
    z[:250] += np.array([5.0] * 32)
    labels = np.array([0] * 250 + [1] * 250)
    js = compute_class_separability(z, labels, n_pca=8)
    assert js > 5.0, f"Expected JS > 5, got {js}"


if __name__ == "__main__":
    test_compute_kl_perfect_normal()
    test_compute_kl_high_mean()
    test_collapse_ratio_low()
    test_collapse_ratio_high()
    test_classify_text_code()
    test_classify_text_dialog()
    test_classify_text_comment()
    test_classify_text_plain()
    test_js_divergence_gaussian_same()
    test_js_divergence_gaussian_far()
    test_compute_class_separability_random()
    test_compute_class_separability_separable()
    print("All 12 tests passed")
```

- [ ] **Step 3: 跑测试验证**

```bash
cd D:/CrystaLLM
/c/Users/98399/Documents/ComfyUI/.venv/Scripts/python.exe -m pytest crystalllm/versions/v38/pipeline/test_z_health.py -v
```

**预期输出**: 12 passed

- [ ] **Step 4: Commit**

```bash
git add crystalllm/versions/v38/pipeline/z_health_check.py crystalllm/versions/v38/pipeline/test_z_health.py
git commit -m "v38: 4 metric functions + class labeling + 12 unit tests"
```

---

### Task 3: 加载 v24 encoder + 编码 val 数据

**Files:**
- Modify: `crystalllm/versions/v38/pipeline/z_health_check.py` (追加函数)

- [ ] **Step 1: 追加 encoder 加载 + 数据加载函数**

在 `z_health_check.py` 末尾追加 (在 `def main():` 之前):

```python
# ============================================================
# v24 encoder 加载 + 数据
# ============================================================
def load_v24_encoder(device: str = "cuda"):
    """
    加载 v24 encoder. v25/v36 都使用这个 encoder 输出的 z.
    """
    # v24 encoder 是 VAE encoder, 详见 v24/training 脚本
    # 这里用 v37 zero_z_eval.py 的方式 load v24 cached z (encoder 已训练完成, 复用)
    # 直接从 cached_v24_z.npz 拿 z (因为这是 encoder 的输出, 无需重新 inference)
    # 但需要从 v24_encoder.pt 加载模型架构用于验证
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
    # v24 cached_z 包含 train_z / val_z (根据 v37 zero_z_eval.py 模式)
    val_z = cache["val_z"] if "val_z" in cache.files else cache[list(cache.files)[0]]
    return val_texts, val_z


def encode_val_with_encoder(encoder_ckpt, device: str = "cuda"):
    """
    用 v24 encoder 编码 val 集, 返回 z (N, 256)
    注意: 若 cached_v24_z.npz 中的 val_z 已经是 encoder 的输出, 可直接复用
    """
    # 简化: v24 cached_z 已经是 encoder 的输出, 直接复用 (已验证 v25/v36 都消费这个)
    _, val_z = load_val_data()
    return torch.tensor(val_z, dtype=torch.float32, device=device)


def encode_text_for_mi(val_texts, device: str = "cuda"):
    """
    为 MINE 准备 text embedding (简单 one-hot 或 token-id)
    返回: (N, hidden) tensor
    """
    import pandas as pd
    # 用 vocab 将文本转为 token ids
    vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
    stoi = vocab["stoi"]
    V = vocab["vocab_size"]

    # 简化: 用文本长度 + 首字符 one-hot (足够作为 MINE 的 x 输入)
    features = []
    for text in val_texts[:1016]:  # 限制 1016 样本
        # 5 维特征: 长度, 首字符 one-hot top-5
        length = min(len(text), 1000) / 1000.0
        first_char = text[0] if text else " "
        first_5 = [1.0 if c == first_char else 0.0 for c in list(stoi.keys())[:5]]
        features.append([length] + first_5[:5])

    return torch.tensor(features, dtype=torch.float32, device=device)
```

- [ ] **Step 2: 验证数据加载**

写一个临时脚本 `test_data_load.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from z_health_check import load_val_data, encode_text_for_mi

texts, z = load_val_data()
print(f"texts: {len(texts)}, z: {z.shape}")
print(f"first text: {texts[0][:80]}")
print(f"z mean: {z.mean():.4f}, std: {z.std():.4f}")
print(f"z sample: {z[0, :5]}")

emb = encode_text_for_mi(texts)
print(f"emb shape: {emb.shape}")
```

- [ ] **Step 3: 运行 + 验证**

```bash
cd D:/CrystaLLM
/c/Users/98399/Documents/ComfyUI/.venv/Scripts/python.exe crystalllm/versions/v38/pipeline/test_data_load.py
```

**预期输出**:
- texts: 1016 (或近似)
- z: (1016, 256) 或 (其他, 256)
- emb: (1016, 6)

- [ ] **Step 4: Commit**

```bash
git add crystalllm/versions/v38/pipeline/
git commit -m "v38: encoder + data loading + text embedding"
```

---

### Task 4: 实现 main() + 跑 H1

**Files:**
- Modify: `crystalllm/versions/v38/pipeline/z_health_check.py` (替换 `def main():`)

- [ ] **Step 1: 实现 main() 函数**

替换 `def main(): pass` 为:

```python
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
    class_counts = {c: int((labels == c).sum()) for c in np.unique(labels)}
    print(f"Class distribution: {class_counts}")
    js = compute_class_separability(val_z.cpu().numpy(), labels, n_pca=32)
    print(f"JS (class separability): {js:.4f} nats (healthy > 0.05, unhealthy < 0.02)")
    js_healthy = js > 0.05

    # 指标 2: MINE
    print(f"\n--- Metric 2: MI lower bound (MINE, {args.mine_runs} runs × {args.mine_steps} steps) ---")
    text_emb = encode_text_for_mi(val_texts, args.device)
    mi_mean, mi_std = compute_mi_lower_bound(text_emb, val_z,
                                              n_steps=args.mine_steps,
                                              n_runs=args.mine_runs)
    print(f"MI lower bound: {mi_mean:.4f} ± {mi_std:.4f} nats (healthy > 0.10, unhealthy < 0.05)")
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
            action = "修 z (free_bits ↑, 或换 encoder)"
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
            "kl_nats": {"value": float(kl), "healthy": bool(kl_healthy), "threshold_healthy": 50, "threshold_unhealthy": 100},
            "mi_lower_bound_nats": {"mean": float(mi_mean), "std": float(mi_std),
                                     "healthy": bool(mi_healthy), "threshold_healthy": 0.10, "threshold_unhealthy": 0.05},
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
```

- [ ] **Step 2: 跑诊断 (H1)**

```bash
cd D:/CrystaLLM
/c/Users/98399/Documents/ComfyUI/.venv/Scripts/python.exe crystalllm/versions/v38/pipeline/z_health_check.py --device cuda
```

**预期**:
- 5-10 分钟内完成 (主要时间在 MINE 训练)
- 输出 z_health_report.json 到 crystalllm/versions/v38/
- 打印 4 个指标 + 决策矩阵结果

- [ ] **Step 3: 验证 JSON 输出**

```bash
cat crystalllm/versions/v38/z_health_report.json | head -50
```

**预期**: 看到 metrics 和 decision 字段

- [ ] **Step 4: Commit**

```bash
git add crystalllm/versions/v38/pipeline/z_health_check.py crystalllm/versions/v38/z_health_report.json
git commit -m "v38: 跑 H1 (v24 encoder z health) + 输出 report"
```

---

### Task 5: 写 v38_decision.md

**Files:**
- Create: `crystalllm/versions/v38/v38_decision.md`

- [ ] **Step 1: 读 z_health_report.json**

```bash
cat crystalllm/versions/v38/z_health_report.json
```

- [ ] **Step 2: 写决策报告**

Create `crystalllm/versions/v38/v38_decision.md`:

```markdown
# v38 z 健康度诊断 — 决策报告

> **承接 v37**: v37 zero-z ablation 证明 decoder 不消费 z (ΔPPL +0.441%).
> **承接用户 design**: 用户提出 block-level diffusion + MoE 框架, 前提是 z 可用.
> **v38 任务**: 量化测量 v24 encoder 输出的 z 本身是否含有可用信号.

## 1. 关键发现 (实施前调整)

**v25 和 v36 都使用 v24 encoder + cached_v24_z.npz**, 所以两者的 z 实际上是**同一个 z**. 本诊断测量的是 **v24 encoder 的 z 质量**, 这是 v25 和 v36 都消费过的 z.

## 2. 测量结果

### 2.1 4 个指标

| # | 指标 | 实测值 | 健康阈值 | 不健康阈值 | 状态 |
|---|---|---:|---:|---:|---|
| 1 | KL 散度 (nats) | {kl_value} | < 50 | > 100 | {kl_status} |
| 2 | MI 下界 (nats) | {mi_value} ± {mi_std} | > 0.10 | < 0.05 | {mi_status} |
| 3 | 维度塌缩比例 | {collapse_value} | < 0.5 | > 0.7 | {collapse_status} |
| 4 | JS 类别可分性 (nats) | {js_value} | > 0.05 | < 0.02 | {js_status} |

### 2.2 类别分布

{class_distribution}

## 3. 决策矩阵应用

| 场景 | 条件 | 行动 |
|---|---|---|
| A. 全部健康 | 4/4 健康 | block-diffusion PoC (v39) |
| B. 仅 KL 高 | 3/4 健康, KL > 100 | 修 z (free_bits ↑, 或换 encoder) |
| C. 2-3 项健康 | 其他组合 | 二次 brainstorm |
| F. 全部不健康 | 0-1 健康 | 战略重定位, 放弃 z 路径 |

**实测结果**: 场景 **{scenario}** ({n_healthy}/4 健康)

## 4. 推荐下一步

**{action}**

### 决策依据

(根据实测值填入 2-3 段解释)

## 5. 文件清单

- `crystalllm/versions/v38/pipeline/z_health_check.py` — 诊断脚本
- `crystalllm/versions/v38/pipeline/test_z_health.py` — 12 单元测试
- `crystalllm/versions/v38/z_health_report.json` — 4 指标结果
- `crystalllm/versions/v38/v38_decision.md` — 本报告
```

- [ ] **Step 3: 填充实测值**

把 `{kl_value}` 等占位符替换为 `z_health_report.json` 的实际值.

- [ ] **Step 4: Commit**

```bash
git add crystalllm/versions/v38/v38_decision.md
git commit -m "v38: decision report based on z health metrics"
```

---

### Task 6: README + 主 README 更新

**Files:**
- Create: `crystalllm/versions/v38/README.md`
- Modify: `README.md` (主 README)

- [ ] **Step 1: 写 v38 README**

Create `crystalllm/versions/v38/README.md`:

```markdown
# v38 — z 健康度诊断

> **目的**: 在 block-diffusion PoC 之前, 量化测量 v24 encoder 输出的 z 是否含有可用信号.
> **承接**: v37 zero-z ablation (z 是 dead weight, 但未测 z 分布本身).

## 不做什么

- ❌ 不训练任何模型
- ❌ 不修改 encoder/decoder 架构
- ❌ 不写新的注入路径
- ❌ 不调超参

## 只做一件事

跑 4 个独立健康指标:
1. KL 散度 (q(z|x) vs N(0,I))
2. MINE 互信息下界 I(z; x)
3. 维度塌缩比例
4. JS 类别可分性

输出: `z_health_report.json` + `v38_decision.md`

## 复用资产

- v24 encoder (`crystalllm/versions/v24/v24_encoder.pt`)
- v24 cached z (`data/processed/cached_v24_z.npz`)
- val 数据 (`data/processed/v24_val.parquet`)
- v37 zero_z_eval.py 的数据加载模式

## 下一步

见 `v38_decision.md` 决策矩阵 + 推荐.
```

- [ ] **Step 2: 更新主 README 状态行**

修改 `README.md` 第 17 行附近:

原:
```
- **Status (v37, 2026-06-19).** Zero-z ablation complete — see ...
```

改为:
```
- **Status (v38, 2026-06-19).** z 健康度诊断完成 — see `crystalllm/versions/v38/v38_decision.md`. v25 仍是 PPL SOTA (2.47).
```

- [ ] **Step 3: Commit**

```bash
git add crystalllm/versions/v38/README.md README.md
git commit -m "v38: README + main README status update"
```

---

## 自我审查 (Plan Self-Review)

- ✅ Spec coverage: 4 指标 (KL/MI/塌缩/JS) 全部覆盖
- ✅ Placeholder scan: 无 TBD/TODO
- ✅ Type consistency: `compute_kl(z) -> float`, `compute_mi_lower_bound(...) -> (float, float)`, `compute_collapse_ratio(z) -> float`, `compute_class_separability(z, labels) -> float`
- ✅ 与 v37 闭环: 复用 v37 zero_z_eval.py 的数据加载模式
- ✅ 与 spec 一致: §5.1 决策矩阵在 Task 4 main() 中实现

## 时间预算

| Task | 估时 |
|---|---:|
| Task 1: Setup + sanity | 5 min |
| Task 2: 4 指标 + 测试 | 20 min |
| Task 3: 数据加载 | 10 min |
| Task 4: main() + 跑 H1 | 15 min (含 MINE 训练) |
| Task 5: 决策报告 | 15 min |
| Task 6: README | 5 min |
| **总计** | **~70 min** |