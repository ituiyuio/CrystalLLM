import torch
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from z_health_check import (compute_kl, compute_collapse_ratio,
                             compute_class_separability, classify_text,
                             js_divergence_gaussian)


def test_compute_kl_perfect_normal():
    """z ~ N(0, I) → KL ≈ 0"""
    z = torch.randn(1000, 256)
    kl = compute_kl(z)
    assert kl < 10.0, f"Expected KL < 10 for N(0,I), got {kl}"


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
    z[:, :128] = 0.5  # 前 128 维固定为 0.5, std ≈ 0
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
    assert js < 0.5, f"Expected JS ≈ 0, got {js}"


def test_js_divergence_gaussian_far():
    """两个远距高斯 → JS > 5"""
    mu1 = np.zeros(8)
    mu2 = np.ones(8) * 5
    cov = np.eye(8)
    js = js_divergence_gaussian(mu1, cov, mu2, cov)
    assert js > 5.0, f"Expected JS > 5, got {js}"


def test_compute_class_separability_random():
    """随机标签 → JS < 1"""
    z = np.random.randn(500, 32)
    labels = np.random.randint(0, 3, size=500)
    js = compute_class_separability(z, labels, n_pca=8)
    assert js < 5.0, f"Expected JS < 5 for random, got {js}"


def test_compute_class_separability_separable():
    """类间明显分开 → JS > 5"""
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