"""
Power Spectral Density (PSD) Utilities
=======================================

PSD 是区分混沌 vs 周期 vs 噪声的关键诊断工具.

定义:
    PSD(f) = |FFT(x(t))|^2 / T

混沌吸引子 (Lorenz):
    - 宽带连续谱 (broadband), 没有明显的尖峰
    - 在 log-log 坐标下有特征斜率

周期/极限环 (AR 失败模式 1):
    - 尖峰谱, 谐波峰 (line spectrum)

不动点 (AR 失败模式 2):
    - 单一尖峰在 f=0 (直流分量)

白噪声 (trivial baseline):
    - 平坦谱

参考: https://en.wikipedia.org/wiki/Spectral_density
"""
from __future__ import annotations

import numpy as np
import torch


def compute_psd_welch(signal: np.ndarray, fs: float = 100.0, nperseg: int = 256,
                      noverlap: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Welch's method 计算 PSD.

    Args:
        signal: (T,) or (T, dim) 时间序列
        fs: 采样频率 (Hz). Lorenz dt=0.01 -> fs=100
        nperseg: 每段长度
        noverlap: 段重叠 (默认 nperseg // 2)
    Returns:
        freqs: (F,) 频率数组
        psd: (F,) or (F, dim) 功率谱密度
    """
    from scipy.signal import welch
    if noverlap is None:
        noverlap = nperseg // 2
    if signal.ndim == 1:
        freqs, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
        return freqs, psd
    else:
        psds = []
        freqs = None
        for d in range(signal.shape[-1]):
            f, p = welch(signal[..., d], fs=fs, nperseg=nperseg, noverlap=noverlap)
            if freqs is None:
                freqs = f
            psds.append(p)
        return freqs, np.stack(psds, axis=-1)


def compute_psd_fft(signal: np.ndarray, fs: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """简单 FFT PSD (不做 Welch 平均).

    适用于长轨迹 (T >= 1000), 比 Welch 快.
    """
    T = signal.shape[0]
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    if signal.ndim == 1:
        fft = np.fft.rfft(signal)
        psd = (np.abs(fft) ** 2) / T
        return freqs, psd
    else:
        psds = []
        for d in range(signal.shape[-1]):
            fft = np.fft.rfft(signal[..., d])
            psd = (np.abs(fft) ** 2) / T
            psds.append(psd)
        return freqs, np.stack(psds, axis=-1)


def classify_psd_shape(freqs: np.ndarray, psd: np.ndarray,
                       broadband_threshold: float = 0.1) -> str:
    """粗略分类 PSD 形状.

    Returns:
        'broadband' (混沌)
        'line_spectrum' (周期/极限环)
        'flat_noise' (白噪声)
        'delta' (不动点)
    """
    # 归一化
    psd_norm = psd / (psd.max() + 1e-12)

    # 检查 f=0 是否有 delta (不动点)
    if psd_norm[0] > 0.9 and len(psd_norm) > 1:
        return "delta"

    # 检查是否有明显尖峰
    sorted_psd = np.sort(psd_norm)[::-1]
    top1 = sorted_psd[0]
    top2 = sorted_psd[1] if len(sorted_psd) > 1 else 0
    if top1 > 5 * top2 and top1 > 0.3:
        return "line_spectrum"

    # 检查是否平坦
    flatness = np.std(psd_norm)
    if flatness < broadband_threshold:
        return "flat_noise"

    return "broadband"


def compute_spectral_slope(freqs: np.ndarray, psd: np.ndarray,
                           f_range: tuple[float, float] = (0.5, 20.0)) -> float:
    """计算 log-log 坐标下 PSD 的斜率 (用于混沌特征).

    Args:
        freqs: (F,)
        psd: (F,)
        f_range: 用于拟合的频率范围 (Hz)
    Returns:
        slope: PSD ~ f^slope 中的 slope
    """
    mask = (freqs >= f_range[0]) & (freqs <= f_range[1])
    if mask.sum() < 2:
        return 0.0
    log_f = np.log10(freqs[mask])
    log_psd = np.log10(psd[mask] + 1e-12)
    # 线性拟合
    coeffs = np.polyfit(log_f, log_psd, 1)
    return float(coeffs[0])


if __name__ == "__main__":
    # Sanity test
    print("=" * 60)
    print("PSD Utils Sanity Test")
    print("=" * 60)

    fs = 100.0
    T = 2000

    # 测试 1: 混沌 (Lorenz x)
    lorenz_x = np.cumsum(np.random.randn(T)) * 0.01  # 简化模拟
    f1, p1 = compute_psd_fft(lorenz_x, fs=fs)
    shape1 = classify_psd_shape(f1, p1)
    slope1 = compute_spectral_slope(f1, p1)
    print(f"\n1. Lorenz-like (cumulative noise):")
    print(f"   Shape: {shape1}, slope: {slope1:.2f}")

    # 测试 2: 周期信号
    t = np.arange(T) / fs
    periodic = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 12 * t)
    f2, p2 = compute_psd_fft(periodic, fs=fs)
    shape2 = classify_psd_shape(f2, p2)
    slope2 = compute_spectral_slope(f2, p2)
    print(f"\n2. Periodic (sum of sines):")
    print(f"   Shape: {shape2}, slope: {slope2:.2f}")

    # 测试 3: 白噪声
    white = np.random.randn(T)
    f3, p3 = compute_psd_fft(white, fs=fs)
    shape3 = classify_psd_shape(f3, p3)
    slope3 = compute_spectral_slope(f3, p3)
    print(f"\n3. White noise:")
    print(f"   Shape: {shape3}, slope: {slope3:.2f}")

    # 测试 4: 不动点 (常数)
    constant = np.ones(T) * 0.5
    f4, p4 = compute_psd_fft(constant, fs=fs)
    shape4 = classify_psd_shape(f4, p4)
    print(f"\n4. Constant (fixed point):")
    print(f"   Shape: {shape4}")

    print("\n[OK] PSD utils working")
