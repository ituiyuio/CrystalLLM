"""
Tensor Core utilization baseline / measurement harness.

不依赖 `train_engine.py` 的 T_total / dataloader，单文件跑小步数（默认 50），
每步用 torch.cuda.Event 精确计时，最后报告：
  - 平均/中位 step time (ms)
  - 估算有效 BF16 TFLOPS（5090 峰值 ~250）
  - 估算张量核心利用率（effective TFLOPS / 理论峰值）

用法:
  python -u _util_smoke.py            # 默认 50 步
  python -u _util_smoke.py 200        # 200 步

未来对比：同一脚本跑不同 forward_step 实现，比较 step time / TFLOPS。
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import time
import statistics
import torch
import importlib.util

# 加载 train_engine.py（不执行 __main__）
spec = importlib.util.spec_from_file_location(
    "train_engine", "D:/CrystaLLM/experiments/spike_pool/train_engine.py"
)
mod = importlib.util.module_from_spec(spec)
# 防 __main__ 触发：临时把 __name__ 改掉
sys.modules["train_engine"] = mod
spec.loader.exec_module(mod)

RTX5090SpikePool = mod.RTX5090SpikePool
CONFIG = dict(mod.CONFIG)


def run_measurement(n_steps: int = 50, warmup: int = 5, T: int = 1):
    # 缩窗口：只跑 n_steps + warmup 步
    cfg = dict(CONFIG)
    cfg['T_total'] = n_steps + warmup + 50   # 给 finalization 留余量（不调）
    pool = RTX5090SpikePool(cfg)

    # === v4: 序列批维度 ===
    # T=1 → 1D target [d_model] (旧路径)
    # T>1 → 2D target [T, d_model] (新 bmm GEMM 路径)
    torch.manual_seed(0)
    if T == 1:
        targets = [torch.randn(cfg['d_model'], device='cuda') for _ in range(cfg['T_total'])]
    else:
        targets = [torch.randn(T, cfg['d_model'], device='cuda') for _ in range(cfg['T_total'])]

    # 跳过 warmup
    for _ in range(warmup):
        pool.forward_step(targets[0])

    # 计时
    times_ms = []
    for i in range(n_steps):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        pool.forward_step(targets[warmup + i])
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))

    mean = statistics.mean(times_ms)
    median = statistics.median(times_ms)
    stdev = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    p10 = sorted(times_ms)[len(times_ms) // 10]
    p90 = sorted(times_ms)[len(times_ms) * 9 // 10]

    # 估算有效 FLOPs (bmm 主算: [K, d_in, d_m] @ [K, d_m, T])
    K = cfg['Top_K_Active']
    d_in = cfg['d_inner']
    d_m = cfg['d_model']
    flops_per_step = 2 * K * d_in * d_m * T
    tflops_per_sec = (flops_per_step / 1e12) / (mean / 1000.0)

    PEAK_BF16_TFLOPS = 250.0   # 5090 BF16 dense peak (公开 spec)
    util_est = tflops_per_sec / PEAK_BF16_TFLOPS * 100.0

    print("=" * 60)
    print(f"  Tensor Core Utilization (T={T}, {n_steps} steps)")
    print("=" * 60)
    print(f"  step time (ms): mean={mean:7.3f}  median={median:7.3f}  "
          f"std={stdev:6.3f}  p10={p10:7.3f}  p90={p90:7.3f}")
    print(f"  GEMM shape:     [{K}, {d_in}, {d_m}] @ [{K}, {d_m}, {T}]")
    print(f"  FLOPs/step:     {flops_per_step / 1e9:7.3f} G")
    print(f"  achieved:       {tflops_per_sec:7.3f} TFLOPS  (FP32 path)")
    print(f"  vs BF16 peak:   {util_est:6.2f} %   (vs {PEAK_BF16_TFLOPS} TFLOPS)")
    print("=" * 60)
    if T == 1:
        print(f"  Note: T=1 走 GEMV (P=1), Tensor Core 大量空闲.")
        print(f"        改用 T>1 触发真 GEMM, 升级到 BF16 才能上 80%+.")
    else:
        if util_est < 50:
            print(f"  Note: T={T} 已升级到真 GEMM, 但仍是 FP32 → 普通 cores.")
            print(f"        下一步: 升级到 BF16, 让 bmm 走 5th-gen Tensor Core.")
        else:
            print(f"  ✓ T={T} + FP32 已达 {util_est:.1f}% 估算 TC 利用率")
    print("=" * 60)

    pool.shutdown()


if __name__ == "__main__":
    # 用法: _util_smoke.py [n_steps] [T]
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    run_measurement(n, T=T)
