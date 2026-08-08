"""
Test TF32 mode: 一行 torch.set_float32_matmul_precision('high') 让 FP32 bmm 走 TC
不用 BF16 cast, 不动 W/S 内存, 直接让 5090 的 5th-gen TC 处理 FP32 bmm (TF32 精度)
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import importlib.util
import statistics
import torch

# 关键: 切到 TF32 (FP32 matmul on TC, peak 250 TFLOPS, 跟 BF16 一样)
torch.set_float32_matmul_precision('high')  # 'high' = TF32 (10-bit mantissa), 'medium' = BF16

spec = importlib.util.spec_from_file_location(
    "train_engine", "D:/CrystaLLM/experiments/spike_pool/train_engine.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["train_engine"] = mod
spec.loader.exec_module(mod)

# 在 __init__ 之前 patch 一下 W/S 用 FP32 cache (绕开 BF16 路径, 直接看 TF32)
RTX5090SpikePool = mod.RTX5090SpikePool
CONFIG = dict(mod.CONFIG)


def run(T, n=30, warmup=10):
    cfg = dict(CONFIG)
    cfg['T_total'] = n + warmup + 50
    pool = RTX5090SpikePool(cfg)
    torch.manual_seed(0)
    targets = [torch.randn(T, cfg['d_model'], device='cuda') for _ in range(cfg['T_total'])]

    for _ in range(warmup):
        pool.forward_step(targets[0])

    times = []
    for i in range(n):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        pool.forward_step(targets[warmup + i])
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))

    K = cfg['Top_K_Active']
    d_in = cfg['d_inner']
    flops = 2 * K * d_in * d_in * T
    med = statistics.median(times)
    achieved = (flops / 1e12) / (med / 1000.0)
    print(f"  T={T:5d} (TF32 + BF16 caches): median={med:7.3f}ms  "
          f"achieved={achieved:7.2f} TFLOPS ({achieved/250*100:5.1f}% BF16 peak)")
    pool.shutdown()


if __name__ == "__main__":
    print("=" * 80)
    print("  TF32 mode: torch.set_float32_matmul_precision('high')")
    print("  (W/S 还是走 BF16 cache 路径, 但 bmm 自身可能 fall back 到 TF32)")
    print("=" * 80)
    for T in [64, 512, 1024, 4096]:
        run(T)
