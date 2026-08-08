"""
Micro-benchmark: 纯 bmm 跑 N 次, 测 GPU 端到端 bmm 时间和 launch overhead。
帮 momo 看清 launch overhead vs 实际 compute 的比例。
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import statistics
import torch

K = 16
d_in = 4096
d_m = 4096


def bench_bmm(T, dtype=torch.bfloat16, N=100, warmup=20):
    W = torch.randn(K, d_in, d_m, dtype=dtype, device='cuda')
    S = torch.randn(K, d_m, T, dtype=dtype, device='cuda')

    for _ in range(warmup):
        deltas = torch.bmm(W, S)
    torch.cuda.synchronize()

    times = []
    for _ in range(N):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        deltas = torch.bmm(W, S)
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))

    median = statistics.median(times)
    flops = 2 * K * d_in * d_m * T
    tflops = flops / 1e12
    achieved = tflops / (median / 1000.0)
    print(f"  bmm T={T:5d} {str(dtype):20s}: median={median:7.3f}ms  "
          f"achieved={achieved:7.2f} TFLOPS ({achieved/250*100:5.1f}% BF16 peak)")
    return median, achieved


if __name__ == "__main__":
    print("=" * 80)
    print("  Pure bmm micro-benchmark (no autograd, no gather, no scale)")
    print("=" * 80)
    for T in [1, 8, 64, 128, 512, 1024, 2048, 4096]:
        bench_bmm(T)
    print()
    print("对比: 实际 forward_step T=4096 step time = ~6ms median")
    print("      如果纯 bmm < 1ms, 那 5ms 是其他 (gather + launch overhead)")
