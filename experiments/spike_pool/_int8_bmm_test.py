"""
INT8 bmm test: 测 torch.bmm INT8 是否走 TC, 跟 BF16 比
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import statistics
import torch

K, d_in, d_m = 16, 4096, 4096

print("=" * 80)
print("  INT8 bmm test (vs BF16 bmm)")
print("=" * 80)

for T in [1, 64, 128, 512, 1024, 4096]:
    # INT8 inputs
    W_int8 = torch.randint(-128, 127, (K, d_in, d_m), dtype=torch.int8, device='cuda')
    S_int8 = torch.randint(-128, 127, (K, d_m, T), dtype=torch.int8, device='cuda')
    try:
        # warmup
        for _ in range(5):
            out = torch.bmm(W_int8, S_int8)
        torch.cuda.synchronize()
        # time
        times = []
        for _ in range(30):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = torch.bmm(W_int8, S_int8)
            e.record()
            e.synchronize()
            times.append(s.elapsed_time(e))
        med = statistics.median(times)
        print(f"  T={T:5d}  INT8 bmm: median={med:7.3f}ms  dtype={out.dtype}  shape={out.shape}")
    except Exception as ex:
        print(f"  T={T:5d}  INT8 bmm FAILED: {type(ex).__name__}: {ex}")

# 试 torch._int_mm (only 2D)
print()
print("  torch._int_mm test (2D, will need loop over K):")
W_int8_2d = torch.randint(-128, 127, (d_in, d_m), dtype=torch.int8, device='cuda')
S_int8_2d = torch.randint(-128, 127, (d_m, 4096), dtype=torch.int8, device='cuda')
try:
    for _ in range(5):
        out = torch._int_mm(W_int8_2d, S_int8_2d)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(10):
        out = torch._int_mm(W_int8_2d, S_int8_2d)
    e.record()
    e.synchronize()
    print(f"  torch._int_mm: 10 calls in {s.elapsed_time(e):.3f}ms, dtype={out.dtype}")
except Exception as ex:
    print(f"  torch._int_mm FAILED: {type(ex).__name__}: {ex}")
