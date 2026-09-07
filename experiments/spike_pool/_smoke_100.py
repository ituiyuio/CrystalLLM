"""
Quick smoke test: 100 步 T=64 训练, 验证不 OOM, 统计 step time / S_norm
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import statistics
import importlib.util
import torch
spec = importlib.util.spec_from_file_location(
    "train_engine", "D:/CrystaLLM/experiments/spike_pool/train_engine.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["train_engine"] = mod
spec.loader.exec_module(mod)

RTX5090SpikePool = mod.RTX5090SpikePool
CONFIG = dict(mod.CONFIG)
CONFIG['T_total'] = 120


pool = RTX5090SpikePool(CONFIG)
print(f"  T_batch={CONFIG['T_batch']}, total steps=110 (10 warmup + 100 timed)")

train_T = CONFIG['T_batch']
if train_T == 1:
    targets = [torch.randn(CONFIG['d_model'], device='cuda') for _ in range(CONFIG['T_total'])]
else:
    targets = [torch.randn(train_T, CONFIG['d_model'], device='cuda') for _ in range(CONFIG['T_total'])]

# warmup
for i in range(10):
    pool.forward_step(targets[i])

# timed
times = []
for i in range(100):
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    pool.forward_step(targets[10 + i])
    e.record()
    e.synchronize()
    times.append(s.elapsed_time(e))

print(f"  step time (ms): mean={statistics.mean(times):.2f}  median={statistics.median(times):.2f}  "
      f"p10={sorted(times)[10]:.2f}  p90={sorted(times)[90]:.2f}")
print(f"  S_norm: {pool.S.norm().item():.2f}")
print(f"  memory: alloc={torch.cuda.memory_allocated()/1e9:.2f}GB  "
      f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB")

pool.shutdown()
