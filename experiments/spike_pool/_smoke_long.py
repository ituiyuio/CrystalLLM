"""
1000 步 smoke test, 每 100 步打 step time, 看是否随时间退化
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
CONFIG['T_total'] = 1100  # 1000 timed + 100 warmup

T = CONFIG['T_batch']
pool = RTX5090SpikePool(CONFIG)
print(f"  T_batch={T}, T_total=1100 (100 warmup + 1000 timed)")

# pre-alloc on pinned memory
if T == 1:
    targets = [torch.randn(CONFIG['d_model'], pin_memory=True) for _ in range(CONFIG['T_total'])]
else:
    targets = [torch.randn(T, CONFIG['d_model'], pin_memory=True) for _ in range(CONFIG['T_total'])]

# warmup
for i in range(100):
    pool.forward_step(targets[i].cuda(non_blocking=True))

# timed, report every 100
print(f"\n  {'step':>6s}  {'mean':>8s}  {'p10':>8s}  {'p90':>8s}  {'mem':>8s}")
window = []
for i in range(1000):
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    pool.forward_step(targets[100 + i].cuda(non_blocking=True))
    e.record()
    e.synchronize()
    window.append(s.elapsed_time(e))
    if (i + 1) % 100 == 0:
        m = statistics.mean(window)
        s_window = sorted(window)
        p10 = s_window[len(window)//10]
        p90 = s_window[len(window)*9//10]
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"  {100+i+1:6d}  {m:6.2f}ms  {p10:6.2f}ms  {p90:6.2f}ms  {mem:5.2f}GB")
        window = []

pool.shutdown()
