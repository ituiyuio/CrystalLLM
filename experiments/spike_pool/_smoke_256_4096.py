"""
F28 smoke test: 256 blocks + d_model=4096 + cpu_offload=False
momo 提议: 关掉 cpu_offload, 4096 维, 256 块
F28 修 _W_freeze_fp32_cache 死占 17.2GB, 验证装得下
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
CONFIG['num_blocks'] = 256      # momo 要的
CONFIG['d_model'] = 4096
CONFIG['d_inner'] = 4096
CONFIG['cpu_offload'] = False   # momo 要关掉
CONFIG['gpu_mem_fraction'] = 1.0  # 满用 32GB
CONFIG['T_total'] = 120

print(f"  [config] num_blocks=256, d_model=4096, d_inner=4096, "
      f"cpu_offload=False, fraction=1.0", flush=True)

pool = RTX5090SpikePool(CONFIG)
print(f"  T_batch={CONFIG['T_batch']}, total steps=110 (10 warmup + 100 timed)", flush=True)
print(f"  [mem] W_pool: {pool.W_pool.nbytes/1e9:.2f}GB on {pool.W_pool.device}", flush=True)
print(f"  [mem] update_buffer: {pool.update_buffer.nbytes/1e9:.2f}GB on {pool.update_buffer.device}", flush=True)
print(f"  [mem] _W_freeze_fp32_cache: {pool._W_freeze_fp32_cache}", flush=True)
print(f"  [mem] _W_bf16_cache: {pool._W_bf16_cache.nbytes/1e9:.2f}GB", flush=True)
print(f"  [mem] after init: alloc={torch.cuda.memory_allocated()/1e9:.2f}GB  "
      f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)

train_T = CONFIG['T_batch']
if train_T == 1:
    targets = [torch.randn(CONFIG['d_model'], device='cuda') for _ in range(CONFIG['T_total'])]
else:
    targets = [torch.randn(train_T, CONFIG['d_model'], device='cuda') for _ in range(CONFIG['T_total'])]

# warmup
for i in range(10):
    pool.forward_step(targets[i])

print(f"  [mem] after warmup: alloc={torch.cuda.memory_allocated()/1e9:.2f}GB  "
      f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)

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
print(f"  peak memory: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

pool.shutdown()
