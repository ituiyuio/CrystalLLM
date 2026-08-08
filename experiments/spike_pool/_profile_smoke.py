"""
Profile each forward_step with torch.profiler to find the actual bottleneck.

报告：每个 kernel 的累计时间 / 调用次数 / avg ms per call。
帮 momo 找到"为什么 step time 不随 T 下降"的真凶。
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

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


def profile_run(T: int, n_steps: int = 20, warmup: int = 5):
    cfg = dict(CONFIG)
    cfg['T_total'] = n_steps + warmup + 50
    pool = RTX5090SpikePool(cfg)

    torch.manual_seed(0)
    if T == 1:
        targets = [torch.randn(cfg['d_model'], device='cuda') for _ in range(cfg['T_total'])]
    else:
        targets = [torch.randn(T, cfg['d_model'], device='cuda') for _ in range(cfg['T_total'])]

    # warmup
    for _ in range(warmup):
        pool.forward_step(targets[0])

    # profile
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for i in range(n_steps):
            with torch.profiler.record_function(f"FWD_T{T}_step{i}"):
                pool.forward_step(targets[warmup + i])

    print(f"\n{'='*70}")
    print(f"  Profile T={T}, n_steps={n_steps}")
    print(f"{'='*70}")
    # 按 CUDA time 排序的 top kernels
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=20
    ))

    pool.shutdown()


if __name__ == "__main__":
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    profile_run(T)
