"""
torch.compile test: 把 forward_step 编译成 inductor IR,
预期: 融合 elementwise op, 减少 launch overhead, 接近 CUDA time (5.7ms).

不能直接 compile forward_step (因为有 self 状态), 改策略:
- 把 forward_step 的核心 bmm+loss+backward 部分抽出来
- 用 torch.compile 编译这部分
- 看实际加速
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


def run(n_steps=30, T=64, use_compile=False):
    cfg = dict(CONFIG)
    cfg['T_total'] = n_steps + 10
    pool = RTX5090SpikePool(cfg)
    torch.manual_seed(0)
    targets = [torch.randn(T, cfg['d_model'], device='cuda') for _ in range(cfg['T_total'])]

    if use_compile:
        # 只编译 forward_step 的核心 bmm 段
        # 注意: 不能编译 self 方法 (有状态), 我们在外面 wrap 一个纯函数
        compiled_fn = torch.compile(pool.forward_step, mode="reduce-overhead", fullgraph=False)
        step_fn = compiled_fn
    else:
        step_fn = pool.forward_step

    # warmup
    for _ in range(5):
        step_fn(targets[0])
    if use_compile:
        # torch.compile 第一次需要触发 tracing, 多跑几次
        for _ in range(3):
            step_fn(targets[0])

    times = []
    for i in range(n_steps):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        step_fn(targets[i])
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))

    mean = statistics.mean(times)
    median = statistics.median(times)
    p10 = sorted(times)[len(times)//10]
    p90 = sorted(times)[len(times)*9//10]

    label = "torch.compile" if use_compile else "eager"
    print(f"  {label} T={T}: mean={mean:6.2f}ms  median={median:6.2f}ms  "
          f"p10={p10:6.2f}  p90={p90:6.2f}  (n={n_steps})")
    pool.shutdown()
    return median


if __name__ == "__main__":
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    print("=" * 50)
    print(f"  eager vs torch.compile (T={T})")
    print("=" * 50)
    m1 = run(30, T, use_compile=False)
    m2 = run(30, T, use_compile=True)
    speedup = m1 / m2 if m2 > 0 else 0
    print(f"\n  speedup: {speedup:.2f}x (eager {m1:.2f}ms -> compile {m2:.2f}ms)")
