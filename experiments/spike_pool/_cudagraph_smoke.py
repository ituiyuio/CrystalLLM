"""
CUDA Graphs 测试: 捕获 forward_step 的 bmm+backward 段, replay 测 launch overhead 节省。

策略:
- gate 决策在 graph 外 (Python-level)
- 把 active_idx 写入固定 buffer
- graph 捕获 bmm + autograd + update
- replay 时, 改 buffer 值, graph 用新值

测试对比: 同一 T, eager step time vs graph replay time
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


def run_cudagraph(T, n_eager=30, n_replay=50, warmup=10):
    cfg = dict(CONFIG)
    cfg['T_total'] = n_eager + n_replay + warmup + 50
    pool = RTX5090SpikePool(cfg)
    torch.manual_seed(0)
    targets = [torch.randn(T, cfg['d_model'], device='cuda') for _ in range(cfg['T_total'])]

    # warmup (eager)
    for _ in range(warmup):
        pool.forward_step(targets[0])

    # ---- Eager timing ----
    eager_times = []
    for i in range(n_eager):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        pool.forward_step(targets[warmup + i])
        e.record()
        e.synchronize()
        eager_times.append(s.elapsed_time(e))

    eager_med = statistics.median(eager_times)

    # ---- CUDA Graphs capture ----
    # We capture the body of forward_step (after gate decision) as a graph
    # The gate decision runs eagerly, writes active_idx to a buffer
    # The graph reads from the buffer

    # Allocate a fixed active_idx buffer (top K blocks for the smoke test)
    K = cfg['Top_K_Active']
    fixed_active_idx = torch.arange(K, device='cuda', dtype=torch.long)
    pool.b_gate[:] = -1e6
    pool.b_gate[:K] = 1.0  # force first K blocks active

    # We need a fixed target buffer (graph reads from it)
    # The graph will use the same target memory location each replay
    target_buf = targets[warmup + n_eager].clone()

    # Capture: do one forward step with the fixed target
    # We need a separate stream for capture
    s_capture = torch.cuda.Stream()
    s_capture.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_capture):
        # warmup on capture stream
        for _ in range(3):
            pool.forward_step(target_buf)
    torch.cuda.current_stream().wait_stream(s_capture)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s_capture):
        pool.forward_step(target_buf)

    # ---- Graph replay timing ----
    replay_times = []
    for i in range(n_replay):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        e.synchronize()
        replay_times.append(s.elapsed_time(e))

    replay_med = statistics.median(replay_times)
    speedup = eager_med / replay_med if replay_med > 0 else 0

    print(f"  T={T:5d}  eager median={eager_med:7.3f}ms  "
          f"graph replay median={replay_med:7.3f}ms  speedup={speedup:.2f}x")

    pool.shutdown()
    return eager_med, replay_med


if __name__ == "__main__":
    print("=" * 70)
    print("  Eager vs CUDA Graph replay")
    print("=" * 70)
    for T in [1, 64, 256, 1024, 4096]:
        try:
            run_cudagraph(T)
        except Exception as ex:
            print(f"  T={T}: FAILED {type(ex).__name__}: {ex}")
