"""
跑 200 步训练并采样 nvidia-smi GPU utilization
直接读 utilization.gpu 数组的 peak/mean (这才是 momo 看到的 30%)
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


def run(T, n=200):
    cfg = dict(CONFIG)
    cfg['T_total'] = n + 20
    pool = RTX5090SpikePool(cfg)
    torch.manual_seed(0)
    targets = [torch.randn(T, cfg['d_model'], device='cuda') for _ in range(cfg['T_total'])]

    # warmup
    for _ in range(10):
        pool.forward_step(targets[0])

    # main loop, 输出 READY 让外部采样 nvidia-smi
    print(f"START_T{T}", flush=True)
    for i in range(n):
        pool.forward_step(targets[i])
    torch.cuda.synchronize()
    print(f"DONE_T{T}", flush=True)


if __name__ == "__main__":
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run(T)
