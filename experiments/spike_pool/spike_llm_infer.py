"""
SpikeLLM 推理测试:
  1. quick 训练 (300 步) - 让模型能产出有结构的 token
  2. save state_dict 到 spike_llm_infer_test.pt
  3. 在 fresh state 测推理: per-token latency + 生成样本
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import time
import math
import numpy as np
import torch
import torch.nn.functional as F

# 复用 spike_llm 模块
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "spike_llm", os.path.join(os.path.dirname(__file__), "spike_llm.py")
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["spike_llm"] = _mod
_spec.loader.exec_module(_mod)
SpikeLLM = _mod.SpikeLLM
make_pool = _mod.make_pool
get_bpe_data = _mod.get_bpe_data
get_batch = _mod.get_batch
estimate_loss = _mod.estimate_loss
generate = _mod.generate
decode_ids = _mod.decode_ids


def quick_train(model, token_ids, steps=300, B=4, L=32, lr=1e-3, log_every=50, device='cuda'):
    """quick 训练, 跟 spike_llm main 同款"""
    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if 'embed' in n or 'output' in n or 'iss_' in n],
        lr=lr, betas=(0.9, 0.95), weight_decay=0.0
    )
    t0 = time.time()
    for step in range(steps):
        inp, _ = get_batch(token_ids, B, L, device)
        loss = model(inp)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % log_every == 0:
            print(f"  train step {step:4d} | loss {loss.item() / (L - 1):.4f} | "
                  f"elapsed {time.time() - t0:.0f}s", flush=True)
    print(f"  quick train done in {time.time() - t0:.0f}s", flush=True)
    # 训完评估一次
    val_loss = estimate_loss(model, token_ids, B, L, n_batches=10, device=device)
    val_ppl = math.exp(val_loss) if val_loss < 20 else float('inf')
    print(f"  val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}", flush=True)
    return val_ppl


def benchmark_generate(model, prompt_ids, max_new_tokens=50, n_warmup=5, device='cuda'):
    """测每 token 生成 latency (生成 n_warmup 跳过 launch overhead)"""
    import pickle
    with open("D:/CrystaLLM/experiments/v49_pre/bpe_tokenizer.pkl", 'rb') as f:
        enc = pickle.load(f)
    # warmup
    print(f"  warmup {n_warmup} tokens...", flush=True)
    for _ in range(n_warmup):
        with torch.no_grad():
            cur_target = model.embed(prompt_ids[-1:].to(device))
            S = model.pool.forward_step_state(cur_target,
                                              model.embed(prompt_ids.unsqueeze(0).to(device))[0, -1:, :])
            logits = model.output_head(S)
    torch.cuda.synchronize()

    # 测 latency
    print(f"  benchmark {max_new_tokens} tokens...", flush=True)
    generated = prompt_ids.clone()
    S = model.embed(generated.unsqueeze(0).to(device))[0, -1:, :]
    token_times = []
    torch.cuda.synchronize()
    with torch.no_grad():
        for i in range(max_new_tokens):
            t0 = time.perf_counter()
            cur_target = model.embed(generated[-1:].to(device))
            S = model.pool.forward_step_state(cur_target, S)
            logits = model.output_head(S)
            probs = F.softmax(logits[0], dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            generated = torch.cat([generated, torch.tensor([next_id])])
            torch.cuda.synchronize()
            token_times.append((time.perf_counter() - t0) * 1000)  # ms
    return generated, token_times


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bpe_file", type=str, default="bpe_train_65M_s42.npy")
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--save_path", type=str, default="D:/CrystaLLM/spike_llm_infer_test.pt")
    args = p.parse_args()

    device = 'cuda'
    print("=" * 60, flush=True)
    print(f"  SpikeLLM 推理测试: train {args.steps}步 + generate benchmark", flush=True)
    print("=" * 60, flush=True)

    # 1. 创建模型
    pool = make_pool(d_model=1024, d_inner=1024, num_blocks=64, top_k=16)
    model = SpikeLLM(pool, vocab_size=4100).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_pool = pool.W_pool.numel()
    print(f"  trainable (embed + head): {n_train:,} | pool INT8: {n_pool:,}", flush=True)

    # 2. 加载数据
    bpe_path = os.path.join("D:/CrystaLLM/experiments/v49_pre", args.bpe_file)
    token_ids = get_bpe_data(bpe_path)
    print(f"  data: {len(token_ids):,} tokens from {args.bpe_file}", flush=True)

    # 3. 快速训练
    val_ppl = quick_train(model, token_ids, steps=args.steps,
                          B=args.batch_size, L=args.seq_len, lr=args.lr,
                          device=device)

    # 4. Save state — SpikeLLM (nn.Module) 包了 pool, model.state_dict() 包含所有
    save_data = {
        'model_state': model.state_dict(),
        'val_ppl': val_ppl,
        'config': {
            'd_model': 1024, 'd_inner': 1024, 'num_blocks': 64,
            'top_k': 16, 'vocab_size': 4100,
        }
    }
    torch.save(save_data, args.save_path)
    print(f"  saved to {args.save_path} ({os.path.getsize(args.save_path) / 1024 / 1024:.1f} MB)",
          flush=True)

    # 5. 推理测试 — 多个 prompt
    import pickle
    with open("D:/CrystaLLM/experiments/v49_pre/bpe_tokenizer.pkl", 'rb') as f:
        enc = pickle.load(f)
    prompts = ["The ", "In ", "def ", "import ", "Once "]

    print(f"\n{'=' * 60}", flush=True)
    print(f"  GENERATION BENCHMARK (max_new_tokens={args.max_new_tokens})", flush=True)
    print(f"{'=' * 60}", flush=True)
    for prompt in prompts:
        prompt_ids = torch.tensor(enc.encode_ordinary(prompt), dtype=torch.long)
        out, token_times = benchmark_generate(model, prompt_ids,
                                               max_new_tokens=args.max_new_tokens,
                                               device=device)
        text = decode_ids(out)
        mean_ms = sum(token_times) / len(token_times)
        p50 = sorted(token_times)[len(token_times) // 2]
        p90 = sorted(token_times)[int(len(token_times) * 0.9)]
        print(f"\n  [{prompt!r}] mean {mean_ms:.1f}ms/tok | p50 {p50:.1f} | p90 {p90:.1f}ms",
              flush=True)
        print(f"  -> {text!r}", flush=True)
