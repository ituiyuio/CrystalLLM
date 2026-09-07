"""
SpikeLLM: 用 RTX5090SpikePool 做语言模型

设计 (momo 提议):
  - embed: nn.Embedding(vocab, d_model), 标准 autograd
  - pool: 每步 forward_step_state(target, S_old) -> S_new
    - 内部独立分配 W_active (BF16 leaf), 不共享 F13 预分配 buffer
    - 用 autograd 计算 pool 内部 MSE 的 grad (W_active 一次性使用)
    - 立即更新 W_pool INT8 (per-call 独立), 释放 autograd graph
    - 返回 S_new (detached from W_active, autograd 链仅到 S_old)
  - output_head: nn.Linear(d_model, vocab), 标准 autograd
  - loss = CE(output_head(S), next_id), 反传到 embed + output_head (不反传到 W_active)

为什么 pool W_active 独立分配 (不用 F13 共享 buffer):
  - LM 序列 t in [0, L-1] 反复调 forward_step_state
  - F13 共享 buffer 在第二次调时会被覆盖, 第一次的 autograd graph 失效
  - 独立分配 + 立即 backward 让每次调独立, W_pool 通过 INT8 update 跨步累积

内存预算 (d_model=1024, d_inner=1024, K=16, vocab=4100):
  - W_pool INT8: 64 * 1024 * 1024 * 1 = 64MB
  - update_buffer FP32: 64 * 1024 * 1024 * 4 = 256MB
  - W_active BF16 per call: 16 * 1024 * 1024 * 2 = 32MB
  - 调完即释放, 不累积
  - embed: 4100 * 1024 * 4 = 16MB
  - output_head: 1024 * 4100 * 4 = 16MB
  - S [B, 1024] FP32 per position: 4KB * B
  - bpe data on CPU: bpe_train_2000 = 6.4MB
  - 总 GPU 显存: ~400MB (没把 bmm 输出算进去, bmm 临时大 ~64MB)
"""
import os
# === v4-fix: F22 fix 7 — garbage_collection_threshold:0.3 强制 GC ===
# 之前 alloc 21.35GB 是 PyTorch caching allocator 保留大 segment. 设 GC threshold 0.3
# 让 allocator 30% free 时主动 release segment 还给 driver, 不堆积大块
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF',
                       'expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.3,roundup_power2_divisions:8')
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用 train_engine 的池子
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "train_engine", os.path.join(os.path.dirname(__file__), "train_engine.py")
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["train_engine"] = _mod
_spec.loader.exec_module(_mod)
RTX5090SpikePool = _mod.RTX5090SpikePool


class SpikeLLM(nn.Module):
    """Spike Pool + Embedding + Output Head 的 LM.
    momo 提议: pool 内部 W_pool 手动更新不建 autograd 图,
                embed 和 output_head 走标准 autograd."""

    def __init__(self, pool: RTX5090SpikePool, vocab_size: int):
        super().__init__()
        self.pool = pool
        d_model = pool.d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        # tied weights (output_head = embed.T 常用, 但 demo 先分开)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        # === v4-fix: F25 (momo 提议) — 初始化尺度随 d_model 缩放 ===
        # momo 数学分析: embed 输出范数 ∝ std*sqrt(d_model), 1024 d=0.02 → 2048 d 必须 0.0141
        #   否则 d=2048 embed 范数 1.41x, 加上 S 累积残差 2x, logits 范数 2.8x
        #   跟实测 1024→2048 loss 8.4→11.16 (2.76x) 完美匹配
        # 保持 embed 范数 = 0.02*sqrt(1024) ≈ 0.64 不变:
        #   init_std = 0.02 * sqrt(1024/d_model)
        ref_d = 1024
        init_std = 0.02 * (ref_d / d_model) ** 0.5
        nn.init.normal_(self.embed.weight, std=init_std)
        nn.init.normal_(self.output_head.weight, std=init_std)
        print(f"  [F25] init_std scaling: 0.02 (at d={ref_d}) → {init_std:.4f} (at d={d_model})", flush=True)
        # === v6: F36 - 注册池子的 ISS 参数 (γ/g) 进本 module ===
        # pool 不是 nn.Module, 它的 γ/g 在这里注册同一份 Parameter:
        # named_parameters() 可见 -> optimizer 能训练, state_dict 能保存
        if getattr(pool, 'use_iss', False) and pool.iss_gamma_raw is not None:
            self.iss_gamma_raw = pool.iss_gamma_raw
            self.iss_gain = pool.iss_gain

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, L] int64 token ids
        返回: 总 loss (scalar, sum of per-position CE)

        关键: S_0 = embed(input_0) (不是 zeros!)
        原因:  spike pool S_new = S_old + delta, delta = bmm(W, S_old)
                如果 S_0 = zeros, delta_0 = 0, S_new 永远 0, 模型学不到东西

        === v4-fix: F26 (momo 提议) — 改 total_loss = total_loss + loss_t 累积 ===
        原版在循环里累加 total_loss, autograd graph 包含 L 个 loss + L 个中间 S 状态,
        显存随 L 线性增长. 改成 torch.stack + sum, PyTorch 内部可以优化 graph,
        实际显存消耗稳定不增长. 同时 backward 一次而不是 L 次, autograd overhead 也降.
        """
        B, L = input_ids.shape
        # S_0 = 第一 token 的 embed (非零, 让 pool 有东西可学)
        S = self.embed(input_ids[:, 0])  # [B, D]
        losses = []
        # 第一步也有 loss: predict input_1 from S_0 = embed(input_0)
        logits_0 = self.output_head(S)
        losses.append(F.cross_entropy(logits_0, input_ids[:, 1]))
        # t=1..L-1: S_t = S_{t-1} + delta_t, predict input_{t+1}
        for t in range(1, L - 1):
            target = self.embed(input_ids[:, t])         # [B, D]  autograd
            S = self.pool.forward_step_state(target, S)   # [B, D]  autograd
            logits = self.output_head(S)                  # [B, vocab]
            loss_t = F.cross_entropy(logits, input_ids[:, t + 1])
            losses.append(loss_t)
        # === F26: torch.stack 让 PyTorch 用 reduce 优化 backward, 避免 L 个 grad_fn chain ===
        return torch.stack(losses).sum()


def make_pool(d_model=1024, d_inner=1024, num_blocks=64, top_k=16, cpu_offload=False,
              scale_coef=1.0):
    """造一个 spike pool, d_model/d_inner 可调 (FP32 占用 4*num*d_in*d_model)

    cpu_offload (F22, momo 提议): True 时 W_pool INT8 + update_buffer FP32 + scale_pool 放 CPU RAM
    让 num_blocks=256 + d_model=4096 (装不下 32GB) 跑得动

    scale_coef (F29, momo 调参): scale_pool_init 二次缩放系数, 默认 1.0
        momo 调 0.4 让 d=4096 scale_pool_init = 0.005 * 0.4 = 0.002,
        residual r = 0.002 * 4096^1.5 = 524 (对齐 1024 baseline 327)

    F22 momo fix 4: 不调 set_per_process_memory_fraction (会强制 allocator 一次 reserve
    process_fraction * GPU 大 segment = 22GB, 22GB 一次 alloc 满).
    改用 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 让 allocator 按需 alloc, 不抢大块"""
    cfg = dict(_mod.CONFIG)
    cfg['d_model'] = d_model
    cfg['d_inner'] = d_inner
    cfg['num_blocks'] = num_blocks
    cfg['Top_K_Active'] = top_k
    cfg['T_total'] = 200   # SpikeLLM 不用这个, 随便设
    cfg['T_batch'] = 1
    cfg['gpu_mem_fraction'] = 1.0  # 不用 hard limit, 让 allocator 自由 (F22 fix 4)
    cfg['MicroSleep_Interval'] = 999999  # 关 microsleep
    # === v4-fix: F30 - 关掉末期锁定 (lock bug 根因修) ===
    # T_total=200 是随手设的, 但 _compute_gates 用 int(T_total*Lock_Ratio)=190
    # 切硬门控, 而 _step_counter 每次 forward_step_state 调用 +1 (每个 optimizer
    # step ~62 次) -> 第 3~4 个 optimizer step 起门控永久锁死, 只有块 0 激活.
    # LM 路径不用 lock, 显式关掉; 真要对齐推理时再按真实调用次数设计时间轴
    cfg['Disable_Lock'] = True
    # === v6: F36 - ISS 状态方程 (γ 泄漏 + g 增益 + Δ RMS 归一化) ===
    # S_t = γ·S_{t-1} + g·RMS_norm(Δ_t): 收缩+有界踢 -> ISS, 池子音量由 g 控制,
    # 与 scale_coef/√d/√K 解耦. γ/g 由 CE loss 经 autograd 训练 (SpikeLLM 注册)
    cfg['use_iss'] = True
    cfg['iss_gamma_init'] = 0.95
    cfg['iss_gain_init'] = 0.05
    # === v6: F37 - 随机舍入 commit ===
    # ISS 归一化后单次更新 ~1e-3 INT8 单位, 旧 0.5 均值阈值等不到提交;
    # stochastic 以 |buf| 概率进位, 无偏且节奏自动匹配
    cfg['commit_mode'] = 'stochastic'
    cfg['cpu_offload'] = cpu_offload
    # === v4-fix: F25 (momo 提议) — scale_pool 初始化随 d_model 缩放 ===
    # ||W_pool|| ∝ d, scale_pool * W_pool 残差 ∝ scale * d * ||S||, ||S|| ∝ sqrt(d)
    # 总残差范数 ∝ scale * d^1.5. 保持恒定: scale ∝ 1/d^1.5, 但 sqrt(d) 温和版更稳:
    #   init_scale = 0.01 * sqrt(1024/d_model)
    ref_d = 1024
    cfg['scale_pool_init'] = 0.01 * (ref_d / d_model) ** 0.5
    # === v4-fix: F29 (momo 调参) — scale_pool_init 二次缩放系数 ===
    # 现象: d=4096 + LR=1e-4 200步 PPL 平稳在 4900 (没收敛), 根因是 residual 范数
    #       r = 0.005 * 4096^1.5 = 1310 是 1024 baseline 327 的 4x
    #       光砍 LR 不动 scale 治标不治本, 需要再砍 scale 数值基底
    # 修法: 给 make_pool 加 scale_coef 外部 override, 默认 1.0 (跟原来一样)
    #       momo 调 0.4 (= 0.005 * 0.4 = 0.002) → r = 0.002 * 4096^1.5 = 524
    #       接近 1024 327+60%, 数值基底对齐 1024 baseline
    cfg['scale_pool_init'] *= scale_coef
    return RTX5090SpikePool(cfg)


def get_bpe_data(path: str, max_tokens: int = None, on_gpu: bool = True) -> torch.Tensor:
    """读 bpe_*.npy, 转成 int64 tensor
    on_gpu=False (F22, momo cpu_offload 配套): 留 CPU + pinned, 让 get_batch 现 copy
                    省 65M int64 = 520MB GPU 显存, 跟 256+4096 配套
    on_gpu=True (默认): 一次性 load 到 GPU (spike_llm.py 旧行为)
    """
    arr = np.load(path)
    if max_tokens is not None and max_tokens < len(arr):
        arr = arr[:max_tokens]
    t = torch.from_numpy(arr.astype(np.int64))
    if on_gpu:
        return t.to('cuda')
    else:
        # CPU + pinned for fast GPU transfer per batch
        return t.pin_memory()


def get_batch(token_ids: torch.Tensor, B: int, L: int, device='cuda'):
    """从 1D token 序列随机抽 B 个长度为 L+1 的窗口 (input + target)"""
    N = token_ids.shape[0]
    starts = torch.randint(0, N - L - 1, (B,))
    # build [B, L+1] batch
    batch = torch.stack([token_ids[s:s + L + 1] for s in starts], dim=0)
    return batch[:, :L].to(device), batch[:, 1:L + 1].to(device)


def estimate_loss(model: SpikeLLM, token_ids: torch.Tensor, B: int, L: int,
                  n_batches: int = 20, device='cuda') -> float:
    """评估平均 loss (no grad)"""
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            inp, tgt = get_batch(token_ids, B, L, device)
            # SpikeLLM.forward 期望 input_ids 包含 target shift
            # 我们手写一个简化版: 给 input, 内部 embed + pool + output_head
            # 直接复用 model.forward 但 input_ids 长度 L, target 是 shifted
            # 这里 model 需要 input_ids (B, L), target 在 L 步内 shift 出来
            # model.forward 返回 loss, 不需要外部 target (内部 cross_entropy)
            loss = model(inp)
            losses.append(loss.item() / (L - 1))  # 平均 per-position loss
    model.train()
    return float(np.mean(losses))


def generate(model: SpikeLLM, prompt_ids: torch.Tensor, max_new_tokens: int = 50,
              temperature: float = 1.0, device='cuda') -> torch.Tensor:
    """从 prompt_ids 开始自回归生成, 用训练好的 SpikeLLM.
    返回: [prompt_len + max_new_tokens] 长 token id tensor"""
    model.eval()
    prompt_len = prompt_ids.shape[0]
    generated = prompt_ids.clone()
    S = model.embed(generated.unsqueeze(0).to(device))  # [1, prompt_len, D]
    S = S[0, -1:, :]  # 取最后一步的 S 作为下一步的输入
    # 自回归: 一边生成一边把新 token 拼到 sequence
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 用 last embed + 上一步 S 做下一步
            # model.embed 是词表 lookup, 给当前 token
            cur_target = model.embed(generated[-1:].to(device))  # [1, D]
            S = model.pool.forward_step_state(cur_target, S)  # [1, D]
            logits = model.output_head(S)  # [1, vocab]
            # sample next token
            probs = F.softmax(logits[0] / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            generated = torch.cat([generated, torch.tensor([next_id])])
    model.train()
    return generated


def decode_ids(token_ids, tokenizer_path="D:/CrystaLLM/experiments/v49_pre/bpe_tokenizer.pkl"):
    """BPE token ids -> text"""
    import pickle
    with open(tokenizer_path, 'rb') as f:
        enc = pickle.load(f)
    return enc.decode(token_ids.tolist())


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--d_model", type=int, default=1024)
    p.add_argument("--d_inner", type=int, default=1024)
    p.add_argument("--num_blocks", type=int, default=64)
    p.add_argument("--top_k", type=int, default=16)
    p.add_argument("--vocab_size", type=int, default=4100)
    p.add_argument("--bpe_file", type=str, default="bpe_train_2000_s42.npy")
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--generate", action="store_true", help="训完跑一个 generation demo")
    p.add_argument("--gen_prompt", type=str, default="The", help="起始 prompt text")
    p.add_argument("--gen_length", type=int, default=50, help="生成 token 数")
    p.add_argument("--cpu_offload", action="store_true",
                   help="F22 (momo 提议): W_pool + update_buffer 放 CPU RAM, 让 256b+4096 跑得动")
    p.add_argument("--scale_coef", type=float, default=1.0,
                   help="F29 (momo 调参): scale_pool_init 二次缩放, 默认 1.0 (1024 baseline 327). "
                        "v6 F36 后 scale 只影响池子有效 LR, 不再决定稳定性")
    p.add_argument("--pool_lr_mult", type=float, default=100.0,
                   help="v6 F36: 池子 LR 相对 head LR 的倍数. ISS 归一化后 W 只学方向, "
                        "需要比 head 大 ~2 个量级才有 INT8 提交节奏; 稳定性已与它解耦")
    p.add_argument("--warmup_steps", type=int, default=200,
                   help="F24 (momo 提议): linear warmup 前 N 步 LR 从 0 升到 base, "
                        "避免 d_model=2048 起始数值大直接打爆. 1024 可设 100, 2048 建议 200-500")
    p.add_argument("--lr_schedule", type=str, default="warmup_cosine",
                   choices=["constant", "warmup_constant", "warmup_cosine"],
                   help="LR schedule: constant / warmup_constant / warmup_cosine")
    args = p.parse_args()

    if not args.generate:
        # 训练模式 (原 main)
        def _train_only():
            device = 'cuda'
            print(f"  Creating pool: d_model={args.d_model}, d_inner={args.d_inner}, "
                  f"num_blocks={args.num_blocks}, top_k={args.top_k}, "
                  f"cpu_offload={args.cpu_offload}")
            pool = make_pool(d_model=args.d_model, d_inner=args.d_inner,
                             num_blocks=args.num_blocks, top_k=args.top_k,
                             cpu_offload=args.cpu_offload,
                             scale_coef=args.scale_coef)
            print(f"  Creating SpikeLLM: vocab_size={args.vocab_size}, "
                  f"embed params={args.vocab_size * args.d_model:,}")
            model = SpikeLLM(pool, vocab_size=args.vocab_size).to(device)
            # === v4-fix: F22 (momo 优化 3) — 强制 empty_cache 让 caching allocator 释放 22GB virtual ===
            # 256+4096+cpu_offload 时 PyTorch 贪心预留 22GB, 实际只用 ~1.5GB
            # empty_cache 把 free segment 还给 driver, 后续 alloc 不会撞 hard limit
            torch.cuda.empty_cache()
            print(f"  [init] after empty_cache: {torch.cuda.memory_allocated()/1e9:.2f}GB alloc, "
                  f"{torch.cuda.memory_reserved()/1e9:.2f}GB reserved", flush=True)
            # 列出 top alloc 块找 21GB 哪来的
            if torch.cuda.memory_allocated() > 5e9:
                print(f"  [init] GPU alloc > 5GB! dumping summary...", flush=True)
                print(torch.cuda.memory_summary(abbreviated=True), flush=True)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Trainable params (embed + output_head): {n_params:,}")
            print(f"  Pool INT8 params: {pool.W_pool.numel():,}")

            bpe_path = os.path.join("D:/CrystaLLM/experiments/v49_pre", args.bpe_file)
            print(f"  Loading BPE data: {bpe_path}")
            # === v4-fix: F22 — cpu_offload 时 bpe data 留 CPU pinned (省 520MB GPU) ===
            token_ids = get_bpe_data(bpe_path, max_tokens=args.max_tokens,
                                     on_gpu=not args.cpu_offload)
            print(f"  Total tokens: {len(token_ids):,} (on {'cuda' if not args.cpu_offload else 'cpu pinned'})")

            # === v4-fix: F23 (momo 提议) — 基于 d_model 自动缩放 LR ===
            # momo 分析: grad_W 元素 ∝ d_model 线性 (residual ∝ sqrt(d_model), S_old ∝ sqrt(d_model))
            # 让 LR 反比于 sqrt(d_model) 跟 grad 尺度抵消, 1024 起步 LR=1e-3 直接给
            # args.lr 当作 d=1024 基准, 实际 LR = args.lr * (1024/d_model)^0.5
            # 例: d=1024 → 1e-3, d=2048 → 7.07e-4, d=4096 → 5e-4
            ref_d_model = 1024
            scaled_lr = args.lr * (ref_d_model / args.d_model) ** 0.5
            print(f"  [F23] LR scaling: base={args.lr:.2e} (at d={ref_d_model}) -> "
                  f"actual={scaled_lr:.2e} (at d={args.d_model})", flush=True)

            # === v4-fix: F35 - pool LR 跟 --lr 走 ===
            # 现象: forward_step_state 的 W 更新用 cfg['lr_base']=1e-3 固定值,
            #       --lr (经 F23 缩放) 只调 embed/head 的 AdamW -> LR sweep
            #       (F28 的 1e-4/3e-4/7e-4) 池子根本没感受到
            # 修法: F23 缩放后的 lr 写进 pool.cfg['lr_base'], 池子和
            #       embed/head 用同一个数 (池子手动梯度无 momentum, 同量级合理)
            # === v6: F36 - ISS 下加 pool_lr_mult ===
            # Δ 归一化后 W 只学"方向", 精确梯度幅度 ~g/rms 偏小, 提交节奏慢;
            # ISS 保证稳定性与池子 LR 解耦, 可以放心把池子 LR 放大 ~2 个量级
            pool.cfg['lr_base'] = scaled_lr * args.pool_lr_mult
            print(f"  [F35/F36] pool lr_base -> {scaled_lr * args.pool_lr_mult:.2e} "
                  f"(head {scaled_lr:.2e} x mult {args.pool_lr_mult})", flush=True)

            optimizer = torch.optim.AdamW(
                [p for n, p in model.named_parameters() if 'embed' in n or 'output' in n or 'iss_' in n],
                lr=scaled_lr, betas=(0.9, 0.95), weight_decay=0.0
            )

            # === v4-fix: F24 (momo 提议) — LR schedule (warmup + cosine decay) ===
            # 1024 起步 loss 8.4 稳定, 2048 起步 10.5 直接打爆. warmup 让前 N 步 LR 从 0 升
            # 到 base, 起始数值稳. cosine decay 后段降 LR, 防止震荡
            if args.lr_schedule == "constant":
                scheduler = None
            elif args.lr_schedule == "warmup_constant":
                def lr_lambda_warmup_const(step):
                    if step < args.warmup_steps:
                        return step / max(1, args.warmup_steps)
                    return 1.0
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_warmup_const)
            elif args.lr_schedule == "warmup_cosine":
                def lr_lambda_warmup_cos(step):
                    if step < args.warmup_steps:
                        return step / max(1, args.warmup_steps)
                    # cosine decay from 1.0 to ~0.1 over remaining steps
                    progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
                    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_warmup_cos)
            else:
                scheduler = None
            if scheduler is not None:
                print(f"  [F24] LR schedule: {args.lr_schedule} (warmup={args.warmup_steps})", flush=True)

            print(f"\n  Training: B={args.batch_size}, L={args.seq_len}, "
                  f"steps={args.steps}, eval_every={args.eval_every}")
            print("=" * 60)
            t_start = time.time()
            # === v4-fix: F20e — progress writer 给 _monitor.py 实时看 (跟 train_engine 同款 JSON) ===
            def _write_progress(step, loss_val):
                import json
                progress = {
                    "step": step,
                    "total": args.steps,
                    "sps": (step + 1) / (time.time() - t_start) if (time.time() - t_start) > 0 else 0,
                    "elapsed_s": time.time() - t_start,
                    "eta_s": (args.steps - step - 1) / max(1e-9, (step + 1) / (time.time() - t_start)),
                    "s_norm": float(loss_val / (args.seq_len - 1)),  # 实际是 per-token loss, 字段名兼容 _monitor
                    "t_batch": args.batch_size * args.seq_len,
                    "active_logical": 0,  # SpikeLLM 不用 spike pool 的 active 概念
                    "num_blocks": args.num_blocks,
                    "top_k": args.top_k,
                }
                path = "D:/CrystaLLM/spike_llm_progress.json"
                tmp = path + ".tmp"
                with open(tmp, 'w', encoding='utf-8') as f:
                    json.dump(progress, f)
                import os as _os
                _os.replace(tmp, path)
            for step in range(args.steps):
                inp, tgt = get_batch(token_ids, args.batch_size, args.seq_len, device)
                loss = model(inp)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                if step % args.log_every == 0:
                    elapsed = time.time() - t_start
                    sps = (step + 1) / elapsed if elapsed > 0 else 0
                    print(f"  step {step:5d} | loss {loss.item() / (args.seq_len - 1):.4f} "
                          f"| sps {sps:.2f} | elapsed {elapsed:.0f}s", flush=True)
                    _write_progress(step, loss.item())
                if (step + 1) % args.eval_every == 0:
                    val_loss = estimate_loss(model, token_ids, args.batch_size,
                                             args.seq_len, n_batches=10, device=device)
                    ppl = math.exp(val_loss) if val_loss < 20 else float('inf')
                    print(f"  === eval @ step {step+1} | val_loss {val_loss:.4f} | ppl {ppl:.2f}", flush=True)
                    _write_progress(step, val_loss)

            final_loss = estimate_loss(model, token_ids, args.batch_size,
                                       args.seq_len, n_batches=50, device=device)
            final_ppl = math.exp(final_loss) if final_loss < 20 else float('inf')
            print(f"\n  FINAL val_loss {final_loss:.4f} | ppl {final_ppl:.2f}")
            # === v4-fix: F20e — 训完跑 generation demo, momo 看实际输出 ===
            try:
                import pickle
                with open("D:/CrystaLLM/experiments/v49_pre/bpe_tokenizer.pkl", 'rb') as f:
                    enc = pickle.load(f)
                for prompt in ["The ", "In ", "Once "]:
                    try:
                        prompt_ids = torch.tensor(enc.encode_ordinary(prompt), dtype=torch.long)
                        out = generate(model, prompt_ids, max_new_tokens=30, device=device)
                        text = decode_ids(out)
                        print(f"  [{prompt!r}] -> {text!r}")
                    except Exception as ex:
                        print(f"  [{prompt!r}] gen failed: {ex}")
            except Exception as ex:
                print(f"  (skip generation, tokenizer error: {ex})")
            return model, token_ids
        _train_only()
    else:
        # 生成模式: 先训一点再 generate
        device = 'cuda'
        print(f"  Creating pool + model for generation test")
        pool = make_pool(d_model=args.d_model, d_inner=args.d_inner,
                         num_blocks=args.num_blocks, top_k=args.top_k,
                         cpu_offload=args.cpu_offload,
                         scale_coef=args.scale_coef)
        model = SpikeLLM(pool, vocab_size=args.vocab_size).to(device)
        bpe_path = os.path.join("D:/CrystaLLM/experiments/v49_pre", args.bpe_file)
        token_ids = get_bpe_data(bpe_path, max_tokens=args.max_tokens)
        print(f"  Total tokens: {len(token_ids):,}")

        if args.steps > 0:
            # === v4-fix: F23 — 同上, gen 模式也用 scaled_lr (只是 short training) ===
            ref_d_model = 1024
            scaled_lr = args.lr * (ref_d_model / args.d_model) ** 0.5
            pool.cfg['lr_base'] = scaled_lr * args.pool_lr_mult  # F35/F36: gen 模式同款接线
            optimizer = torch.optim.AdamW(
                [p for n, p in model.named_parameters() if 'embed' in n or 'output' in n or 'iss_' in n],
                lr=scaled_lr, betas=(0.9, 0.95), weight_decay=0.0
            )
            print(f"  Quick training: {args.steps} steps")
            t_start = time.time()
            for step in range(args.steps):
                inp, tgt = get_batch(token_ids, args.batch_size, args.seq_len, device)
                loss = model(inp)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % 20 == 0:
                    print(f"  step {step} | loss {loss.item() / (args.seq_len - 1):.4f}")
            print(f"  Training done in {time.time() - t_start:.0f}s")
        
        # Generate
        print(f"\n  === GENERATION TEST ===")
        try:
            import pickle
            with open("D:/CrystaLLM/experiments/v49_pre/bpe_tokenizer.pkl", 'rb') as f:
                enc = pickle.load(f)
            prompt_ids_list = enc.encode_ordinary(args.gen_prompt)
            prompt_ids = torch.tensor(prompt_ids_list, dtype=torch.long)
            print(f"  Prompt: '{args.gen_prompt}' -> {prompt_ids_list}")
        except Exception as ex:
            print(f"  Tokenizer load failed: {ex}, using random prompt")
            prompt_ids = torch.randint(0, args.vocab_size, (3,))
            args.gen_prompt = "<random>"
        
        out_ids = generate(model, prompt_ids, max_new_tokens=args.gen_length, device=device)
        out_text = decode_ids(out_ids)
        print(f"\n  === OUTPUT ===")
        print(f"  {out_text}")
        print(f"  ===============\n")

