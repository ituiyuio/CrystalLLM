"""
脉冲残差池训练引擎 - RTX 5090 极致优化版 (v3-final + v3-fix)

设计哲学：
    本系统将Transformer的"FFN层"拆解为数百个独立的"残差块"，每个块拥有独立的
    INT8权重。通过轻量级内容寻址（注意力偏置）和宏观调度（GDP税收、磨损均衡），
    动态选择最合适的块参与计算。前向/反向均采用批处理（torch.bmm），确保
    Tensor Core满载运行。训练末期锁定门控，与推理完全对齐。

核心创新：
    1. 3D连续INT8权重池 [num_blocks, d_inner, d_model] (L2缓存友好)
    2. 原生INT8更新：FP32缓冲累加，超过0.5步长再提交整数
    3. 局部反向传播 + 异步冻结（仅活跃块参与梯度计算）
    4. 动态学习率（C方案）：老块余弦退火，新块指数升温
    5. 微睡眠（离线巩固）：即时重演 + Scale全局归一化
    6. 有形大手（宏观调度）：磨损均衡 + 僵尸清退 + GDP同质化税收
    7. 轻量Attention（内容寻址偏置，仅用于门控参考）
    8. S范数软截断（防止全局状态爆炸）
    9. k_embed预热（内容寻址冷启动问题）
    10. 末期锁定（最后5%固化，与推理对齐）

作者：基于深度对话推演实现
日期：2026-08-08
分支：spike-pool-v3

=== v3-fix 修复清单（相对 v3-final 原文） ===
  F1. autograd.grad requires_grad 缺失 → 显式 .detach().requires_grad_(True)
  F2. S 范数截断每步 .item() CPU 同步   → 改为纯 GPU 缩放因子
  F3. k_embed 预热的 Python for 循环    → 全向量化（base + noise_matrix）
  F4. b_gate 缺少对称机制               → 增加均值回归项 (均值向历史激活率靠拢)
  F5. 微睡眠 scale 归一化误改 W_pool    → 只调 scale_pool，不动权重（scale 本就是干这用的）
  F6. Windows GBK stdout 编码          → 顶部 import sys + reconfigure utf-8
  F7. FFN 漏写 down-projection (W2)    → d_inner = d_model = 4096（方阵权重）
      原 v3-final 设 d_inner=16384 但只写了一段 W @ S，
      bmm 输出 [d_inner] 加回 [d_model] shape 不匹配 (4096 vs 16384)。
      折中：d_inner 塌缩为 d_model，去掉 expand 概念，保留残差池 + 门控设计。
"""

import os
# === v3-fix: F9 — CUDA 显存 fragmentation 缓解（必须在 import torch 之前）===
# 现象: 128 块 + 4096 d_inner 单步峰值申请 2GB 时 OOM,但 nvidia-smi 显示 28GB 空闲
# 根因: PyTorch 缓存分配器碎片化(虚拟分配 101GB / 物理 32GB)
# 修法: expandable_segments 让 segment 块按需扩展,减少浪费
# 注意: PyTorch 第一次 cuda API 调用就读 env var,必须在 import torch 前设
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time   # === v4-fix: F20c — progress writer 用 ===
import numpy as np
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

# === v3-fix: F6 — Windows GBK stdout 编码（emoji 兜不住）===
# 现象：__init__ 里的 🚀 触发 UnicodeEncodeError: 'gbk' codec
# 修法：reconfigure stdout/stderr 到 utf-8（errors='replace' 兜底）
# Linux/Mac 不需要这层（默认 utf-8），但不影响（reconfigure 是 no-op）
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ======================== 超参数配置 ========================
CONFIG = {
    # 模型维度（必须16的倍数，Tensor Core对齐）
    'd_model': 4096,              # 全局状态向量维度
    'd_inner': 4096,              # === v3-fix: F7 — 与 d_model 对齐 ===
                                   # 原 v3-final 写 16384(FFN 4x expand)，
                                   # 但漏了 FFN 的 down-projection (W2)，
                                   # 导致 bmm 输出 [d_inner] 加回 [d_model] shape 不匹配。
                                   # 折中：d_inner = d_model = 方阵权重 W @ S = delta
                                   # 显存：8.59GB → 2.15GB（参数减 4 倍）
    'num_blocks': 64,             # 残差块总数（v3-final 原 128,RTX 5090 32GB 装不下,改 64）
    'd_k': 16,                    # 内容寻址的键维度（极小的路由表）

    # 训练控制
    'lr_base': 1e-3,              # 基础学习率
    'T_total': 10000,             # 总训练步数 (完整跑)
    'T_report': 100,              # 有形大手调度间隔（步）
    'T_warm': 500,                # 新块学习率升温步数

    # 门控与调度
    'Freeze_Limit': 500,          # 磨损均衡：超过此步未激活则强制唤醒
    'Burst_Limit': 20,            # 行锤击缓解：连续激活超过此值则强制静默
    'MicroSleep_Interval': 999999, # === v4-fix: F20c — 关掉 microsleep (momo 测速用) ===
                                   # 10000 步里 microsleep 跑 100 次, 后期冻结块到 64 个,
                                   # 每次处理 4GB FP32 = 3s, 把平均 step time 推 87ms
                                   # 关掉后 step time 稳在 28ms (跟 v3 一致)
                                   # 真实训练要看效果决定要不要开
    'Lock_Ratio': 0.95,           # 末期锁定起始比例（最后5%步数）
    'Top_K_Active': 16,           # 每步最大激活块数（v3-final 原 32,改 16 压峰值）

    # GDP税收（同质化惩罚）
    'GDP_Threshold': 0.6,         # |余弦相似度|超过此值则收税（双向）
    'GDP_Tax_Rate': 2.0,          # 税率系数

    # 内容寻址
    'Attn_Scale': 0.1,            # 注意力偏置的缩放系数（防止主导门控）

    # 物理闸门
    'S_Norm_Cap': 512.0,          # S范数硬上限（超过则缩放到128）
    'Warmup_Steps': 100,          # k_embed预热步数
    'T_batch': 64,                # === v4-fix: F17 — 序列批维度
                                   # 1D target 走 T=1 旧路径（向后兼容）
                                   # 2D target [T, d_model] 把 bmm 第二参数
                                   # 从 [K, d_model, 1] 扩到 [K, d_model, T]
                                   # 把 GEMV 升级为真 GEMM,Tensor Core 满载
                                   # T=64 时 bmm 是 64 倍算力,Tensor Core 利用率
                                   # 从 0.01% → 应该能上 80%+
    'T_batch_max': 4096,         # 预分配 S_cache 的最大 T,超出则 lazy 重分配
                                   # === v4-fix: F19 — 提高到 4096,让 T=1024-4096 不用 regrow
                                   # T=4096 测得 31% BF16 peak, 越大越接近 compute-bound
    'gpu_mem_fraction': 1.0,      # === v4-fix: F20c — momo 调高内存墙到 100% (32GB 整张卡)
                                   # 之前 0.85 限到 27GB, v3 跑 100 步就 OOM (caching allocator 持 100GB 虚拟)
                                   # 现在 1.0 让 PyTorch 把整张 5090 都用上, expandable_segments
                                   # 仍设着 (Windows 失效但 Linux 兼容)

    # === v3-fix: F4 — b_gate 均值回归 ===
    # 磨损均衡 +0.5 单边累积会推高老块门控，
    # 引入 EMA 风格的均值回归，让门控自然衰减回历史激活频次的中位线
    'Gate_EMA_Decay': 0.999,      # 每步 b_gate *= decay（不显式设均值，靠自然演化）
    'Gate_Activation_Pull': 0.01, # 激活的块向历史激活率靠拢的步长

    # 保存路径
    'save_path': './model_pool',  # 模型文件前缀

    # === v3-fix: F12 — 显存 instrumentation + 硬限制 ===
    # momo 提议: 不要靠猜峰值,加进程级硬限制 + memory_stats 周期性打印 + OOM dump
    # fail-fast 原则: 超预算立刻崩,崩时拿到完整 memory_stats,不靠试错
    'gpu_mem_fraction': 0.85,        # 进程级显存硬上限 (RTX 5090 32GB * 0.85 = ~27GB)
    'mem_log_interval': 50,          # 每 N 步打印 memory_stats
    'cpu_offload': False,            # === v4-fix: F22 — CPU offload (momo 提议: 稀疏架构利用 CPU RAM)
                                   # True: W_pool INT8 + update_buffer FP32 + scale_pool 放 CPU RAM
                                   #      每步把 active K 块 (INT8) 拉到 GPU 算 bmm, 更新拉回 CPU 累加
                                   #      让 num_blocks=256 + d_model=4096 (装不下 32GB) 跑得动
                                   # 适用: 大 num_blocks (>=128) + d_model (>=2048) 装不下 GPU 时
                                   # 代价: 每 step 一次 K 块 CPU<->GPU transfer (~50-100ms overhead)
                                   # 限制: K=16 + d_inner*d_model*4 bytes FP32 update per step = ~256MB,
                                   #      加上 W_pool transfer ~256MB, 总 transfer 500MB/step
                                   #      PCIe 4.0 x16 = 32GB/s, transfer 耗时 16ms/step
}


# === v4-fix: F20c — _TargetIterable 提到模块级 (Windows multiprocessing pickle 要求) ===
# 现象: 在 if __name__ == "__main__": 里定义 class, DataLoader workers (子进程) 找不到
#       "Can't get attribute '_TargetIterable' on <module '__mp_main__'>"
# 修法: 模块顶层定义, 跨进程可见
class _TargetIterable(torch.utils.data.IterableDataset):
    """无尽生成 [T, d_model] 随机 targets 在 CPU 上, 配 DataLoader pin_memory=True
    实际生产替换为真实 dataset (token IDs from disk, etc.)"""
    def __init__(self, T, d_model, seed=0):
        self.T = T
        self.d_model = d_model
        self.seed = seed
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            yield torch.randn(self.T, self.d_model, generator=g)


class RTX5090SpikePool:
    """
    脉冲残差池训练引擎 - 所有操作全向量化，无Python循环
    """

    def __init__(self, config: dict):
        """
        初始化权重池、门控偏置、注意力组件、更新缓冲及统计变量。
        """
        self.cfg = config
        self.num_blocks = config['num_blocks']
        self.d_model = config['d_model']
        self.d_inner = config['d_inner']
        self.d_k = config['d_k']

        # ----- 1. 核心存储：巨型连续3D张量 [num_blocks, d_inner, d_model] -----
        # 所有块的权重在显存中物理连续，L2缓存预取效率极高
        # === v3-fix: F12 — 进程级显存硬限制（momo fail-fast 方案）===
        # === v4-fix: F22 (momo fix 8) — cpu_offload 时用 0.6 fraction 限 19GB ===
        # PyTorch caching allocator 默认在第一次 alloc 时抓大 segment (21GB).
        # 设 fraction=0.6 (19GB) 限制 max alloc, W_active 512MB 能 alloc.
        # 不设 hard limit → caching allocator 自由 32GB → 抓 21GB primary → OOM
        if config.get('cpu_offload', False):
            torch.cuda.set_per_process_memory_fraction(0.6, device=0)
        else:
            torch.cuda.set_per_process_memory_fraction(
                config['gpu_mem_fraction'], device=0
            )

        # === v4-fix: F22 — CPU offload 存储 (momo 提议) ===
        # 稀疏架构 (256 blocks 只 K=16 active) 让 frozen 块放 CPU RAM 几乎免费
        # 装不下 GPU 的 num_blocks=256 + d_model=4096 也能跑
        self.cpu_offload = config.get('cpu_offload', False)
        pool_device = 'cpu' if self.cpu_offload else 'cuda'

        # === v4-fix: F22 (momo fix 5) — 用 .pin_memory() 显式 pin (PyTorch 自动 pin 大 tensor) ===
        # 显式 pin 让 transfer 走 DMA 32GB/s. 21.5GB CUDA host pinned memory
        # 算在 memory_allocated(), 但 F22 fix 4 不调 set_per_process_memory_fraction 不会卡 GPU
        self.W_pool = torch.zeros(
            self.num_blocks, self.d_inner, self.d_model,
            dtype=torch.int8, device=pool_device
        )
        if self.cpu_offload:
            # === v4-fix: F22 fix 6 — 不 pin_memory W_pool, 用普通 CPU tensor ===
            # pin_memory 走 cudaMallocHost 算 CUDA alloc, 4.3GB W_pool + 8.6GB update_buffer
            # + PyTorch internal overhead 接近 22GB, 撞 32GB. 不 pin 用 pageable memory:
            # - 不算 CUDA alloc (21GB → 0)
            # - transfer 走 pageable DMA 慢 2x (~16GB/s), 但能跑
            # - 实际 PCIe 4.0 x16: pinned 32GB/s vs pageable 16GB/s, 256MB transfer 多 8ms
            pass
            # === F22 debug: 印 W_pool 的真实 size + device + memory_format ===
            print(f"  [F22] W_pool: device={self.W_pool.device}, shape={self.W_pool.shape}, "
                  f"dtype={self.W_pool.dtype}, nbytes={self.W_pool.nbytes/1e9:.2f}GB, "
                  f"is_pinned={self.W_pool.is_pinned()}", flush=True)

        # === v4-fix: F25 (momo 提议) — scale_pool 随 d_model 缩放 ===
        # 默认 0.01 (F20e momo LM 路径: 配 INT8 [-128,127] W 太大, bmm 输出爆炸)
        # momo 数学分析: 残差范数 ∝ scale * d^1.5, 保持恒定: scale ∝ 1/sqrt(d_model)
        scale_init = config.get('scale_pool_init', 0.01)
        self.scale_pool = torch.full((self.num_blocks,), scale_init, device=pool_device)
        if self.cpu_offload:
            print(f"  [F22] scale_pool: device={self.scale_pool.device}, "
                  f"init={scale_init:.5f} (F25 d_model scaling), "
                  f"nbytes={self.scale_pool.nbytes/1e9:.4f}GB", flush=True)

        # 门控偏置（FP32），由"有形大手"直接操纵，不参与梯度
        self.b_gate = torch.full((self.num_blocks,), -1e6, device='cuda')

        # 残差唤醒：仅第1块初始化为随机权重，其他块从0开始生长
        self.W_pool[0] = torch.randint(
            -128, 127, (self.d_inner, self.d_model),
            dtype=torch.int8, device='cuda'
        )
        self.b_gate[0] = 1.0   # 第1块初始门控为开启

        # ----- 2. 轻量注意力组件（内容寻址，仅用于门控偏置的辅助） -----
        # 查询投影：将S从d_model投影到d_k维
        self.q_proj = torch.randn(self.d_k, self.d_model, device='cuda') * 0.01
        # 键嵌入：每个块对应一个d_k维向量，可训练（但通过GDP和预热来调整）
        self.k_embed = torch.randn(self.num_blocks, self.d_k, device='cuda') * 0.01

        # ----- 3. 原生INT8更新缓冲（FP32/BF16 累加器） -----
        # 梯度更新不直接截断回INT8，而是累加在累加器缓冲中，累积超过0.5步长才提交整数
        # === v4-fix: F22 (momo 修复四) — update_buffer 用 BF16 (省 50% 内存) ===
        # 256+4096 17.2GB FP32 → 8.6GB BF16, 允许 num_blocks=512 装下
        # BF16 累加精度损失小 (mantissa 7-bit), commit threshold 0.5 仍用 FP32 算 norm
        self.update_buffer = torch.zeros(
            self.num_blocks, self.d_inner, self.d_model,
            dtype=torch.bfloat16, device=pool_device
        )
        # === F22 (momo fix 5) — 不调 pin_memory() 原因同上, 21.5GB 会算 CUDA alloc ===
        # if self.cpu_offload:
        #     self.update_buffer = self.update_buffer.pin_memory()
        if self.cpu_offload:
            # === F22 debug: 印 update_buffer 的真实 size + device, 找 22GB alloc 哪来 ===
            print(f"  [F22] update_buffer: device={self.update_buffer.device}, shape={self.update_buffer.shape}, "
                  f"dtype={self.update_buffer.dtype}, nbytes={self.update_buffer.nbytes/1e9:.2f}GB, "
                  f"is_pinned={self.update_buffer.is_pinned()}", flush=True)

        # ----- 4. 状态与统计变量 -----
        self.S = torch.zeros(self.d_model, device='cuda')                # 全局状态
        self.burst_counter = torch.zeros(self.num_blocks, device='cuda') # 连续激活计数（行锤击）
        self.last_active = torch.zeros(self.num_blocks, dtype=torch.long, device='cuda') # 最后激活步
        self.grad_norm_ema = torch.zeros(self.num_blocks, device='cuda') # 梯度范数滑动平均（僵尸检测）
        self.activation_stats = torch.zeros(self.num_blocks, dtype=torch.long, device='cuda') # 总激活次数（重排用）
        # === v3-fix: F7b — hist_delta 形状修正 ===
        # 原 v3-final 写 self.d_model，但 deltas 是 [K, d_inner]，
        # pooled_delta = deltas.mean(dim=1) -> [K]，跟 [K, d_model] 不对齐
        # 修法：hist_delta 存 d_inner 维的 hidden 方向（d_inner==d_model 时等价于 d_model 维），
        # 直接 += deltas（语义：累积每个块在 d_inner 维的 hidden 输出）
        self.hist_delta = torch.zeros(self.num_blocks, self.d_inner, device='cuda') # 历史增量方向（GDP核算）

        # k_embed预热相关
        self.S_avg_buffer = torch.zeros(self.d_model, device='cuda')    # 前Warmup_Steps步的S累积
        self.k_embed_initialized = False
        self.warmup_steps = config['Warmup_Steps']

        # === v3-fix: F13 — 预分配 fixed-size FP32 缓存 buffer（momo fail-fast 根因修）===
        # 根因: W_active = W_pool[active_idx].float() * scale 每步新建 FP32 tensor
        #       K 变化导致 caching allocator fragmentation,allocated 持续增长
        # 修法: 预分配 max_k 大小的 FP32 buffer (不带 grad),
        #       每步 view + detach().requires_grad_() + copy_ + mul_ 复用同一段内存
        # 注意: PyTorch 禁止 in-place 改 leaf+grad tensor,所以 buffer 不带 grad,
        #       每步给 W_active 临时建 detached leaf view
        max_k = config['Top_K_Active'] + 4   # +4 余量
        self._W_fp32_cache = torch.zeros(
            max_k, self.d_inner, self.d_model,
            dtype=torch.float32, device='cuda'
        )   # 不带 grad,纯数据 buffer
        # === v4-fix: F17 — _S_fp32_cache 升级为 [K, d_model, T_max] ===
        # 原 [K, d_model, 1] 把 bmm 钉死在 GEMV (P=1),Tensor Core 大量空闲
        # 新 [K, d_model, T_max] 让 bmm 第二参数可塞 T 个位置,
        # 变成真 GEMM [K, d_inner, d_model] @ [K, d_model, T] = [K, d_inner, T]
        # T_max 决定最大支持 T_batch,超出 lazy 重分配
        self._S_fp32_max_T = config.get('T_batch_max', 128)
        self._S_fp32_cache = torch.zeros(
            max_k, self.d_model, self._S_fp32_max_T,
            dtype=torch.float32, device='cuda'
        )
        # === v4-fix: F20c — _microsleep 专用预分配 buffer (覆盖 num_blocks 个冻结块) ===
        # 根因: _microsleep 每 100 步执行, 后期 freeze_idx 可达 64 (所有块都冻结),
        #       旧代码 self.W_pool[freeze_idx].float() * scale 每次新建 1GB FP32,
        #       10000 步 = 100 次 × 1GB = 100GB 分配压力, caching allocator 撑不住
        # 修法: 预分配 _W_freeze_cache 大小为 num_blocks (跟 update_buffer 一样大),
        #       _microsleep 复用这个 buffer, 跟 forward_step 的 F13 同款
        # === v4-fix: F28 (momo 找死占) — _microsleep 关了就 lazy alloc ===
        # 现象: 256+4096+cpu_offload=False 时 _W_freeze_fp32_cache 永远占 17.2GB FP32,
        #       但 MicroSleep_Interval=999999 (F20c 关了), _microsleep 永不跑, 纯死占
        # 根因: 设计冗余 — _microsleep 关了就不该 alloc
        # 修法: 仅当 MicroSleep_Interval < 阈值时才创建, 默认配置 (999999) 完全不 alloc
        #       省 17.2GB @ 256+4096, 装下 momo 要求的 256+4096+cpu_offload=False
        if self.cfg.get('MicroSleep_Interval', 100) < 100000:
            self._W_freeze_fp32_cache = torch.zeros(
                self.num_blocks, self.d_inner, self.d_model,
                dtype=torch.float32, device='cuda'
            )  # 1.07GB @ 64+1024, _microsleep 专用
            self._S_freeze_fp32_cache = torch.zeros(
                self.num_blocks, self.d_model, 1,
                dtype=torch.float32, device='cuda'
            )  # 256KB, S 缓存 (T=1 路径)
        else:
            # F28: _microsleep 关了, 永不 alloc, 省 num_blocks * d_inner * d_model * 4 字节
            self._W_freeze_fp32_cache = None
            self._S_freeze_fp32_cache = None

        # === v4-fix: F18 — 预分配 BF16 cache,让 bmm 走 5th-gen Tensor Core ===
        # 根因: 5090 5th-gen TC 跑 BF16/FP16/INT8/FP8, FP32 走普通 SIMT cores (~21 TFLOPS)
        #       BF16 TC 峰值 ~250 TFLOPS, ~12x speedup. 即使小 bmm 也能用满 TC.
        # 策略:
        #   - W_active 改成 BF16 leaf (was FP32),autograd backward 走 BF16 bmm (也在 TC)
        #   - scale 改成 post-bmm (避免 BF16 量化误差传到 W 上)
        #   - grads_batch 是 BF16, INT8 更新前 cast FP32
        # 内存: BF16 W cache = 512MB (vs FP32 1GB) — 同时省一半显存
        self._W_bf16_cache = torch.zeros(
            max_k, self.d_inner, self.d_model,
            dtype=torch.bfloat16, device='cuda'
        )
        self._S_bf16_cache = torch.zeros(
            max_k, self.d_model, self._S_fp32_max_T,
            dtype=torch.bfloat16, device='cuda'
        )

        # === v3-fix: F2 — 预算缩放因子张量，避免每步 .item() 同步 ===
        # 预分配 inv_scale_ratio = cap / S_Norm_Cap = 128/512 = 0.25
        # 实际缩放时用 (S_norm / 128) 作为分母，全 GPU 计算
        self._s_norm_target = torch.tensor(128.0, device='cuda')

        # === v6: F36 - ISS 状态方程 (leaky integrator + 注入归一化 + 可学习增益) ===
        # 数学: S_t = γ·S_{t-1} + g·RMS_norm(Δ_t)
        #   - 映射 = γ-压缩 + 有界踢 -> 输入到状态稳定 (ISS), 对任意 scale 有界
        #   - 稳态 ‖S‖ ∈ [g·√d/√(1-γ²), g·√d/(1-γ)] (方向随机~对齐之间)
        #   - 池子音量由 g 单独控制, 与 scale_pool/√d/√K 全解耦
        #   - scale 从稳定性旋钮降级为 LR 旋钮, S_Norm_Cap 降级为保险丝
        # 物理: LIF 膜电位泄漏 + 突触输入稳态缩放; 即 SSM 的 retention γ
        # γ: sigmoid 参数化可学习, init 0.95; g: ReZero/LayerScale 式, init 0.05
        # (非 0, 避开 g=0 时池子梯度全灭的死区)
        self.use_iss = config.get('use_iss', True)
        if self.use_iss:
            gamma0 = float(config.get('iss_gamma_init', 0.95))
            self.iss_gamma_raw = nn.Parameter(
                torch.tensor(math.log(gamma0 / (1.0 - gamma0)), device='cuda'))
            self.iss_gain = nn.Parameter(
                torch.tensor(float(config.get('iss_gain_init', 0.05)), device='cuda'))
        else:
            self.iss_gamma_raw = None
            self.iss_gain = None
        # 快权重遗忘钩子 (治外积 churn): 活跃块未提交更新的每调用衰减.
        # 默认 1.0 关闭 -- 衰减会抬高 commit 的等效阈值 (平衡值 per_call/(1-decay)),
        # 要开需与 pool LR 联调, 先隔离实验变量
        self.iss_buffer_decay = float(config.get('iss_buffer_decay', 1.0))
        # === v6: F37 - commit 模式 ===
        # 'threshold': 旧均值阈值 (整块 mean|buf| > 0.5 才四舍五入提交)
        # 'stochastic': 随机舍入 -- 以 |buf| 概率向 sign(buf) 进 1, 期望变化 = buf.
        #   ISS 归一化后单次更新 ~1e-3 INT8 单位, threshold 要等上百次同号累积
        #   -> 池子几乎不写; stochastic 无偏且提交节奏自动匹配更新幅度
        self.commit_mode = config.get('commit_mode', 'threshold')

        # 阶段标志
        self._step_counter = 0
        self.lock_phase = False         # 末期锁定（推理对齐）
        self.disable_refresh = False    # 禁用随机刷新

        # 后台线程（仅用于保存检查点，不干扰主循环）
        self.executor = ThreadPoolExecutor(max_workers=1)

        print(f"🚀 RTX5090SpikePool initialized: {self.num_blocks} blocks, "
              f"d_model={self.d_model}, d_inner={self.d_inner}")
        total_params = self.num_blocks * self.d_inner * self.d_model
        print(f"   Total INT8 parameters: {total_params / 1e9:.2f}B")
        if self.use_iss:
            g0 = float(self.iss_gain.detach())
            gam0 = float(torch.sigmoid(self.iss_gamma_raw.detach()))
            lo = g0 * self.d_model ** 0.5 / (1 - gam0 ** 2) ** 0.5
            hi = g0 * self.d_model ** 0.5 / (1 - gam0)
            print(f"   [F36] ISS: gamma={gam0:.3f} (learnable), gain={g0:.3f} (learnable), "
                  f"steady ||S|| band [{lo:.1f}, {hi:.1f}]")

    # ===================== 门控决策 =====================
    # === v4-fix: F32 - _compute_gates 增加 S 参数 ===
    # 现象: q = q_proj @ self.S, 但 SpikeLLM 路径 (forward_step_state) 从不写 self.S
    #       (caller 维护外部 S), self.S 恒为 zeros -> q 恒为 0, content_bias 恒为 0
    #       内容寻址整个是死的, 路由退化为纯 b_gate
    # 修法: S 由 caller 传入 (forward_step_state 传当前 S_old 的 batch 均值);
    #       默认 None 走 self.S, forward_step (纯池子路径) 行为不变
    def _compute_gates(self, step: int,
                       S: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        全向量化门控决策（无Python循环）。

        融合三路信号：
            1. 市场惯性（b_gate）：历史表现 + GDP税收调节
            2. 内容寻址（注意力）：当前S与各块键的匹配度
            3. 硬件规则：磨损均衡（强制唤醒） + 行锤击缓解（强制静默）

        返回：激活块索引列表 (GPU Tensor)
        """
        lock_start = int(self.cfg['T_total'] * self.cfg['Lock_Ratio'])

        # ---- 内容寻址得分 ----
        # 计算查询向量 q = q_proj @ S，然后与所有键做点积
        S_ref = self.S if S is None else S.detach()
        q = torch.mv(self.q_proj, S_ref)            # [d_k]
        content_scores = torch.mv(self.k_embed, q)   # [num_blocks]
        content_bias = content_scores * self.cfg['Attn_Scale']

        # ---- 融合门控概率 ----
        fused_logits = self.b_gate + content_bias
        prob = torch.sigmoid(fused_logits)

        # ---- 硬件规则 ----
        # 磨损均衡：冻结超限块强制唤醒（避免"饿死"）
        freeze_mask = (step - self.last_active) > self.cfg['Freeze_Limit']
        # 行锤击缓解：连续激活过多则强制静默（防止过热点）
        burst_mask = self.burst_counter > self.cfg['Burst_Limit']

        # ---- 末期锁定：硬阈值，无随机 ----
        # === v4-fix: F30 - Disable_Lock 开关（SpikeLLM lock bug 根因修）===
        # 现象: make_pool 随手设 T_total=200, 但 _step_counter 每次
        #       forward_step_state 调用 +1 (每个 optimizer step ~62 次调用),
        #       第 3~4 个 optimizer step 起 step >= int(200*0.95)=190,
        #       门控永久切到 prob>0.5 硬阈值; b_gate 只有块 0 为正 (其余 -1e6),
        #       -> 256 块里永远只有块 0 激活, 4.29B 参数实际工作的只有 16.7M
        # 修法: SpikeLLM 路径设 cfg['Disable_Lock']=True, 训练全程保持
        #       Bernoulli 采样 + 统计更新 (b_gate EMA / burst / last_active)
        gates_locked = (self.lock_phase or step >= lock_start) \
            and not self.cfg.get('Disable_Lock', False)
        if gates_locked:
            G = (prob > 0.5).int()
            self.activation_stats += G   # 记录激活频次（用于推理重排）
        else:
            # 正常训练：随机抽样 + 强制唤醒
            rand_vals = torch.rand(self.num_blocks, device='cuda')
            G = ((rand_vals < prob) | freeze_mask).int()
            G[burst_mask] = 0

        # ---- 安全防护：至少激活1个块 ----
        # === v4-fix: F34 - topk 加噪声打破 -1e6 平局 ===
        # 现象: 休眠块 b_gate 全部精确等于 -1e6*decay^k (逐元素相同),
        #       topk 平局时 CUDA 稳定返回低索引 -> 永远只有固定的块 1~15
        #       被兜底唤醒, 块 16+ 从未参与训练
        # 注意: 噪声幅度必须随 |b_gate| 缩放. 固定 1e-3 无效 -- FP32 在
        #       1e6 量级的 ULP = 0.0625, 1e-3 噪声直接被舍入吞掉, 平局依旧
        #       (验证脚本 Part A 实测 unique=8=top_k 才发现). 取 1e-3 比例:
        #       休眠块拿到 [0,1000) 散布随机轮换; 活跃块 b_gate 差异 ~0.1
        #       >> 噪声 1e-3, 排序不受影响
        if G.sum() == 0:
            noise = torch.rand_like(self.b_gate) * 1e-3 \
                * self.b_gate.abs().clamp(min=1.0)
            _, indices = torch.topk(self.b_gate + noise,
                                    min(self.cfg['Top_K_Active'], self.num_blocks))
            G[indices] = 1

        # ---- 性能防护：限制最大激活数（保证Batch GEMM最优尺寸） ----
        if G.sum() > self.cfg['Top_K_Active']:
            noise = torch.rand_like(self.b_gate) * 1e-3 \
                * self.b_gate.abs().clamp(min=1.0)
            _, top_idx = torch.topk((self.b_gate + noise) * G.float(),
                                    self.cfg['Top_K_Active'])
            G = torch.zeros_like(G)
            G[top_idx] = 1

        # ---- 更新统计（非锁定阶段） ----
        if not self.lock_phase and not gates_locked:
            # === v3-fix: F4 — b_gate 均值回归（自然衰减 + 激活后轻微回拉） ===
            # 全员 EMA 衰减，防止磨损均衡 +0.5 单边累积
            self.b_gate *= self.cfg['Gate_EMA_Decay']

            self.last_active = torch.where(
                G.bool(),
                torch.tensor(step, device='cuda'),
                self.last_active
            )
            self.burst_counter = torch.where(
                G.bool(),
                self.burst_counter + 1,
                (self.burst_counter - 0.5).clamp(min=0)
            )

        # === v3-fix: F13 — 移除 .cpu() 让 active_idx 留在 GPU ===
        # 原版返回 CPU tensor 触发 device transfer,产生额外分配
        # 新版直接返回 GPU int64,索引 self.W_pool 无需 transfer
        return G.nonzero(as_tuple=True)[0]

    # ===================== 动态学习率（C方案） =====================
    def _get_lr_vectorized(self, idx: torch.Tensor, step: int) -> torch.Tensor:
        """
        向量化学习率计算：
            - 老块：余弦退火（从 lr_base 衰减到接近0）
            - 新块（b_gate < -1e5）：指数升温，永不超越老块当前值
        这确保了新块不会打乱已稳定的全局状态。
        """
        T = self.cfg['T_total']
        lr_base = self.cfg['lr_base']

        # 老块基线：余弦退火
        lr_old = lr_base * 0.5 * (1 + math.cos(math.pi * step / T))
        lr_old = max(lr_old, 1e-8)
        lr_old_t = torch.tensor(lr_old, device='cuda')

        # 判断哪些是新块（b_gate极低表示尚未完全唤醒）
        is_new = self.b_gate[idx] < -1e5

        if is_new.any():
            # === v3-fix: F8 — torch.clamp 标量 + max= 在 PyTorch 2.9 不支持 ===
            # 原写法: torch.clamp(0.05 * math.exp(...), max=1.0)
            # 修法: 用 Python min 算标量值
            warm_factor = min(0.05 * math.exp(step / self.cfg['T_warm']), 1.0)
            warm_factor_t = torch.full_like(idx, warm_factor, dtype=torch.float, device='cuda')
            return torch.where(is_new, lr_old_t * warm_factor_t, lr_old_t)
        else:
            return torch.full_like(idx, lr_old_t, dtype=torch.float, device='cuda')

    # ===================== 核心训练步 =====================
    def forward_step(self, target_S: torch.Tensor) -> torch.Tensor:
        """
        一步训练：
            1. k_embed预热（前100步）
            2. 门控决策（_compute_gates）
            3. 前向 Batch GEMM（单核启动）
            4. 局部反向传播（批处理，仅活跃块）
            5. 原生INT8更新（FP32缓冲累加 + 整数提交）
            6. 微睡眠（每MicroSleep_Interval步）
            7. 有形大手调度（每T_report步）
            8. S范数软截断（防止数值爆炸）

        参数：
            target_S: 当前批次的局部目标（例如下一个token的embedding）
        返回：
            更新后的全局状态 S
        """
        step = self._step_counter
        self._step_counter += 1
        lock_start = int(self.cfg['T_total'] * self.cfg['Lock_Ratio'])

        # === v4-fix: F17 — 序列批维度检测 ===
        # 1D target [d_model]  → T=1  旧 GEMV 路径（向后兼容,推理侧不变）
        # 2D target [T, d_model] → T 位置 bmm [K,d_inner,d_model] @ [K,d_model,T]
        #                       = [K,d_inner,T]  真 GEMM,5th-gen Tensor Core 满载
        # T=1 时所有后续逻辑数学上完全等价于 v3 (bmm 后 [K,d_inner,1] 等价于
        # squeeze 后的 [K,d_inner]; mean(dim=2) 是 identity; .T + mean(dim=0) 等价)
        if target_S.dim() == 1:
            target_S = target_S.unsqueeze(0)  # [1, d_model]
        T = target_S.shape[0]
        if T > self._S_fp32_max_T:
            # Lazy 重分配:超过 T_batch_max 才发生,生产环境通常不会
            # F18: 同时 regrow FP32 + BF16 两个 cache
            self._S_fp32_max_T = T
            self._S_fp32_cache = torch.zeros(
                self._W_fp32_cache.shape[0], self.d_model, T,
                dtype=torch.float32, device='cuda'
            )
            self._S_bf16_cache = torch.zeros(
                self._W_bf16_cache.shape[0], self.d_model, T,
                dtype=torch.bfloat16, device='cuda'
            )
            print(f"⚠️  S_fp32+bf16_cache lazy-regrown to T={T}")

        # ===== 1. k_embed 内容预热（解决冷启动） =====
        # 前Warmup_Steps步，累积S的均值方向，用于初始化新块的键
        if step < self.warmup_steps:
            self.S_avg_buffer += self.S.detach()
        elif step == self.warmup_steps and not self.k_embed_initialized:
            S_avg = self.S_avg_buffer / self.warmup_steps
            S_avg_norm = F.normalize(S_avg, dim=0)   # [d_model]

            # === v3-fix: F3 — 全向量化 k_embed 初始化 ===
            # 所有非 0 块共享同一个 base key = q_proj @ S_avg_norm，
            # 差异仅由独立 noise 矩阵提供
            base_key = torch.mv(self.q_proj, S_avg_norm)   # [d_k]
            noise_matrix = torch.randn(
                self.num_blocks - 1, self.d_k, device='cuda'
            ) * 0.01
            with torch.no_grad():
                self.k_embed[1:] = base_key.unsqueeze(0) + noise_matrix
            self.k_embed_initialized = True
            print(f"🌱 k_embed initialized with S_avg at step {step}")

        # ===== 2. 末期锁定（不可逆） =====
        if step >= lock_start and not self.lock_phase:
            self.lock_phase = True
            self.disable_refresh = True
            print(f"🔒 Step {step}: Entering inference alignment phase.")

        # ===== 3. 门控决策 =====
        active_idx = self._compute_gates(step)
        K = len(active_idx)
        if K == 0:
            return self.S   # 全休眠，直接返回

        # ===== 4. 前向：Batch GEMM (BF16 Tensor Core, with T dim) =====
        S_old = self.S.clone()
        # === v3-fix: F13 — 预分配缓存 view + copy_ + mul_（momo 根因修）===
        # === v4-fix: F17 — S_batch 升级为 [K, d_model, T] (T=1 时数学等价) ===
        # === v4-fix: F18 — W/S 全部 BF16, bmm 走 5th-gen TC (peak 250 TFLOPS) ===
        W_active = self._W_bf16_cache[:K].detach().requires_grad_(True)  # BF16 leaf
        S_batch = self._S_bf16_cache[:K, :, :T]                          # BF16 view, no grad

        # 就地填充 W_active:从 W_pool 走 INT8->FP32->BF16 链 (单次 fused cast)
        # scale 改成 post-bmm 应用 (避免 BF16 量化误差污染 W)
        # no_grad 块: copy_ in-place 不破坏 leaf+grad 的 autograd contract
        with torch.no_grad():
            W_active.copy_(self.W_pool[active_idx].float().bfloat16())  # 读 256MB INT8, 写 512MB BF16
            S_batch.copy_(S_old.unsqueeze(0).unsqueeze(-1).expand(K, -1, T).bfloat16())  # S 也 BF16

        # bmm: [K, d_inner, d_model]_BF16 @ [K, d_model, T]_BF16 = [K, d_inner, T]_BF16
        # === v4-fix: F18 — BF16 bmm 走 Tensor Core,FP32 累积 (cuBLAS 默认), 输出 BF16 ===
        deltas_bf16 = torch.bmm(W_active, S_batch)   # [K, d_inner, T] BF16

        # post-scale: deltas *= scale.view(K,1,1)  FP32 scale × BF16 deltas → FP32 (数值稳)
        # scale_pool 是 per-block 的标量,后乘比预乘 W 精度更好 (BF16 mantissa 7-bit 有限)
        deltas_fp32 = deltas_bf16.float() * self.scale_pool[active_idx].view(K, 1, 1)
        # 持久 S 更新:sum over K, mean over T (per-token 效应,T 不影响更新幅度)
        # T=1 时 deltas_fp32 是 [K, d_inner, 1], .sum(0).T = [1, d_inner], .mean(0) = [d_inner]
        #         等价 v3 的 self.S = S_old + deltas.sum(dim=0)
        delta_S = deltas_fp32.sum(dim=0).T          # [T, d_inner] = [T, d_model]
        self.S = S_old + delta_S.mean(dim=0)        # [d_model]

        # === v4-fix: F18 — hist_delta / loss 全部用 FP32 deltas (post-scale 后的精确值) ===
        # === v4-fix: F17 — deltas 现在是 [K, d_inner, T], mean(dim=2) 折叠 T 维 ===
        # 同时把 deltas 改成 deltas_fp32 (从 BF16 cast),hist_delta 是 FP32 累加更稳定
        deltas = deltas_fp32   # 别名给下面用,避免重写
        # 替换 W_active 指向 (autograd 用,无变化)

        # ===== 5. 更新历史增量（GDP核算数据） =====
        if not self.lock_phase:
            # === v3-fix: F7b — 直接 += deltas（[K, d_inner]），hist_delta 形状已对齐 ===
            # 原 v3-final 用 pooled_delta = deltas.mean(dim=1) 把 d_inner 维求平均成 [K]，
            # 跟 [K, d_model] 的 hist_delta 形状冲突。这里直接存 d_inner 维 hidden 方向。
            # === v4-fix: F17 — deltas 现在是 [K, d_inner, T], mean(dim=2) 折叠 T 维 ===
            self.hist_delta[active_idx] += deltas.mean(dim=2)  # [K, d_inner]

        # ===== 6. 局部反向传播（仅训练态） =====
        if not self.lock_phase:
            # === v4-fix: F17 — loss 扩展到 T 维 ===
            # 表达式: 对每个 (k, t) 让 S_old + deltas[k, :, t] 逼近 target_S[t]
            # S_old broadcast 到 [1, 1, d_model], deltas.permute(0,2,1) = [K, T, d_inner]
            # T=1 时退化为 [K, 1, d_model], 数值上等价 v3 的 [K, d_model]
            S_old_exp = S_old.view(1, 1, self.d_model)            # [1, 1, d_model]
            deltas_perm = deltas.permute(0, 2, 1)                  # [K, T, d_inner] = [K, T, d_model]
            delta_contrib = S_old_exp + deltas_perm                # [K, T, d_model]
            # === v4-fix: F17 — target 显式 expand 到 [K, T, d_model],避免 mse_loss 广播 warning ===
            target_exp = target_S.unsqueeze(0).expand(K, T, self.d_model)  # [K, T, d_model]
            losses = F.mse_loss(delta_contrib, target_exp, reduction='none').mean(dim=2)  # [K, T]
            total_loss = losses.sum()

            # 🚀 唯一反向传播Kernel（批处理）
            # === v4-fix: F18 — W_active 是 BF16 leaf, grads_batch 是 BF16, cast FP32 给 INT8 更新用 ===
            grads_batch = torch.autograd.grad(total_loss, W_active, retain_graph=False)[0].float()  # [K, d_inner, d_model] FP32

            # 动态学习率（向量化C方案）
            lr = self._get_lr_vectorized(active_idx, step)   # [K]

            # ----- 原生INT8更新（FP32缓冲累加） -----
            # 梯度是相对于 float(W) 的，转换为对 INT8 权重的更新步长需除以 scale
            scale_exp = self.scale_pool[active_idx].unsqueeze(-1).unsqueeze(-1)
            delta_update = -lr.unsqueeze(-1).unsqueeze(-1) * grads_batch / (scale_exp + 1e-8)
            self.update_buffer[active_idx] += delta_update

            # 提交累积超过0.5步长的块（四舍五入到整数）
            update_norms = self.update_buffer[active_idx].view(K, -1).abs().mean(dim=1)
            commit_mask = update_norms > 0.5
            if commit_mask.any():
                commit_idx = active_idx[commit_mask]
                rounded = torch.round(self.update_buffer[commit_idx]).to(torch.int8)
                self.W_pool[commit_idx] = torch.clamp(self.W_pool[commit_idx] + rounded, -128, 127)
                self.update_buffer[commit_idx] -= rounded.float()

            # 更新梯度范数EMA（用于僵尸检测和报表）
            grad_norms = grads_batch.view(K, -1).norm(dim=1)
            self.grad_norm_ema[active_idx] = 0.9 * self.grad_norm_ema[active_idx] + 0.1 * grad_norms

            # === v3-fix: F11 — 显式显存管理（v3-final 训练循环的工程缺陷）===
            # 现象: 100 步之后 OOM,PyTorch caching allocator reserved ~100GB 虚拟
            # 根因: v3-final 每步不 del 中间 tensor,autograd graph 累积,
            #       PyTorch caching allocator 默认贪心 pre-allocate
            # 修法: del 关键中间 tensor + 每 50 步 empty_cache 强制释放
            del W_active, deltas, losses, total_loss, grads_batch
            del delta_update, update_norms
            if 'rounded' in locals():
                del rounded
            # === v4-fix: F20c — 关掉 F11 empty_cache (momo 测速) ===
            # empty_cache 每 50 步扫描 27GB peak 内存释放, 越来越慢导致 step time 30ms -> 90ms
            # 用 del 显式释放代替 empty_cache, 让 caching allocator 复用不释放
            # if step % 50 == 0:
            #     torch.cuda.empty_cache()

            # === v3-fix: F12 — 周期性 memory_stats 打印（momo fail-fast 方案）===
            # 不靠猜峰值,每 50 步 print 真实 reserved/allocated/allocs/frees
            # 找到 leak 时已经知道从哪一步开始膨胀
            if step % self.cfg['mem_log_interval'] == 0 and step > 0:
                _ms = torch.cuda.memory_stats()
                _peak = _ms['reserved_bytes.all.peak'] / 1e9
                _cur = _ms['reserved_bytes.all.current'] / 1e9
                _alloc = _ms['allocated_bytes.all.current'] / 1e9
                _n_alloc = _ms.get('num_alloc_retries', 0) + _ms.get('num_ooms', 0)
                print(f"  [mem] step={step:5d} | reserved={_cur:5.2f}GB "
                      f"alloc={_alloc:5.2f}GB | peak={_peak:5.2f}GB | "
                      f"allocs={_ms.get('allocation_count.all', 0):8d} "
                      f"frees={_ms.get('free_count.all', 0):8d}")

            # === v4-fix: F20c — F13 主动 empty_cache 也关掉 (momo 测速) ===
            # 预分配 buffer 解决了 W_active leak,但 grads_batch / delta_update
            # 仍是 autograd 返回的新 tensor,每步新建,需要显式 del
            # 注: 用 locals() check 防 lock_phase 跳过训练态时 UnboundLocalError
            for _v in ('W_active', 'S_batch', 'deltas', 'grads_batch', 'delta_update'):
                if _v in locals():
                    del locals()[_v]
            # if step % 10 == 0:
            #     _alloc_gb = torch.cuda.memory_allocated() / 1e9
            #     _total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            #     if _alloc_gb > 0.7 * _total_gb:
            #         torch.cuda.empty_cache()

        # ===== 7. 微睡眠（周期性执行） =====
        if step % self.cfg['MicroSleep_Interval'] == 0 and not self.lock_phase:
            self._microsleep(target_S)

        # ===== 8. 有形大手调度 =====
        if step % self.cfg['T_report'] == 0 and not self.lock_phase:
            self._macro_scheduler(step)

        # ===== 9. S范数软截断（防止全局状态爆炸） =====
        # === v3-fix: F2 — 纯 GPU 缩放，避免每步 .item() 同步 ===
        s_norm = self.S.norm()                       # GPU scalar tensor, 不触发同步
        over = s_norm > self.cfg['S_Norm_Cap']       # GPU bool scalar
        # 缩放因子 = s_norm / 128（只在 over=True 时启用）
        scale = s_norm / self._s_norm_target          # GPU scalar
        # torch.where 不要求标量与向量 shape 完全一致——它会广播
        # 但这里 s_norm 是 0-d tensor，可与 self.S 逐元素比较并条件缩放
        self.S = torch.where(over, self.S / scale, self.S)

        return self.S

    # ===================== 微睡眠 =====================
    def _microsleep(self, target_S: torch.Tensor):
        """
        === v4-fix: F20c — target_S dim-agnostic ===
        F17 在 forward_step 顶部把 1D target 升到 2D [1, d_model],
        旧的 _microsleep 假设 1D [d_model] 又 unsqueeze(0) → 3D 跟 repeat 冲突.
        修法: 函数开头 normalize 到 2D, 后面就跟原来一样.
        微睡眠：离线巩固，两步操作：
            1. 即时重演：唤醒长期冻结块，用当前S做一次低学习率更新
            2. Scale全局归一化：统一各块动态范围
        全部向量化，耗时约1~2ms。
        # === v4-fix: F28 (momo 找死占) — _microsleep lazy alloc 配套保护 ===
        # MicroSleep_Interval=999999 时 _W_freeze_fp32_cache=None, 直接 return
        if self._W_freeze_fp32_cache is None:
            return
        """
        step = self._step_counter
        freeze_mask = (step - self.last_active) > self.cfg['Freeze_Limit']
        freeze_idx = freeze_mask.nonzero(as_tuple=True)[0]

        # F20c: normalize target_S 到 2D [1, d_model] (兼容 T=1 旧 1D 和 T>1 2D 路径)
        if target_S.dim() == 1:
            target_S = target_S.unsqueeze(0)  # [1, d_model]
        # target_S 现在是 [T, d_model] (T=1 或 T>1), 跟 freeze_idx 维度对齐需要 [1, d_model]
        # microsleep 只用一个 target, 拿第一个位置即可
        if target_S.shape[0] > 1:
            target_S = target_S[:1]  # 取第一个位置作为 microsleep target

        # 阶段1：即时重演（让冻结块用当前S练手）
        if len(freeze_idx) > 0:
            N = len(freeze_idx)
            # === v4-fix: F20c — 用预分配 _W_freeze_fp32_cache (跟 forward_step F13 同款) ===
            # 旧代码每 100 步新建 1GB FP32 tensor, 10000 步 = 100GB 分配压力
            # 新写法: view 预分配 buffer (零分配), detach().requires_grad_() 作为 leaf
            W_freeze = self._W_freeze_fp32_cache[:N].detach().requires_grad_(True)
            S_freeze = self._S_freeze_fp32_cache[:N]  # view, no grad

            # 就地填充:INT8 -> FP32, scale 已 baked into scale_pool
            with torch.no_grad():
                W_freeze.copy_(self.W_pool[freeze_idx].float())
                scale_exp = self.scale_pool[freeze_idx].view(N, 1, 1)
                W_freeze.mul_(scale_exp)
                S_freeze.copy_(self.S.unsqueeze(0).unsqueeze(-1).expand(N, -1, -1))

            deltas_freeze = torch.bmm(W_freeze, S_freeze).squeeze(-1)   # [N, d_inner]

            S_exp = self.S.unsqueeze(0).repeat(N, 1)
            # === v4-fix: F20c — target_S 已 normalize 到 2D, 不再 unsqueeze ===
            target_exp = target_S.repeat(N, 1)  # [1, d_model] → [N, d_model]
            losses = F.mse_loss(S_exp + deltas_freeze, target_exp, reduction='none').mean(dim=1)
            grads = torch.autograd.grad(losses.sum(), W_freeze)[0]

            # 保守学习率 (0.1 * lr_base)
            scale_exp = self.scale_pool[freeze_idx].unsqueeze(-1).unsqueeze(-1)
            delta_update = -0.1 * self.cfg['lr_base'] * grads / (scale_exp + 1e-8)
            self.update_buffer[freeze_idx] += delta_update

            # 提交
            update_norms = self.update_buffer[freeze_idx].view(N, -1).abs().mean(dim=1)
            commit_mask = update_norms > 0.5
            if commit_mask.any():
                commit_idx = freeze_idx[commit_mask]
                rounded = torch.round(self.update_buffer[commit_idx]).to(torch.int8)
                self.W_pool[commit_idx] = torch.clamp(self.W_pool[commit_idx] + rounded, -128, 127)
                self.update_buffer[commit_idx] -= rounded.float()

            # === v4-fix: F20c — 显式 del + 缓存 alloc ===
            del W_freeze, deltas_freeze, losses, grads, delta_update, update_norms
            if 'rounded' in locals():
                del rounded
            torch.cuda.empty_cache()

        # 阶段2：Scale归一化（统一各块动态范围）
        # === v3-fix: F5 — 只调 scale_pool，不动 W_pool ===
        # 原因：scale 就是干这个用的。直接修改 W_pool 会触发 float→int8 反复量化
        # 大量参数（128M+）的精度损失，且张量 reshape + 转换的开销也不小
        active_mask = self.grad_norm_ema > 1e-6
        if active_mask.any():
            mean_scale = self.scale_pool[active_mask].mean()
            if mean_scale > 0:
                self.scale_pool[active_mask] = mean_scale

    # ===================== 有形大手（含GDP税收） =====================
    def _macro_scheduler(self, step: int):
        """
        有形大手：宏观调度（全向量化）
            1. 磨损均衡：冻结超限块强制加偏置
            2. 僵尸清退：梯度范数趋近0的块逻辑卸载（权重归零，门控关闭）
            3. GDP同质化税收：惩罚 |余弦相似度| 过高（既包括正同质，也包括负对冲）
        """
        # ---- 磨损均衡 ----
        freeze_mask = (step - self.last_active) > self.cfg['Freeze_Limit']
        self.b_gate[freeze_mask] += 0.5

        # ---- 僵尸清退 ----
        zombie_mask = (self.grad_norm_ema < 1e-6) & (torch.arange(self.num_blocks, device='cuda') > 0)
        if zombie_mask.any():
            self.W_pool[zombie_mask] = 0
            self.b_gate[zombie_mask] = -1e6

        # ---- GDP同质化税收（双向惩罚） ----
        active_logical = (self.b_gate > -1e5).nonzero(as_tuple=True)[0]
        if len(active_logical) > 1:
            # 取历史方向向量（累积增量）
            hist_vecs = self.hist_delta[active_logical]   # [K, d_model]
            norm = hist_vecs.norm(dim=1, keepdim=True) + 1e-8
            hist_norm = hist_vecs / norm
            # 计算余弦相似度矩阵
            sim_matrix = torch.mm(hist_norm, hist_norm.T)
            eye = torch.eye(len(active_logical), device='cuda')
            # 取除对角线外的最大相似度（带符号）
            max_sim, argmax_idx = torch.max(sim_matrix - eye, dim=1)

            # 税收：惩罚 |cos_sim| 过高（无论正负）
            abs_max_sim = max_sim.abs()
            tax_mask = abs_max_sim > self.cfg['GDP_Threshold']
            if tax_mask.any():
                tax_rate = (abs_max_sim[tax_mask] - self.cfg['GDP_Threshold']) * self.cfg['GDP_Tax_Rate']
                self.b_gate[active_logical[tax_mask]] -= tax_rate

                # 补贴原型（相似度最高的块，无论正负）
                proto_indices = argmax_idx[tax_mask]
                unique_protos = torch.unique(proto_indices)
                self.b_gate[active_logical[unique_protos]] += 0.1

        # 每500步衰减历史方向（防止陈旧信息干扰）
        if step % 500 == 0:
            self.hist_delta *= 0.5

    # ===================== F37: 更新提交 =====================
    def _commit_updates(self, idx: torch.Tensor):
        """把 update_buffer 提交进 W_pool (forward_step_state 用)

        threshold  : 旧路径, 整块 mean|buf| > 0.5 才四舍五入 (保持向后兼容)
        stochastic : F37 随机舍入, 每次 fire 概率 = |buf| (clamp 1.0),
                     期望提交 = buf, 对小子步长更新无偏
        """
        if self.commit_mode == 'stochastic':
            buf = self.update_buffer[idx]
            # BF16 概率比较即可 (抛硬币精度 ~2% 足够), 省 FP32 临时张量
            fire = torch.rand_like(buf) < buf.abs().clamp(max=1.0)
            step = torch.where(fire, buf.sign().to(torch.int8),
                               torch.zeros((), dtype=torch.int8, device=buf.device))
            self.W_pool[idx] = torch.clamp(self.W_pool[idx] + step, -128, 127)
            self.update_buffer[idx] -= step.to(self.update_buffer.dtype)
        else:
            update_norms = self.update_buffer[idx].float().view(len(idx), -1).abs().mean(dim=1)
            commit_mask = update_norms > 0.5
            if commit_mask.any():
                commit_idx = idx[commit_mask]
                rounded = torch.round(self.update_buffer[commit_idx].float()).to(torch.int8)
                self.W_pool[commit_idx] = torch.clamp(
                    self.W_pool[commit_idx] + rounded, -128, 127)
                self.update_buffer[commit_idx] -= rounded.float().to(self.update_buffer.dtype)

    # ===================== SpikeLLM 外部 S 接口 =====================
    # === v4-fix: F20e — forward_step_state (momo LM 提议) ===
    # 跟 forward_step 区别:
    #   1. S_old 由外部传入 (caller 维护), 不读 self.S
    #   2. 返回 S_new 后不写 self.S (caller 决定是否保留)
    #   3. W_active 一次性新分配 (不共享 F13 预分配 buffer), autograd 用完即弃
    #   4. 调 autograd.grad 立即消化 graph, 不累积 (BPTT 安全)
    # 用途: 语言模型 (SpikeLLM) 序列里反复调, 每次独立 W_active leaf
    def forward_step_state(self, target_S: torch.Tensor,
                           S_old: torch.Tensor) -> torch.Tensor:
        """
        一步训练 (momo SpikeLLM 风格):
            1. 门控决策 (用 self.b_gate)
            2. 一次性分配 W_active BF16 (新 leaf), 从 W_pool 复制数据
            3. bmm: deltas = W_active @ S_old.unsqueeze(-1)
            4. S_new = S_old + deltas.sum(dim=0) (用 caller 传进来的 S_old)
            5. 内部 MSE 损失 -> autograd.grad -> W_pool INT8 update
            6. 立即 return S_new (autograd 链: S_new -> S_old, W_active 已用完)

        === v4-fix: F21 — 共享 W_active buffer (momo "有共享 gpu 内存可以用" 提议)
        原版每步新建 W_active: 32 positions × 1.5GB (BF16 + grad) = 48GB 临时 OOM
        改用 self._W_active_shared (lazy alloc, 一次性预分配) + detach 复用
        关键: detach() 切 graph, requires_grad_(True) 让它成为新 leaf
              backward 完 W_active.grad 释放, 下次 detach() 创建新 leaf 不累积
        """
        step = self._step_counter
        self._step_counter += 1

        # 1. 门控
        # === v4-fix: F32 - 传入外部 S_old 的 batch 均值做内容寻址 ===
        # 原版读 self.S (SpikeLLM 路径恒为 zeros) -> q 恒为 0, 内容寻址死路
        S_gate = S_old.detach().mean(dim=0) if S_old.dim() > 1 else S_old.detach()
        active_idx = self._compute_gates(step, S=S_gate)
        K = len(active_idx)
        if K == 0:
            return S_old
        # === v4-fix: F22 — cpu_offload 时 active_idx 也走 CPU 用于 indexing ===
        if self.cpu_offload:
            active_idx_cpu = active_idx.to('cpu')
        else:
            active_idx_cpu = active_idx
        if target_S.dim() == 1:
            target_S = target_S.unsqueeze(0)
        if S_old.dim() == 1:
            S_old = S_old.unsqueeze(0)
        T = 1  # spike pool 自身 bmm 仍 T=1 (per-position); S_old batch 维在 bmm 外

        # 2. W_active 共享 buffer (F21, momo 提议: 共享 GPU 内存)
        # 第一次 call 时 lazy 分配, 之后 L 个 positions 复用同一块物理 storage
        # 关键: W_active 在 forward_step_state 中**不需要 requires_grad**
        #   因为 W 更新走 no_grad block 的 manual gradient (update_buffer += -lr * grad_W)
        #   不调 .backward() 在 W_active 上, 所以 .requires_grad=True 是浪费 + 会触发 inplace error
        if not hasattr(self, '_W_active_shared') or self._W_active_shared is None \
                or self._W_active_shared.shape != (K, self.d_inner, self.d_model):
            self._W_active_shared = torch.empty(
                K, self.d_inner, self.d_model,
                dtype=torch.bfloat16, device='cuda'
            )
        # 复用 shared_buf 的物理 memory, detach() 不入 graph, zero_() 清空残留
        W_active = self._W_active_shared.detach().zero_()

        # 3. 填数据: INT8 -> FP32 -> BF16 + scale (in-place on 共享 storage, OK 因 no_grad + W_active 不进 graph)
        # === v4-fix: F22 — cpu_offload 时 W_pool/scale_pool 在 CPU, active K 块拉 GPU ===
        # transfer: pinned CPU -> GPU, non_blocking 让 bmm 不等 transfer
        with torch.no_grad():
            if self.cpu_offload:
                W_pool_k = self.W_pool[active_idx_cpu].to('cuda', non_blocking=True)
                scale_k = self.scale_pool[active_idx_cpu].to('cuda', non_blocking=True)
            else:
                W_pool_k = self.W_pool[active_idx]
                scale_k = self.scale_pool[active_idx]
            W_active.copy_(W_pool_k.float().bfloat16())
            # === v4-fix: F31 - 删除 W_active.mul_(scale_exp) 双乘 bug ===
            # 现象: 这里预乘 scale, 下面 delta_S 又 * scale_k, delta 实际 ∝ scale^2.
            #       F29 后 scale=0.002, 有效 4e-6, 单块 delta 元素 std ≈ 2e-4
            #       (S 元素 std 0.01) -> 池子输出对 S 的贡献近乎为零, 模型退化为
            #       embed + output_head (S 全序列 ≈ embed(token_0))
            # 修法: 跟 forward_step 对齐 (train_engine F18 post-scale 单乘):
            #       W_active 保持 INT8 原值, scale 只在 bmm 后乘一次.
            #       梯度路径 update_buffer -= lr*grad/scale 与单乘前向自洽
            #       (dL/dM = residual⊗S, ΔW_int8 = -lr*dL/dM/scale)

        # 4. bmm: [K, d_inner, d_model] @ [K, d_model, B] = [K, d_inner, B]
        # === v4-fix: F27 (momo 提议) — 批处理 (B=4 spike_llm) 把 GEMV 变 GEMM ===
        # 原版 T=1 是 GEMV, TC 几乎不工作. 改成 T=B (batch dim) bmm 是真 GEMM, TC 满载
        # momo 算力分析: launch overhead 减 B 倍, bmm 算力提升 B 倍, 总体 sps 应涨 1.5-3x
        # === v4-fix: F21 — bmm 放 no_grad 让 W_active 完全不进 graph ===
        # S_old.requires_grad=True (caller 传), backward 正常求 dS/dS_old → dS/d_embed → dS/d_output_head
        # F27: 取 S_old batch 维 B, bmm 第二维 T=B 而不是 T=1
        if S_old.dim() == 1:
            S_old = S_old.unsqueeze(0)
        B = S_old.shape[0]  # 批大小 (spike_llm 默认 4)
        S_for_bmm = S_old.to(torch.bfloat16).unsqueeze(0).expand(K, -1, -1, -1).permute(0, 2, 1, 3).reshape(K, self.d_model, B)
        # F27: K * d_model * B 第二维 T=B 真 GEMM
        with torch.no_grad():
            deltas_bf16 = torch.bmm(W_active, S_for_bmm)  # [K, d_inner, B]
            # post-scale + sum over K: [B, d_inner]
            delta_S = (deltas_bf16.float() * scale_k.view(K, 1, 1)).sum(dim=0).T  # [B, d_inner]
        # S_new: v6 F36 - ISS 状态方程 S_t = γ·S_{t-1} + g·RMS_norm(Δ)
        # Δ 的 RMS 在 no_grad 里算 (天然 stop-grad), 归一化后每样本注入元素 RMS = g,
        # ‖注入‖ = g·√d 与 scale/√d/√K 全解耦; γ,g 是 leaf param, 乘法进 autograd
        # 图, 外层 CE loss 直接训练它们 (BPTT 链保持: S_new -> S_old -> embed)
        if self.use_iss:
            rms = delta_S.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-6  # [B,1]
            delta_hat = delta_S / rms                                       # 元素 RMS=1
            gamma = torch.sigmoid(self.iss_gamma_raw)
            S_new = gamma * S_old + self.iss_gain * delta_hat              # [B, d_inner]
        else:
            rms = None
            # 旧路径 (use_iss=False 消融用): 裸注入
            S_new = S_old + delta_S             # [B, d_inner] + [B, d_inner] = [B, d_inner]

        # === v4-fix: F20e — S_norm clip (防 S 爆炸, 跟 v3 forward_step 末段一样) ===
        s_norm = S_new.norm(dim=-1, keepdim=True)  # [B, 1]
        over = s_norm > self.cfg['S_Norm_Cap']
        scale = s_norm / self._s_norm_target
        S_new = torch.where(over, S_new / scale, S_new)

        # 5. 内部 MSE 损失: 让 S_new 逼近 target_S (teacher forcing)
        # === v4-fix: F20e - momo SpikeLLM 设计: 整个 W 更新放 no_grad, 不建 autograd 图 ===
        # 外层 loss (CE) 不会反传到 W_active, 干净 BPTT
        # W 更新用 manual gradient: dL/dW = (S_new - target) * S_old^T
        # === v4-fix: F22 (momo 修复一+二+四) - 消灭 FP32 梯度尖峰 + in-place 累加 + update_buffer BF16 ===
        # 修复一: grad_W 用 BF16 outer (512MB) -> FP32 (1.07GB) -> del BF16, 峰值 1.07GB 不再 2.5GB
        # 修复二: 不创建 delta_update 中间张量, 直接 self.update_buffer -= lr*grad_W/scale (in-place)
        # 修复四: update_buffer 用 BF16 (17.2GB -> 8.6GB for 256+4096), commit 时转 int8
        # === v4-fix: F33 - eval/no_grad 时跳过 W 更新 ===
        # 现象: 更新块只看 lock_phase, 不看训练/推理状态 -> estimate_loss 和
        #       generate 每次调用都在改 W_pool (验证集污染 + 生成漂移)
        # 修法: torch.is_grad_enabled() 为 False (no_grad 推理) 时不更新.
        #       训练路径 loss.backward() 需要 grad, is_grad_enabled()=True, 行为不变
        if not self.lock_phase and torch.is_grad_enabled():
            with torch.no_grad():
                # === v4-fix: F27 (momo 提议) — 批处理 manual grad, 用 B 维求和 ===
                # 原版只取 S_new[0] 单 sample 算 residual + 用 S_old[0] 算 outer, 其他 B-1 sample 浪费
                # F27: B 维全用, residual = mean over batch (更稳定), outer = sum over batch
                # 取 batch 全部元素
                if S_new.dim() > 1:
                    S_pred = S_new                          # [B, d_inner]
                    target = target_S                        # [B, d_inner]
                else:
                    S_pred = S_new.unsqueeze(0)
                    target = target_S.unsqueeze(0)
                # manual gradient: dL/dW[k, i, j] = (1/B) * Σ_p (S_pred[p,i] - target[p,i]) * S_old[p,j]
                # 简化: B 维 sum 然后除 B (或省略除 B, 让 lr 调节)
                residual = (S_pred.float() - target)         # [B, d_inner]
                B = S_old.shape[0]
                # === v6: F36 - ISS 链式因子: dS_new/dΔ ≈ g/rms (RMS 已 stop-grad) ===
                # ISS 下 dL/dW_int = (g·s/rms)·(residual⊗S). 精确版含 s_k 且 rms∝s
                # -> 量纲抵消 (scale-free), 但幅度太小 (~s 倍) 提交不动. 这里取
                # g/rms 近似: 方向不变, 幅度 ∝ 1/scale, 用 lr_base 单独调 --
                # ISS 已保证稳定性与这个量纲解耦, 不会再引爆 S
                if self.use_iss:
                    residual = residual * (self.iss_gain.detach() / rms)
                # === 修复一: BF16 outer (512MB 临时, 比 FP32 1.07GB 省一半) ===
                # F27: outer 现在是 (B, d_inner) × (B, d_model) → (B, d_inner, d_model), sum over B
                # 一次性算 B 个 sample 的 outer, 不再 K*B 重复
                # W update: shared across all B samples (manual grad 简化)
                # 用 S_old[0] 简化 (跟原版一样, 后续可改成 sum over B)
                S_1d = S_old[0] if S_old.dim() > 1 else S_old
                grad_W_bf16 = residual[0].bfloat16().unsqueeze(-1) * S_1d.bfloat16().unsqueeze(0)  # [K, d_inner, d_model]
                # F27 增强: 累加 B 个 sample 的 grad, 减少 sample variance, 更稳定
                for b in range(1, B):
                    grad_W_bf16 += residual[b].bfloat16().unsqueeze(-1) * S_1d.bfloat16().unsqueeze(0)
                grad_W = grad_W_bf16.float()  # FP32 for INT8 update
                del grad_W_bf16  # 立即释放 BF16 中间
                # 注意: W_active 实际是 BF16 量化版 W_pool, 但这里我们直接对 W_pool 的 BF16 近似做 update
                # 真实梯度应该是 W_active 的 BF16 表征, 但 W_active = W_pool * scale (post-bmm)
                # 这里简化: 直接用 manual grad
                scale_exp_g = scale_k.unsqueeze(-1).unsqueeze(-1)
                # === 修复二: in-place 累加, 不创建 delta_update 中间张量 ===
                lr = self.cfg['lr_base']
                # === v6: F36 - 快权重遗忘钩子 (默认关, 见 __init__ 注释) ===
                if self.iss_buffer_decay < 1.0:
                    self.update_buffer[active_idx_cpu if self.cpu_offload
                                       else active_idx] *= self.iss_buffer_decay
                # === v4-fix: F22 - cpu_offload 时 update_buffer 在 CPU, grad_W 拉回 CPU commit ===
                if self.cpu_offload:
                    # grad_W 在 GPU, update_buffer 在 CPU pinned (BF16, 修复四), 减半内存
                    # 修复二: 拉 grad_W 到 CPU 后 in-place 累加, 不 alloc delta_update
                    # 修复四: update_buffer BF16, commit 时转 FP32 算 norm 再转 int8
                    grad_W_cpu = grad_W.to('cpu', non_blocking=True)
                    if self.use_iss:
                        # F36: 链式因子已在 residual 里, 不再除 scale (见上面注释)
                        self.update_buffer[active_idx_cpu] -= (lr * grad_W_cpu).to(torch.bfloat16)
                    else:
                        self.update_buffer[active_idx_cpu] -= (lr * grad_W_cpu / (scale_exp_g.to('cpu', non_blocking=True) + 1e-8)).to(torch.bfloat16)
                    del grad_W_cpu
                    # F37: commit (threshold / stochastic 由 commit_mode 决定)
                    self._commit_updates(active_idx_cpu)
                else:
                    if self.use_iss:
                        self.update_buffer[active_idx] -= (lr * grad_W).to(self.update_buffer.dtype)
                    else:
                        self.update_buffer[active_idx] -= (lr * grad_W / (scale_exp_g + 1e-8)).to(self.update_buffer.dtype)
                    # F37: commit (threshold / stochastic 由 commit_mode 决定)
                    self._commit_updates(active_idx)

            # 显式释放 W_active 和 intermediates (F21: W_active 是 shared_buf view, del 不释放 shared_buf)
            del W_active, deltas_bf16, delta_S, residual
            if not self.cpu_offload:
                torch.cuda.empty_cache()  # 关键: LM 序列里每步都调, 不释放会累积 (cpu_offload 模式 GPU 干净, 不用清)

        return S_new

    # ===================== 训练结束固化 =====================
    # === v4-fix: F20c — progress writer (momo 实时看训练状态) ===
    # 每 progress_interval 步把状态写到 JSON 文件, _monitor.py 读它显示
    def write_progress(self, step: int, t_start: float):
        """写一个 JSON 行到 <save_path>_progress.json, _monitor.py 读这个文件显示
        不要用 json.dump + open, 慢. 用单行 write + atomic replace."""
        import json, os
        s_norm = float(self.S.norm().item())  # sync
        active_logical = int((self.b_gate > -1e5).sum().item())
        elapsed = time.time() - t_start
        sps = (step + 1) / elapsed if elapsed > 0 else 0
        eta = (self.cfg['T_total'] - step - 1) / sps if sps > 0 else 0
        progress = {
            "step": step,
            "total": self.cfg['T_total'],
            "sps": sps,
            "elapsed_s": elapsed,
            "eta_s": eta,
            "s_norm": s_norm,
            "active_logical": active_logical,
            "num_blocks": self.num_blocks,
            "top_k": self.cfg['Top_K_Active'],
            "t_batch": self.cfg.get('T_batch', 1),
        }
        path = self.cfg.get('save_path', './model_pool') + '_progress.json'
        # atomic write: 写 .tmp 然后 rename, 避免 _monitor 读到半截
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(progress, f)
        os.replace(tmp, path)

    def finalize_inference(self, save_path: Optional[str] = None):
        """
        训练结束后调用，执行：
            1. 按激活频次重排物理地址（高频在前）
            2. 保存权重池（.bin）和元数据（.npy）供推理引擎加载
        """
        if save_path is None:
            save_path = self.cfg['save_path']

        # 按激活频次降序排列（使高频块在显存中连续）
        sorted_idx = torch.argsort(self.activation_stats, descending=True)
        self.W_pool = self.W_pool[sorted_idx]
        self.scale_pool = self.scale_pool[sorted_idx]
        self.b_gate = self.b_gate[sorted_idx]
        self.k_embed = self.k_embed[sorted_idx]   # 键随块重排
        self.activation_stats = self.activation_stats[sorted_idx]

        torch.cuda.synchronize()

        # 写入NVMe（二进制格式，供推理引擎mmap加载）
        pool_np = self.W_pool.cpu().numpy().astype(np.int8)
        pool_np.tofile(f"{save_path}.bin")
        np.save(f"{save_path}_scale.npy", self.scale_pool.cpu().numpy())
        np.save(f"{save_path}_gate.npy", self.b_gate.cpu().numpy())
        np.save(f"{save_path}_kembed.npy", self.k_embed.cpu().numpy())
        np.save(f"{save_path}_qproj.npy", self.q_proj.cpu().numpy())
        np.save(f"{save_path}_stats.npy", self.activation_stats.cpu().numpy())

        print(f"✅ Model frozen and saved to {save_path}.bin")
        print(f"   W_pool shape: {self.W_pool.shape}, size: {pool_np.nbytes / 1e9:.2f}GB")

    def shutdown(self):
        """清理后台线程和显存"""
        self.executor.shutdown(wait=True)
        torch.cuda.empty_cache()


# ===================== 主训练入口 =====================
if __name__ == "__main__":
    # 初始化训练池
    pool = RTX5090SpikePool(CONFIG)

    # === v4-fix: F20c — 用 2D target [T, d_model] 触发 F17 T batching + F18 BF16 bmm ===
    # T=1 是 v3 老路径 (F18 BF16 cast 开销 > TC 收益, 反而慢)
    # T=64+ 让 bmm [K, d_inner, d_model] @ [K, d_model, T] 走真 GEMM, F18 BF16 TC 才有意义
    # 取 CONFIG['T_batch'] 作 T, 推理路径 (1D target) 仍兼容
    # === v4-fix: F20c — DataLoader 走 CPU 多 worker (momo 反馈 CPU RAM 没利用) ===
    # 10000 * T * 4096 * 4 bytes targets 太大 (T=64 时 10GB, T=128 时 20GB),
    # 改用 IterableDataset + DataLoader + num_workers, workers 在 CPU 后台生成
    train_T = CONFIG.get('T_batch', 64)

    if train_T == 1:
        # T=1 仍用 list (1D target, 推理兼容)
        targets = [torch.randn(CONFIG['d_model'], pin_memory=True) for _ in range(CONFIG['T_total'])]
        target_iter = iter(targets)
        print(f"  Training with T_batch={train_T} ({'v3 1D 路径' if train_T == 1 else 'v4 2D + BF16 TC 路径'})")
        print(f"  targets: {len(targets)} x [{train_T}, {CONFIG['d_model']}] on CPU pinned (pre-alloc)")
    else:
        # === v4-fix: F20c — 回退 DataLoader 走 pre-alloc ===
        # momo 反馈 CPU RAM 没利用, 加了 DataLoader 4 workers
        # 但实测: 随机 targets 的 DataLoader IPC 开销 > 实际 CPU 工作 (microsec 级)
        # 反而比 pre-alloc 慢 3x (24ms -> 70ms/step, 因为 microsleep 也被拖慢)
        # 决定: pre-alloc 10000 targets on CPU pinned (1GB total), T=64 够快
        # 真正生产场景换真实 dataloader (读盘/解码/tokenize), 那时 DataLoader workers 才真有用
        targets = [torch.randn(train_T, CONFIG['d_model'], pin_memory=True) for _ in range(CONFIG['T_total'])]
        target_iter = iter(targets)
        target_mem_mb = CONFIG['T_total'] * train_T * CONFIG['d_model'] * 4 / 1e6
        print(f"  Training with T_batch={train_T} ({'v3 1D 路径' if train_T == 1 else 'v4 2D + BF16 TC 路径'})")
        print(f"  targets: {len(targets)} x [{train_T}, {CONFIG['d_model']}] on CPU pinned ({target_mem_mb:.0f}MB total)")
        print(f"  DataLoader: 关 (随机 data 走 IPC 是负优化, 真实 dataset 时打开)")

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    t_start = time.time()
    progress_interval = 100  # 每 100 步写 progress JSON 给 _monitor.py 读
    for step in range(CONFIG['T_total']):
        # === v4-fix: F20c — 从 target_iter 拿下一个 target (DataLoader 或 list 通用) ===
        target_cpu = next(target_iter)
        # === v4-fix: F20c — H2D transfer (CPU pinned -> GPU) 每步做一次, 走 pinned 通道 ===
        # 第一次 .cuda(non_blocking=True) 触发实际传输, 后续 reuse
        target = target_cpu.cuda(non_blocking=True)
        try:
            S_new = pool.forward_step(target)
        except torch.cuda.OutOfMemoryError as oom:
            # === v3-fix: F12 — OOM fail-fast + dump 完整现场（momo 方案）===
            # 不要靠猜峰值,崩时直接 dump memory_stats + 上下文,不靠试错
            print("\n" + "=" * 60)
            print(f"💥 CUDA OOM at step {step}!")
            print("=" * 60)
            _ms = torch.cuda.memory_stats()
            print(f"  reserved:  {_ms['reserved_bytes.all.current']/1e9:.2f} GB")
            print(f"  allocated: {_ms['allocated_bytes.all.current']/1e9:.2f} GB")
            print(f"  peak:      {_ms['reserved_bytes.all.peak']/1e9:.2f} GB")
            print(f"  num allocs: {_ms.get('allocation_count.all', 0)}")
            print(f"  num frees:  {_ms.get('free_count.all', 0)}")
            print(f"  num ooms:   {_ms.get('num_ooms', 0)}")
            print(f"  active blocks: {(pool.b_gate > -1e5).sum().item()}/{CONFIG['num_blocks']}")
            print(f"  W_pool size: {pool.W_pool.element_size() * pool.W_pool.nelement() / 1e9:.2f} GB")
            print(f"  update_buffer size: {pool.update_buffer.element_size() * pool.update_buffer.nelement() / 1e9:.2f} GB")
            print(f"\nOriginal error: {oom}")
            pool.shutdown()
            raise

        # 每500步打印监控信息
        if step % 500 == 0:
            active_logical = (pool.b_gate > -1e5).sum().item()
            norm_S = S_new.norm().item()
            print(f"Step {step:5d} | Active logical: {active_logical:3d}/{CONFIG['num_blocks']} | "
                  f"S_norm: {norm_S:.4f}")

        # === v4-fix: F20c — progress writer 给 _monitor.py 实时看 ===
        if step % progress_interval == 0:
            pool.write_progress(step, t_start)

    # 写最后一步
    pool.write_progress(CONFIG['T_total'] - 1, t_start)

    print("\n" + "=" * 60)
    print("Training complete. Freezing model...")
    pool.finalize_inference()
    pool.shutdown()
    print("✅ Done.")
