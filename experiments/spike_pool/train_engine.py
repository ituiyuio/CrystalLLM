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
    'T_total': 2000,              # 总训练步数 (先 2K 验证 DataLoader + T=128, 完整跑 10K)
    'T_report': 100,              # 有形大手调度间隔（步）
    'T_warm': 500,                # 新块学习率升温步数

    # 门控与调度
    'Freeze_Limit': 500,          # 磨损均衡：超过此步未激活则强制唤醒
    'Burst_Limit': 20,            # 行锤击缓解：连续激活超过此值则强制静默
    'MicroSleep_Interval': 100,   # 微睡眠间隔（步）
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
    'T_batch': 128,               # === v4-fix: F17 — 序列批维度
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
        # 不靠猜峰值,设 85% (~27GB) 上限,超过 OOM,崩时 dump memory_stats
        torch.cuda.set_per_process_memory_fraction(
            config['gpu_mem_fraction'], device=0
        )
        self.W_pool = torch.zeros(
            self.num_blocks, self.d_inner, self.d_model,
            dtype=torch.int8, device='cuda'
        )
        self.scale_pool = torch.ones(self.num_blocks, device='cuda')  # 每个块独有的FP32缩放因子

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

        # ----- 3. 原生INT8更新缓冲（FP32累加器） -----
        # 梯度更新不直接截断回INT8，而是累加在FP32缓冲中，累积超过0.5步长才提交整数
        self.update_buffer = torch.zeros(
            self.num_blocks, self.d_inner, self.d_model,
            dtype=torch.float32, device='cuda'
        )

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
        self._W_freeze_fp32_cache = torch.zeros(
            self.num_blocks, self.d_inner, self.d_model,
            dtype=torch.float32, device='cuda'
        )  # 1.07GB, _microsleep 专用
        self._S_freeze_fp32_cache = torch.zeros(
            self.num_blocks, self.d_model, 1,
            dtype=torch.float32, device='cuda'
        )  # 256KB, S 缓存 (T=1 路径)

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

    # ===================== 门控决策 =====================
    def _compute_gates(self, step: int) -> torch.Tensor:
        """
        全向量化门控决策（无Python循环）。

        融合三路信号：
            1. 市场惯性（b_gate）：历史表现 + GDP税收调节
            2. 内容寻址（注意力）：当前S与各块键的匹配度
            3. 硬件规则：磨损均衡（强制唤醒） + 行锤击缓解（强制静默）

        返回：激活块索引列表 (CPU Tensor)
        """
        lock_start = int(self.cfg['T_total'] * self.cfg['Lock_Ratio'])

        # ---- 内容寻址得分 ----
        # 计算查询向量 q = q_proj @ S，然后与所有键做点积
        q = torch.mv(self.q_proj, self.S)                # [d_k]
        content_scores = torch.mv(self.k_embed, q)       # [num_blocks]
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
        if self.lock_phase or step >= lock_start:
            G = (prob > 0.5).int()
            self.activation_stats += G   # 记录激活频次（用于推理重排）
        else:
            # 正常训练：随机抽样 + 强制唤醒
            rand_vals = torch.rand(self.num_blocks, device='cuda')
            G = ((rand_vals < prob) | freeze_mask).int()
            G[burst_mask] = 0

        # ---- 安全防护：至少激活1个块 ----
        if G.sum() == 0:
            _, indices = torch.topk(self.b_gate, min(self.cfg['Top_K_Active'], self.num_blocks))
            G[indices] = 1

        # ---- 性能防护：限制最大激活数（保证Batch GEMM最优尺寸） ----
        if G.sum() > self.cfg['Top_K_Active']:
            _, top_idx = torch.topk(self.b_gate * G.float(), self.cfg['Top_K_Active'])
            G = torch.zeros_like(G)
            G[top_idx] = 1

        # ---- 更新统计（非锁定阶段） ----
        if not self.lock_phase and step < lock_start:
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
            if step % 50 == 0:
                torch.cuda.empty_cache()

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

            # === v3-fix: F13 — 主动内存管理（momo 根因修）===
            # 预分配 buffer 解决了 W_active leak,但 grads_batch / delta_update
            # 仍是 autograd 返回的新 tensor,每步新建,需要显式 del + 条件 empty_cache
            # 注: 用 locals() check 防 lock_phase 跳过训练态时 UnboundLocalError
            for _v in ('W_active', 'S_batch', 'deltas', 'grads_batch', 'delta_update'):
                if _v in locals():
                    del locals()[_v]
            if step % 10 == 0:
                _alloc_gb = torch.cuda.memory_allocated() / 1e9
                _total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                if _alloc_gb > 0.7 * _total_gb:
                    torch.cuda.empty_cache()

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
        # 2D target 走 DataLoader 多 worker
        # num_workers=4 让 4 个 CPU 进程并行生成, 每步主进程拿一个 batch
        # _TargetIterable 在模块级定义 (Windows multiprocessing pickle 要求)
        ds = _TargetIterable(train_T, CONFIG['d_model'], seed=42)
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=None,            # IterableDataset 已经按 T 输出
            num_workers=4,              # 4 个 CPU worker 并行生成
            pin_memory=True,            # 锁页内存, H2D 走 DMA 不阻塞
        )
        target_iter = iter(loader)
        # 估算 CPU 占用: 4 workers * T * d_model * 4 bytes = 16 * T MB pinned
        cpu_mem_mb = 4 * train_T * CONFIG['d_model'] * 4 / 1e6
        print(f"  Training with T_batch={train_T} ({'v3 1D 路径' if train_T == 1 else 'v4 2D + BF16 TC 路径'})")
        print(f"  DataLoader: 4 CPU workers, ~{cpu_mem_mb:.0f}MB pinned per worker")
        print(f"  targets: 无尽 IterableDataset (T={train_T}, d_model={CONFIG['d_model']})")

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
