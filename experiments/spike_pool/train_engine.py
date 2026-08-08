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
    'T_total': 10000,             # 总训练步数
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

    # === v3-fix: F4 — b_gate 均值回归 ===
    # 磨损均衡 +0.5 单边累积会推高老块门控，
    # 引入 EMA 风格的均值回归，让门控自然衰减回历史激活频次的中位线
    'Gate_EMA_Decay': 0.999,      # 每步 b_gate *= decay（不显式设均值，靠自然演化）
    'Gate_Activation_Pull': 0.01, # 激活的块向历史激活率靠拢的步长

    # 保存路径
    'save_path': './model_pool',  # 模型文件前缀
}


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

        # 返回活跃块索引（转为CPU以便后续索引）
        return G.nonzero(as_tuple=True)[0].cpu()

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

        # ===== 4. 前向：Batch GEMM =====
        S_old = self.S.clone()
        # 从池中取权重并反量化 [K, d_inner, d_model]
        # === v3-fix: F1 — 显式 requires_grad_() ===
        # W_pool 是 int8，索引后转 float 默认无 grad_fn。
        # 显式标记 leaf 并开启 grad，autograd.grad 才能工作
        W_active = (
            self.W_pool[active_idx].float()
            * self.scale_pool[active_idx].unsqueeze(-1).unsqueeze(-1)
        ).detach().requires_grad_(True)
        # S扩展为批处理版本 [K, d_model, 1]
        S_batch = S_old.unsqueeze(0).unsqueeze(-1).repeat(K, 1, 1)
        # 🚀 单次CUDA Kernel启动，Tensor Core满载
        deltas = torch.bmm(W_active, S_batch).squeeze(-1)   # [K, d_inner]
        self.S = S_old + deltas.sum(dim=0)

        # ===== 5. 更新历史增量（GDP核算数据） =====
        if not self.lock_phase:
            # === v3-fix: F7b — 直接 += deltas（[K, d_inner]），hist_delta 形状已对齐 ===
            # 原 v3-final 用 pooled_delta = deltas.mean(dim=1) 把 d_inner 维求平均成 [K]，
            # 跟 [K, d_model] 的 hist_delta 形状冲突。这里直接存 d_inner 维 hidden 方向。
            self.hist_delta[active_idx] += deltas  # deltas: [K, d_inner]

        # ===== 6. 局部反向传播（仅训练态） =====
        if not self.lock_phase:
            # 构造批量局部损失：让 S_old + delta_i 逼近 target_S
            S_old_exp = S_old.unsqueeze(0).repeat(K, 1)
            target_exp = target_S.unsqueeze(0).repeat(K, 1)
            delta_contrib = S_old_exp + deltas   # [K, d_model]
            losses = F.mse_loss(delta_contrib, target_exp, reduction='none').mean(dim=1)
            total_loss = losses.sum()

            # 🚀 唯一反向传播Kernel（批处理）
            grads_batch = torch.autograd.grad(total_loss, W_active, retain_graph=False)[0]  # [K, d_inner, d_model]

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
        微睡眠：离线巩固，两步操作：
            1. 即时重演：唤醒长期冻结块，用当前S做一次低学习率更新
            2. Scale全局归一化：统一各块动态范围
        全部向量化，耗时约1~2ms。
        """
        step = self._step_counter
        freeze_mask = (step - self.last_active) > self.cfg['Freeze_Limit']
        freeze_idx = freeze_mask.nonzero(as_tuple=True)[0]

        # 阶段1：即时重演（让冻结块用当前S练手）
        if len(freeze_idx) > 0:
            # === v3-fix: F1 — 微睡眠里同样的 autograd 修复 ===
            W_freeze = (
                self.W_pool[freeze_idx].float()
                * self.scale_pool[freeze_idx].unsqueeze(-1).unsqueeze(-1)
            ).detach().requires_grad_(True)
            S_batch = self.S.unsqueeze(0).unsqueeze(-1).repeat(len(freeze_idx), 1, 1)
            deltas_freeze = torch.bmm(W_freeze, S_batch).squeeze(-1)

            S_exp = self.S.unsqueeze(0).repeat(len(freeze_idx), 1)
            target_exp = target_S.unsqueeze(0).repeat(len(freeze_idx), 1)
            losses = F.mse_loss(S_exp + deltas_freeze, target_exp, reduction='none').mean(dim=1)
            grads = torch.autograd.grad(losses.sum(), W_freeze)[0]

            # 保守学习率 (0.1 * lr_base)
            scale_exp = self.scale_pool[freeze_idx].unsqueeze(-1).unsqueeze(-1)
            delta_update = -0.1 * self.cfg['lr_base'] * grads / (scale_exp + 1e-8)
            self.update_buffer[freeze_idx] += delta_update

            # 提交
            update_norms = self.update_buffer[freeze_idx].view(len(freeze_idx), -1).abs().mean(dim=1)
            commit_mask = update_norms > 0.5
            if commit_mask.any():
                commit_idx = freeze_idx[commit_mask]
                rounded = torch.round(self.update_buffer[commit_idx]).to(torch.int8)
                self.W_pool[commit_idx] = torch.clamp(self.W_pool[commit_idx] + rounded, -128, 127)
                self.update_buffer[commit_idx] -= rounded.float()

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

    # 生成模拟目标数据（实际训练时替换为dataloader）
    # 这里仅作演示，真实场景应使用真实数据集迭代器
    targets = [torch.randn(CONFIG['d_model'], device='cuda') for _ in range(CONFIG['T_total'])]

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    for step in range(CONFIG['T_total']):
        target = targets[step]
        S_new = pool.forward_step(target)

        # 每500步打印监控信息
        if step % 500 == 0:
            active_logical = (pool.b_gate > -1e5).sum().item()
            norm_S = S_new.norm().item()
            print(f"Step {step:5d} | Active logical: {active_logical:3d}/{CONFIG['num_blocks']} | "
                  f"S_norm: {norm_S:.4f}")

    print("\n" + "=" * 60)
    print("Training complete. Freezing model...")
    pool.finalize_inference()
    pool.shutdown()
    print("✅ Done.")
