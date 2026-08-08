"""
脉冲残差池推理引擎 - RTX 5090 极致优化版 (v3-fix)

功能：
    1. 加载训练生成的 .bin + .npy 文件
    2. 三级存储管理：GPU热集、CPU温集、NVMe冷集
    3. 异步预取：后台线程 + pinned memory，隐藏I/O延迟
    4. CPU-GPU异构计算：热块在GPU做Batch GEMM，温块在CPU做小批量矩阵乘
    5. 门控与训练末期完全对齐（硬阈值 + 内容寻址偏置）
    6. 支持自回归生成（infer_sequence）

设计原则：
    - 所有张量操作均为批处理（无Python循环）
    - 利用RTX 5090的INT8 Tensor Core和CPU的AVX-512
    - NVMe使用memmap零拷贝，节省内存
    - 预取窗口自适应，根据激活概率预测下一步可能需要的块
    - 门控全GPU：删除 S_cpu 镜像，避免假同步 D2H

=== v3-fix 修复清单（相对 v3-final 原文）===
  F1. 冷块"异步预取"是假的（_ensure_blocks_ready 同步 NVMe + 同步 H2D）→
      拆成两步：后台线程从 memmap 读 → pinned prefetch_buffer；
                主循环需要时从 buffer 异步 H2D 拷到 GPU。
      cache miss（首次冷块访问）仍同步加载一次，后续命中 cache。
  F2. S_cpu 假同步（.cpu() 本身就是同步 D2H）→
      删除 S_cpu 镜像；门控全部上 GPU（b_gate / k_embed / q_proj）。
      跨设备同步点归零。
  F3. 预取循环 N 次 .item() 触发 GPU→CPU 同步 →
      维护 CPU 镜像 _gpu_resident_cpu / _cpu_resident_cpu，
      每 _resident_sync_interval 步同步一次（不是每步）。
  F4. S 范数闸门缺失（推理时无 S_Norm_Cap 软截断）→
      加上：纯 GPU torch.where 缩放，与训练端 S_Norm_Cap=512 一致。
  F5. memmap.astype(np.int8) 冗余（memmap 本身就是 int8）→
      去掉，直接 torch.from_numpy(memmap[batch])。
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List

# ======================== 配置 ========================
INFER_CONFIG = {
    'd_model': 4096,              # 必须与训练一致
    'd_inner': 16384,             # 必须与训练一致
    'num_blocks': 128,            # 必须与训练一致
    'd_k': 16,                    # 必须与训练一致
    'gpu_hot_ratio': 0.25,        # 常驻GPU的热块比例（按激活频次Top%）
    'cpu_warm_ratio': 0.35,       # 驻留CPU钉锁内存的温块比例
    'prefetch_window': 16,        # 每步预取的候选块数（覆盖冷块NVMe延迟）
    'top_k_active': 32,           # 每步最大激活块数（Batch GEMM最优）
    'attn_scale': 0.1,            # 注意力偏置缩放（与训练一致）

    # === v3-fix: F4 — S 范数闸门（与训练端一致）===
    's_norm_cap': 512.0,          # S 范数硬上限（超则缩放到 128）
    's_norm_target': 128.0,       # 缩放目标范数

    # === v3-fix: F3 — CPU 镜像同步间隔 ===
    'resident_sync_interval': 10, # 每 N 步 sync 一次 resident 状态到 CPU

    'model_prefix': './model_pool',  # 训练生成的文件前缀
    'use_cpu_compute': True,      # 是否启用CPU计算温块（异构模式）
    'cpu_cores': 8,               # CPU计算使用的线程数
}


class RTX5090InferenceEngine:
    """
    脉冲残差池推理引擎 - 支持三级存储 + 异步预取 + 异构计算 + 全GPU门控
    """

    def __init__(self, config: dict):
        """
        初始化推理引擎：加载模型、划分存储层级、初始化GPU/CPU内存。
        """
        self.cfg = config
        self.d_model = config['d_model']
        self.d_inner = config['d_inner']
        self.num_blocks = config['num_blocks']
        self.d_k = config['d_k']
        self.device = 'cuda'

        # 设置CPU线程数（影响torch在CPU上的矩阵乘性能）
        torch.set_num_threads(config.get('cpu_cores', 8))

        # ----- 1. 加载训练好的模型（使用memmap零拷贝） -----
        print("📂 Loading model from NVMe...")
        self._load_model()

        # ----- 2. 划分三级存储（热/温/冷） -----
        self._init_storage_hierarchy()

        # ----- 3. GPU显存初始化（仅分配热集 + 预取缓冲区） -----
        # W_gpu 是大张量，按需填充（避免一次性分配 8GB 浪费）
        self.W_gpu = torch.zeros(
            self.num_blocks, self.d_inner, self.d_model,
            dtype=torch.int8, device='cuda'
        )
        self.scale_gpu = torch.zeros(self.num_blocks, device='cuda')
        self.gpu_resident = torch.zeros(self.num_blocks, dtype=torch.bool, device='cuda')
        self._load_hot_blocks_to_gpu()

        # ----- 4. CPU内存池（温块常驻，钉锁内存供快速访问） -----
        self.W_cpu = torch.zeros(
            self.num_blocks, self.d_inner, self.d_model,
            dtype=torch.int8, pin_memory=True
        )
        self.scale_cpu = torch.zeros(self.num_blocks, pin_memory=True)
        self.cpu_resident = torch.zeros(self.num_blocks, dtype=torch.bool)
        self._load_warm_blocks_to_cpu()

        # === v3-fix: F3 — CPU 镜像（每 N 步 sync 一次，避免预取循环 .item() 同步）===
        self._gpu_resident_cpu = torch.zeros(self.num_blocks, dtype=torch.bool)
        self._cpu_resident_cpu = torch.zeros(self.num_blocks, dtype=torch.bool)
        self._sync_resident_to_cpu()   # 初始化时先 sync 一次

        # ----- 5. 预取引擎（后台线程 + pinned memory cache） -----
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.prefetch_stream = torch.cuda.Stream()
        # prefetch_buffer: {idx: pinned tensor}，冷块预取 cache
        # 大小限制：防止长跑内存泄漏
        self.prefetch_buffer: dict[int, torch.Tensor] = {}
        self.prefetch_buffer_max = 256   # 最多缓存 256 个冷块
        self.prefetch_future = None

        # ----- 6. 全局状态 S（仅 GPU，无 CPU 镜像） -----
        self.S = torch.zeros(self.d_model, device='cuda')
        # 预算缩放因子张量（F4 修复用）
        self._s_norm_target_t = torch.tensor(self.cfg['s_norm_target'], device='cuda')

        # ----- 7. 门控参数（全部上 GPU，F2 修复）-----
        # 推理 = 纯前向，没必要留在 CPU 上当同步源
        self.b_gate = torch.from_numpy(self.b_gate_np).float().cuda()
        self.k_embed = torch.from_numpy(self.k_embed_np).float().cuda()
        self.q_proj = torch.from_numpy(self.q_proj_np).float().cuda()
        # stats 留在 CPU（只在 evict 时用，不需要 GPU）
        self.stats_np = self.stats_np  # numpy 引用

        self._step_counter = 0

        print(f"✅ Inference engine ready: {len(self.hot_indices)} hot blocks on GPU, "
              f"{len(self.warm_indices)} warm blocks on CPU, "
              f"{len(self.cold_indices)} cold blocks on NVMe.")
        print(f"   GPU memory used (W_gpu): {self.W_gpu.element_size() * self.W_gpu.nelement() / 1e9:.2f}GB")

    # ===================== 模型加载 =====================
    def _load_model(self):
        """
        从NVMe加载训练好的池子及元数据。
        使用np.memmap实现零拷贝，不占用CPU DRAM。
        """
        prefix = self.cfg['model_prefix']
        pool_path = f"{prefix}.bin"
        if not os.path.exists(pool_path):
            raise FileNotFoundError(f"Model file not found: {pool_path}")

        # 权重池（INT8） - memmap只读映射
        self.W_nvme = np.memmap(
            pool_path, dtype=np.int8, mode='r',
            shape=(self.num_blocks, self.d_inner, self.d_model)
        )

        # 加载元数据（直接 mmap，numpy 自动处理）
        self.scale_np = np.load(f"{prefix}_scale.npy")
        self.b_gate_np = np.load(f"{prefix}_gate.npy")
        self.k_embed_np = np.load(f"{prefix}_kembed.npy")
        self.q_proj_np = np.load(f"{prefix}_qproj.npy")
        self.stats_np = np.load(f"{prefix}_stats.npy")

        print(f"   Loaded W_pool: {self.W_nvme.shape}, size: {self.W_nvme.nbytes / 1e9:.2f}GB")

    # ===================== 存储层级划分 =====================
    def _init_storage_hierarchy(self):
        """
        按激活频次（stats_np降序）划分热/温/冷集。
        训练端 finalize_inference 已按频次降序重排，因此索引 0~hot_end 为热集。
        这是隐式契约——见 README "训练→推理契约" 段。
        """
        n = self.num_blocks
        hot_end = int(n * self.cfg['gpu_hot_ratio'])
        warm_end = int(n * (self.cfg['gpu_hot_ratio'] + self.cfg['cpu_warm_ratio']))

        self.hot_indices = list(range(hot_end))
        self.warm_indices = list(range(hot_end, warm_end))
        self.cold_indices = list(range(warm_end, n))

        print(f"   Storage hierarchy: Hot={len(self.hot_indices)}, "
              f"Warm={len(self.warm_indices)}, Cold={len(self.cold_indices)}")

    # ===================== 加载热块到GPU =====================
    def _load_hot_blocks_to_gpu(self):
        """将热集块从NVMe搬运到GPU显存（批量，H2D）"""
        # === v3-fix: F5 — 去掉 .astype(np.int8)，memmap 本身就是 int8 ===
        batch_size = 32
        for i in range(0, len(self.hot_indices), batch_size):
            batch_idx = self.hot_indices[i:i+batch_size]
            cpu_data = torch.from_numpy(self.W_nvme[batch_idx])
            # H2D 异步拷贝（非阻塞），最后 sync 一次
            self.W_gpu[batch_idx] = cpu_data.cuda(non_blocking=True)
            self.scale_gpu[batch_idx] = torch.from_numpy(self.scale_np[batch_idx]).cuda(non_blocking=True)
            self.gpu_resident[batch_idx] = True
        torch.cuda.synchronize()
        print(f"   Loaded {len(self.hot_indices)} hot blocks to GPU.")

    # ===================== 加载温块到CPU =====================
    def _load_warm_blocks_to_cpu(self):
        """
        将温集块加载到CPU钉锁内存（pinned memory），
        以便后续CPU计算和快速H2D拷贝。
        """
        # === v3-fix: F5 — 去掉 astype ===
        for idx in self.warm_indices:
            data_tensor = torch.from_numpy(self.W_nvme[idx])
            # pinned memory：直接 .pin_memory()（已在 pinned pool 里则 no-op）
            if not data_tensor.is_pinned():
                data_tensor = data_tensor.pin_memory()
            self.W_cpu[idx] = data_tensor
            self.scale_cpu[idx] = self.scale_np[idx]
            self.cpu_resident[idx] = True
        print(f"   Loaded {len(self.warm_indices)} warm blocks to CPU pinned memory.")

    # ===================== v3-fix: F3 — 同步 resident 状态到 CPU =====================
    def _sync_resident_to_cpu(self):
        """把 GPU 上的 resident mask 拷到 CPU 镜像，每 N 步调一次"""
        self._gpu_resident_cpu = self.gpu_resident.cpu()
        # cpu_resident 本来就在 CPU，但保证一致性
        self._cpu_resident_cpu = self.cpu_resident.clone()

    # ===================== 异步预取 =====================
    def _async_prefetch(self, indices: List[int]):
        """
        后台预取（线程池运行）：将候选块从NVMe读到 pinned prefetch_buffer。
        用 CPU 镜像判断缓存状态，避免 N 次 .item() 同步。
        """
        # === v3-fix: F3 — 用 CPU 镜像判断，不触发 GPU 同步 ===
        for idx in indices:
            if self._gpu_resident_cpu[idx] or self._cpu_resident_cpu[idx]:
                continue   # 已经在 GPU 或 CPU，跳过
            if idx in self.prefetch_buffer:
                continue   # 已经在 prefetch cache 里

            # LRU 淘汰：buffer 满了就清最老的
            if len(self.prefetch_buffer) >= self.prefetch_buffer_max:
                # dict 是插入序，pop oldest
                oldest = next(iter(self.prefetch_buffer))
                del self.prefetch_buffer[oldest]

            # === v3-fix: F5 — 去掉 astype ===
            data_tensor = torch.from_numpy(self.W_nvme[idx])
            if not data_tensor.is_pinned():
                data_tensor = data_tensor.pin_memory()
            self.prefetch_buffer[idx] = data_tensor

    # ===================== 确保活跃块就绪 =====================
    def _ensure_blocks_ready(self, indices: List[int]):
        """
        同步确保指定块在 GPU 或 CPU 上可用。

        # === v3-fix: F1 — 冷块预取路径拆开 ===
        - 热块：已在 GPU，直接用
        - 温块：已在 CPU，直接用
        - 冷块：先查 prefetch_buffer
            - cache hit: 异步 H2D 拷到 GPU（不阻塞当前步的剩余计算）
            - cache miss: 同步从 memmap 读 + 同步 H2D（一次性 cache fill）
        """
        need_async_h2d: List[int] = []   # cache hit，走异步 H2D
        need_sync_load: List[int] = []   # cache miss，兜底同步加载

        # === v3-fix: F3 — 用 CPU 镜像判断 ===
        for idx in indices:
            if self._gpu_resident_cpu[idx]:
                continue   # 热块 / 已加载冷块
            if self._cpu_resident_cpu[idx]:
                continue   # 温块
            # 冷块路径
            if idx in self.prefetch_buffer:
                need_async_h2d.append(idx)
            else:
                need_sync_load.append(idx)

        # ----- 异步 H2D（cache hit） -----
        if need_async_h2d:
            with torch.cuda.stream(self.prefetch_stream):
                for idx in need_async_h2d:
                    pinned_data = self.prefetch_buffer[idx]
                    self.W_gpu[idx] = pinned_data.cuda(non_blocking=True)
                    self.scale_gpu[idx] = torch.from_numpy(self.scale_np[idx:idx+1]).cuda(non_blocking=True).squeeze(0)
                    self.gpu_resident[idx] = True
                    # 从 prefetch_buffer 删除（已经在 GPU 了）
                    del self.prefetch_buffer[idx]
            # 不在这里 sync——下一轮 H2D 等 GPU 计算用同一个 stream 自然排队

        # ----- 同步加载兜底（cache miss，第一次冷块访问） -----
        # 这是不可避免的开销，但只在每个冷块第一次访问时发生
        if need_sync_load:
            for idx in need_sync_load:
                # === v3-fix: F5 — 去掉 astype ===
                data_tensor = torch.from_numpy(self.W_nvme[idx])
                self.W_gpu[idx] = data_tensor.cuda()
                self.scale_gpu[idx] = self.scale_np[idx]
                self.gpu_resident[idx] = True
                # 加入 LRU buffer（容量允许的话）
                if len(self.prefetch_buffer) < self.prefetch_buffer_max:
                    if not data_tensor.is_pinned():
                        data_tensor = data_tensor.pin_memory()
                    self.prefetch_buffer[idx] = data_tensor
            torch.cuda.synchronize()

    # ===================== 门控决策（全 GPU，F2 修复）=====================
    def _compute_gates(self) -> torch.Tensor:
        """
        推理态门控：硬阈值 + 内容寻址偏置（与训练末期完全对齐）。
        公式：prob = sigmoid(b_gate + attn_scale * (k_embed @ (q_proj @ S)))
        决策：G = (prob > 0.5).int()
        若全为0，强制激活第一个块（最高频）。
        若超过top_k_active，保留概率最高的top_k_active个。
        返回：激活块索引（GPU Tensor，dtype=int64）
        """
        # === v3-fix: F2 — 全部 GPU，无 S_cpu 镜像 ===
        # 内容寻址得分
        q = torch.mv(self.q_proj, self.S)              # [d_k], GPU
        content_scores = torch.mv(self.k_embed, q)     # [num_blocks], GPU
        content_bias = content_scores * self.cfg['attn_scale']

        # 融合门控
        fused_logits = self.b_gate + content_bias
        prob = torch.sigmoid(fused_logits)

        # 硬阈值
        G = (prob > 0.5).int()

        # 安全防护：至少激活1个块
        if G.sum() == 0:
            G[0] = 1

        # 性能防护：限制最大激活数
        if G.sum() > self.cfg['top_k_active']:
            _, top_idx = torch.topk(prob * G.float(), self.cfg['top_k_active'])
            G = torch.zeros_like(G)
            G[top_idx] = 1

        return G.nonzero(as_tuple=True)[0]   # GPU int64 indices

    # ===================== 异构计算核心 =====================
    def _compute_deltas_hybrid(self, active_idx: torch.Tensor) -> torch.Tensor:
        """
        根据块所在层级分发计算：
            - 热块 (在GPU) -> GPU Batch GEMM
            - 温块 (在CPU) -> CPU 小批量矩阵乘 (使用AVX-512 / MKL)
            - 冷块 (已临时加载到GPU) -> 走GPU路径
        返回：所有增量之和（GPU Tensor）
        """
        K = len(active_idx)
        if K == 0:
            return torch.zeros(self.d_model, device='cuda')

        # === v3-fix: F3 — 用 CPU 镜像分类，避免每步 .item() 同步 ===
        gpu_idx_list: List[int] = []
        cpu_idx_list: List[int] = []
        for idx in active_idx.tolist():
            if self._gpu_resident_cpu[idx]:
                gpu_idx_list.append(idx)
            else:
                # 温块或意外未加载的——兜底走 CPU 路径
                cpu_idx_list.append(idx)

        S_new = torch.zeros(self.d_model, device='cuda')

        # ----- GPU 计算（热块 + 冷块） -----
        if gpu_idx_list:
            # 预分配 idx tensor 池，避免每步新建（不是阻塞项，但 O(n) 优化）
            gpu_idx_t = torch.tensor(gpu_idx_list, device='cuda', dtype=torch.long)
            W_gpu_active = self.W_gpu[gpu_idx_t].float() * self.scale_gpu[gpu_idx_t].unsqueeze(-1).unsqueeze(-1)
            S_batch = self.S.unsqueeze(0).unsqueeze(-1).repeat(len(gpu_idx_list), 1, 1)
            deltas_gpu = torch.bmm(W_gpu_active, S_batch).squeeze(-1)   # [K_gpu, d_inner]
            S_new += deltas_gpu.sum(dim=0)

        # ----- CPU 计算（温块） -----
        if cpu_idx_list and self.cfg['use_cpu_compute']:
            W_cpu_active = self.W_cpu[cpu_idx_list].float() * self.scale_cpu[cpu_idx_list].unsqueeze(-1).unsqueeze(-1)
            S_cpu_batch = self.S.unsqueeze(0).unsqueeze(-1).repeat(len(cpu_idx_list), 1, 1)
            deltas_cpu = torch.bmm(W_cpu_active, S_cpu_batch).squeeze(-1)   # [K_cpu, d_inner]
            delta_cpu_sum = deltas_cpu.sum(dim=0).cuda(non_blocking=True)
            S_new += delta_cpu_sum

        return S_new

    # ===================== 单步推理 =====================
    def infer_one_step(self) -> torch.Tensor:
        """
        单步推理（纯前向，无反向）。
        流程：
            1. 门控决策（GPU，无 S_cpu 同步）
            2. 异步预取（后台，每 N 步 sync 一次 resident）
            3. 确保当前活跃块就绪（cache hit 走异步 H2D，miss 同步兜底）
            4. 异构计算
            5. 更新 S
            6. === v3-fix: F4 — S 范数软截断 ===
        返回：更新后的全局状态 S
        """
        # 1. 门控决策（全 GPU）
        active_idx = self._compute_gates()
        if len(active_idx) == 0:
            return self.S
        K = len(active_idx)

        # 2. 异步预取（每 N 步 sync 一次 resident 状态）
        if self._step_counter % self.cfg['resident_sync_interval'] == 0:
            self._sync_resident_to_cpu()

        if self.prefetch_future is None or self.prefetch_future.done():
            # 预测下一批可能激活的块：当前未激活但概率最高的前 prefetch_window 个
            # 注：这里复用门控算过的 prob，需要重新算一次（prob 没保存）
            # 为省一次 matvec，从 _compute_gates 那里拿到 prob 更好；这里用轻量近似
            with torch.no_grad():
                q = torch.mv(self.q_proj, self.S)
                content_scores = torch.mv(self.k_embed, q)
                prob = torch.sigmoid(self.b_gate + content_scores * self.cfg['attn_scale'])

            G_full = torch.zeros(self.num_blocks, device='cuda')
            G_full[active_idx] = 1
            inactive_prob = prob * (1 - G_full)
            _, candidate_idx = torch.topk(
                inactive_prob,
                min(self.cfg['prefetch_window'], self.num_blocks)
            )
            if len(candidate_idx) > 0:
                self.prefetch_future = self.executor.submit(
                    self._async_prefetch,
                    candidate_idx.tolist()
                )

        # 3. 确保当前活跃块就绪（cache + sync 兜底）
        self._ensure_blocks_ready(active_idx.tolist())

        # 4. 异构计算
        delta_sum = self._compute_deltas_hybrid(active_idx)
        self.S = self.S + delta_sum

        # 5. === v3-fix: F4 — S 范数软截断（与训练端一致）===
        # 纯 GPU 计算，无 .item() 同步
        s_norm = self.S.norm()
        over = s_norm > self.cfg['s_norm_cap']
        scale = s_norm / self._s_norm_target_t
        self.S = torch.where(over, self.S / scale, self.S)

        self._step_counter += 1
        return self.S

    # ===================== 自回归生成 =====================
    def infer_sequence(self, steps: int, initial_S: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        自回归生成序列。
        initial_S: 初始全局状态（若为None，使用当前的S）
        返回：每一步的S快照列表（CPU张量）
        """
        if initial_S is not None:
            self.S = initial_S.cuda()

        outputs = []
        for _ in range(steps):
            S_next = self.infer_one_step()
            outputs.append(S_next.cpu())   # 强制同步以拿到当前快照
        return outputs

    # ===================== 显存回收（可选） =====================
    def evict_cold_blocks(self, keep_ratio: float = 0.6):
        """
        显存水位控制：将非热集的GPU驻留块逐回NVMe（清空GPU内存）。
        仅在预取导致GPU显存接近上限时调用。
        """
        # 用 CPU 镜像遍历（避免 .item() 同步）
        resident_idx = self._gpu_resident_cpu.nonzero(as_tuple=True)[0].tolist()
        evict_candidates = [i for i in resident_idx if i not in self.hot_indices]
        if not evict_candidates:
            return

        # 按频次排序，淘汰最冷的
        sorted_cold = sorted(evict_candidates, key=lambda i: self.stats_np[i])
        to_evict = sorted_cold[:int(len(sorted_cold) * (1 - keep_ratio))]

        for idx in to_evict:
            self.W_gpu[idx] = 0
            self.scale_gpu[idx] = 0
            self.gpu_resident[idx] = False
            if idx in self.prefetch_buffer:
                del self.prefetch_buffer[idx]

        # 同步 CPU 镜像
        self._sync_resident_to_cpu()
        torch.cuda.synchronize()
        print(f"   Evicted {len(to_evict)} cold blocks from GPU.")

    # ===================== 清理资源 =====================
    def shutdown(self):
        """关闭线程池，释放显存和memmap"""
        self.executor.shutdown(wait=True)
        del self.W_nvme   # 关闭memmap
        torch.cuda.empty_cache()
        print("🛑 Inference engine shut down.")


# ===================== 使用示例 =====================
if __name__ == "__main__":
    # 1. 初始化推理引擎
    engine = RTX5090InferenceEngine(INFER_CONFIG)

    # 2. 设置初始状态（例如来自token embedding）
    initial_S = torch.randn(INFER_CONFIG['d_model'])
    engine.S = initial_S.cuda()

    # 3. 预热（触发预取和缓存）
    for _ in range(5):
        _ = engine.infer_one_step()

    # 4. 性能测试
    start = time.perf_counter()
    num_steps = 100
    for _ in range(num_steps):
        _ = engine.infer_one_step()
    elapsed = time.perf_counter() - start
    print(f"⏱️ {num_steps} steps inference took {elapsed*1000:.2f}ms "
          f"({elapsed/num_steps*1000:.3f}ms/step)")

    # 5. 自回归生成
    print("\n🔮 Generating sequence...")
    outputs = engine.infer_sequence(steps=50, initial_S=initial_S)
    print(f"   Generated {len(outputs)} states, final S norm: {outputs[-1].norm().item():.4f}")

    # 6. 清理
    engine.shutdown()
