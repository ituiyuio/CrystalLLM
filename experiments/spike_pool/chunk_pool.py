"""
ChunkedSpikePool — chunk 级路由 + 双缓冲预取 + CPU pinned 池 (实验线 C)

背景 (2026-09-06 讨论): SpikeLLM 撞上存储墙 — 每 token 位置都要
gather + dequant + bmm + manual-grad + update_buffer RMW + commit 一整轮,
其中大部分流量是搬运 (零 FLOP), 且 cpu_offload 模式下 PCIe 往返全在关键路径.

本文件的三个改动 (尽量保持 forward_step_state 的数学语义):

  C1. chunk 级摊销: 一次 gather/dequant 服务 C 个位置 (shared 模式),
      或 C 个位置 union 一次更新 (exact 模式); manual-grad 从
      "每位置一次 [d_inner,d_model] RMW" 变成 "每 chunk 一次掩码 GEMM";
      commit 从每位置一次变每 chunk 一次; 干掉每位置 empty_cache.
  C2. token 门控 (gate_source='token'): chunk 的路由由 chunk 内 token
      embedding 均值决定 (而非 S_old), 因此所有 chunk 的路由在 forward
      开始前就已知 — 这是预取的前提. 基线路由本来就近似随机
      (b_gate 除块 0 全 -1e6, content bias 可忽略), 换门控源代价很小.
  C3. CPU pinned 池 + 预取线程: W_pool INT8 驻留 pinned RAM, 后台线程
      提前把后续 chunk 的活跃块经 copy stream 拉进 GPU 双缓冲 slot;
      commit 的 int8 增量也经后台线程异步写回 CPU 池. PCIe 离开计算关键路径.

模式说明:
  mode='exact'  : 每位置独立 _compute_gates (state 源, 与基线对齐),
                  chunk 只做 union 更新摊销. 仅支持 GPU 池.
                  C=1 时与基线 forward_step_state 数值一致 (对拍用).
  mode='shared' : 每 chunk 一次路由 (token 源), C 个位置共用同一组块.
                  路由粒度变粗, 是带宽-精度 trade 的激进端. 支持 GPU/CPU 池.

已知偏差 (实验接受):
  - CPU 池下 chunk c+2 可能读到 chunk c commit 前的权重 (stochastic commit
    fire 的元素占 ~0.1%, 幅度 ±1 int8 单位, 跨 chunk 边界的模糊性).
  - manual-grad 的求和顺序与基线不同 (基线 BF16 逐 sample 累加, 这里
    FP32 求和后转 BF16), C=1 对拍误差 ~1e-3 相对量级.
"""

import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF',
                      'max_split_size_mb:512,garbage_collection_threshold:0.3')

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import math
import time
import queue
import threading
import importlib.util
import torch

_spec = importlib.util.spec_from_file_location(
    "train_engine", os.path.join(os.path.dirname(__file__), "train_engine.py")
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["train_engine"] = _mod
_spec.loader.exec_module(_mod)
RTX5090SpikePool = _mod.RTX5090SpikePool


# ======================================================================
# C3: 预取/回写线程 (CPU pinned 池专用)
# ======================================================================
class ChunkPrefetcher:
    """单后台线程, 串行处理两类任务 (串行 => W_pool_cpu 无并发读写):
      ('pf', c, ids)    : gather_buf <- W_pool_cpu[ids] (CPU),
                          stage_i8[c%2] <- gather_buf (DMA, copy stream)
      ('commit', s, ids): 线程等 ev 后
                          W_pool_cpu[ids] = clamp(W_pool_cpu[ids] + staging[s])

    协议 (n_chunks 个 chunk, num_slots=2 双缓冲):
        pf.start(all_block_ids)           # 下发全部预取任务
        for c in chunks:
            W_i8, scale = pf.get(c)       # 阻塞到 copy 完成, 计算流 wait 事件
            ...用 slot 计算 chunk c...
            pf.notify_compute_done(c)     # 记录事件, 允许 slot c%2 被 c+2 复用
            ...apply_update + commit...
            pf.enqueue_commit(step_i8, idx_cpu)
    """

    def __init__(self, pool: "ChunkedSpikePool", K: int, num_slots: int = 2):
        self.pool = pool
        self.K = K
        self.num_slots = num_slots
        d_inner, d_model = pool.d_inner, pool.d_model

        self.stream = torch.cuda.Stream()

        # GPU 侧双缓冲 INT8 staging + scale
        self.stage_i8 = torch.zeros(num_slots, K, d_inner, d_model,
                                    dtype=torch.int8, device='cuda')
        self.stage_scale = torch.zeros(num_slots, K, device='cuda')
        # CPU 侧 pinned gather 中转: W_pool_cpu[ids] 高级索引的结果是非 pinned
        # 临时张量, 直接 non_blocking H2D 会退化成 pageable 慢速通道
        self.gather_buf = torch.zeros(K, d_inner, d_model, dtype=torch.int8,
                                      pin_memory=True)
        self.gather_scale = torch.zeros(K, pin_memory=True)

        self.copy_ev = [torch.cuda.Event() for _ in range(num_slots)]
        self.compute_ev = [torch.cuda.Event() for _ in range(num_slots)]

        # commit 回写: pinned staging 轮转 (2 个)
        self.commit_staging = torch.zeros(2, K, d_inner, d_model, dtype=torch.int8,
                                          pin_memory=True)
        self.commit_free = queue.Queue()
        self.commit_free.put(0)
        self.commit_free.put(1)

        self.task_q = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._pending_pf = 0

    # ---------- 主线程接口 ----------
    def start(self, all_block_ids_cpu: torch.Tensor):
        """下发全部 chunk 的预取任务, 返回每 chunk 的就绪事件列表.
        事件跟 route 走 (不按 slot 键存), 允许两条 route 短暂在途不串数据."""
        n = all_block_ids_cpu.shape[0]
        evs = []
        for c in range(n):
            ev = threading.Event()
            evs.append(ev)
            self.task_q.put(('pf', c, all_block_ids_cpu[c].clone(), ev))
        self._pending_pf += n
        return evs

    def get(self, chunk_id: int, ready_ev: threading.Event):
        """阻塞到该 chunk 的 H2D 已入队, 计算流 wait copy 事件.
        返回 (stage_i8_view [K,d,d], scale_gpu [K])."""
        ready_ev.wait()
        slot = chunk_id % self.num_slots
        cur = torch.cuda.current_stream()
        cur.wait_event(self.copy_ev[slot])
        return self.stage_i8[slot], self.stage_scale[slot]

    def notify_compute_done(self, chunk_id: int):
        """chunk_id 计算完成后调用 (在计算流上), 允许 slot 被复用."""
        self.compute_ev[chunk_id % self.num_slots].record()

    def enqueue_commit(self, step_i8_gpu: torch.Tensor, commit_idx_cpu: torch.Tensor):
        """把 GPU 上算好的 int8 提交增量异步写回 CPU W_pool.
        阻塞直到有 staging slot 空闲 (背压)."""
        slot = self.commit_free.get()
        ev_ready = torch.cuda.Event()
        ev_ready.record()                      # 计算流: step_i8 已就绪
        ev_done = torch.cuda.Event()
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(ev_ready)
            self.commit_staging[slot].copy_(step_i8_gpu, non_blocking=True)
            ev_done.record(self.stream)
        self.task_q.put(('commit', slot, commit_idx_cpu, ev_done))

    def join_pending(self):
        """等待全部预取+commit 任务被消费 (收尾用, 不等 GPU 执行完)."""
        while self._pending_pf > 0 or self.task_q.qsize() > 0:
            time.sleep(0.001)

    def shutdown(self):
        self._stop.set()
        self.task_q.put(('stop', None, None, None))

    # ---------- 后台线程 ----------
    def _worker(self):
        while True:
            task = self.task_q.get()
            kind = task[0]
            if kind == 'stop':
                return
            if kind == 'pf':
                _, chunk_id, block_ids, ready_ev = task
                slot = chunk_id % self.num_slots
                # 等 GPU 用完这个 slot 上一轮的数据
                # (从未录制的 CUDA event 上 synchronize 会立即返回)
                self.compute_ev[slot].synchronize()
                with torch.no_grad():
                    # CPU gather: pinned <- pinned, ~10ms/268MB, 释放 GIL
                    torch.index_select(self.pool.W_pool, 0, block_ids,
                                       out=self.gather_buf)
                    torch.index_select(self.pool.scale_pool, 0, block_ids,
                                       out=self.gather_scale)
                with torch.cuda.stream(self.stream):
                    self.stage_i8[slot].copy_(self.gather_buf, non_blocking=True)
                    self.stage_scale[slot].copy_(self.gather_scale,
                                                 non_blocking=True)
                    self.copy_ev[slot].record(self.stream)
                ready_ev.set()
                self._pending_pf -= 1
            elif kind == 'commit':
                _, slot, commit_idx, ev_done = task
                ev_done.synchronize()
                with torch.no_grad():
                    # int16 加法再 clamp (int8 直接加会回绕溢出)
                    w = self.pool.W_pool[commit_idx].int() \
                        + self.commit_staging[slot][:len(commit_idx)].int()
                    self.pool.W_pool[commit_idx] = torch.clamp(
                        w, -128, 127).to(torch.int8)
                self.commit_free.put(slot)


# ======================================================================
# C1+C2+C3: chunk 化池子
# ======================================================================
class ChunkedSpikePool(RTX5090SpikePool):
    """在 RTX5090SpikePool 上加 chunk 化前向/更新; 继承门控/LR/commit 机制.
    update_buffer 强制驻 GPU (更新路径全 GPU), W_pool 可驻 pinned CPU."""

    SUB = 8  # union 更新的子批块数 (bmm 输出 [8, d, d] bf16 = 268MB @ d=4096)

    def __init__(self, config: dict, pin_pool: bool = True):
        super().__init__(config)
        # === 证伪阶梯 E3: 标量 γ → 逐通道 γ (Mamba 式对角衰减) ===
        # 必须在 SpikeLLM 包装之前替换 (module 注册的是同一个 Parameter 对象)
        if config.get('iss_gamma_per_channel') and self.iss_gamma_raw is not None:
            with torch.no_grad():
                vec = self.iss_gamma_raw.detach().expand(self.d_model).clone()
            self.iss_gamma_raw = torch.nn.Parameter(vec)
            print(f"  [E3] iss_gamma_raw -> per-channel [{self.d_model}] "
                  f"(init gamma={float(torch.sigmoid(vec[0])):.3f} x {self.d_model})")
        # === 证伪阶梯 E4/E5: 输入驱动 ===
        # 病根: S_t = γS + g·norm(W@S) 无输入项, embed(x_t) 只进 MSE target,
        # 状态链是只由 x_0 启动的自治系统 -> 理论天花板 = unigram 附近
        # (E0: unigram 7.04 nats vs SpikeLLM 8.25; E1 直注外壳 300 步 PPL 32 佐证).
        # E4 'sum'   : Δ = W@S + embed(x) 共享一个 RMS_norm — 实测 W@S 元素
        #              尺度 ~ embed 的 130x, 输入方向被淹死 (VAL PPL 1797 vs E1 32)
        # E5 'adjunct': S_t = γS + g·RMS_norm(embed(x)) + g_w·RMS_norm(W@S)
        #              输入走 E1 已验证通道 (带 autograd, embed 全量训练),
        #              池子作为受控附加记忆 (g_w 可学习, init 0.05)
        _id = config.get('pool_input_drive', False)
        self.input_drive = True if _id is True else _id  # True→'sum' 兼容 E4
        if self.input_drive == 'adjunct':
            self.iss_gain_w = torch.nn.Parameter(
                torch.tensor(float(config.get('iss_gain_w_init', 0.05)),
                             device='cuda'))
            print(f"  [E5] pool adjunct: S_t = γS + g·RMS_norm(embed(x)) "
                  f"+ g_w·RMS_norm(W@S), g_w init="
                  f"{float(self.iss_gain_w.detach()):.3f} (learnable)")
        elif self.input_drive:
            print(f"  [E4] pool input-driven (sum): Δ_t = W@S + embed(x_t)")
        # === 证伪阶梯 E6: BPTT 真梯度训池子 ===
        # 假设: manual 单步梯度在递归系统里是错误学习规则 (忽略 S_old 依赖
        # 之前的 W 更新), 是池子项越强越糟的根因候选.
        # True: W_active 进 autograd 图 (shared dequant 后 clone 成 leaf),
        # 训练循环 backward 后调 apply_bptt_grads() 把真梯度经 INT8 commit 落盘.
        # 需要 input_drive='adjunct' (无输入项的递归 BPTT 也救不了 unigram 天花板)
        self.bptt_pool = bool(config.get('bptt_pool', False))
        self._bptt_leaves = []      # [(W_leaf, block_idx)], backward 后消费
        if self.bptt_pool:
            print(f"  [E6] BPTT pool: W_active 进 autograd, 真梯度走 INT8 commit")
            # 释放父类 legacy 缓存 (只有废弃的 forward_step 路径用, ~4.7GB @ d=4096)
            for _attr in ('_W_fp32_cache', '_S_fp32_cache', '_W_bf16_cache',
                          '_S_bf16_cache', '_W_active_shared'):
                if hasattr(self, _attr):
                    delattr(self, _attr)
            torch.cuda.empty_cache()
        self.is_cpu_pool = self.cpu_offload
        if self.is_cpu_pool:
            if pin_pool:
                try:
                    self.W_pool = self.W_pool.pin_memory()
                    self.scale_pool = self.scale_pool.pin_memory()
                    print("  [C3] W_pool + scale_pool pinned (DMA 通道)")
                except Exception as ex:
                    print(f"  [C3] pin_memory 失败 ({ex}), 退回 pageable (慢 ~2x)")
            # 更新路径全 GPU: update_buffer 从 pool_device 挪到 cuda
            self.update_buffer = self.update_buffer.to('cuda')
            print(f"  [C1] update_buffer -> GPU "
                  f"({self.update_buffer.nbytes/1e9:.2f}GB), commit 经预取线程异步写回")

        K = self.cfg['Top_K_Active']
        n_slots = 2 if self.is_cpu_pool else 1
        # 每槽 BF16 dequant 结果 (chunk 内 C 个位置共用, 零重复 dequant)
        self._chunk_bf16 = torch.zeros(n_slots, K, self.d_inner, self.d_model,
                                       dtype=torch.bfloat16, device='cuda')
        if not self.is_cpu_pool:
            self._chunk_i8 = torch.zeros(1, K, self.d_inner, self.d_model,
                                         dtype=torch.int8, device='cuda')
        self._prefetcher = None

    def attach_prefetcher(self):
        assert self.is_cpu_pool, "prefetcher 仅用于 CPU 池"
        K = self.cfg['Top_K_Active']
        self._prefetcher = ChunkPrefetcher(self, K)
        return self._prefetcher

    # ------------------------------------------------------------------
    # C2: token 门控 (整条序列的 chunk 路由一次算完)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def route_token_chunks(self, E: torch.Tensor, t0: int, t1: int, C: int):
        """E: [B, L, d_model] (可带 grad, 内部 detach). 返回 ([n,K] GPU, [n,K] CPU).
        门控源 = chunk 内 embedding 均值 (替代基线的 S_old 均值).
        注意: 一次性算完整条序列, b_gate EMA/burst 统计按 chunk 粒度近似."""
        n = math.ceil((t1 - t0) / C)
        means = []
        for c in range(n):
            a, b = t0 + c * C, min(t0 + (c + 1) * C, t1)
            means.append(E[:, a:b, :].detach().mean(dim=(0, 1)))
        S_ref = torch.stack(means)                       # [n, d_model]
        q = S_ref @ self.q_proj.T                        # [n, d_k]
        scores = (q @ self.k_embed.T) * self.cfg['Attn_Scale']   # [n, num_blocks]
        # F34 式平局噪声: 休眠块 -1e6 的 ULP 问题同款
        noise = torch.rand_like(self.b_gate) * 1e-3 * self.b_gate.abs().clamp(min=1.0)
        K = min(self.cfg['Top_K_Active'], self.num_blocks)
        idx = torch.topk(scores + self.b_gate.unsqueeze(0) + noise.unsqueeze(0),
                         K, dim=1).indices               # [n, K]
        # chunk 粒度统计: 等效基线每位置 b_gate EMA
        C_eff = (t1 - t0) / n
        self.b_gate *= self.cfg['Gate_EMA_Decay'] ** C_eff
        self.last_active[idx] = self._step_counter
        self.burst_counter[idx] += C_eff
        self.burst_counter.clamp_(min=0)
        self._step_counter += (t1 - t0)
        return idx, idx.cpu()

    # ------------------------------------------------------------------
    # C1: chunk 化前向 (含 chunk 末尾的 manual-grad 更新 + commit)
    # ------------------------------------------------------------------
    def forward_chunk(self, targets: torch.Tensor, S_old: torch.Tensor,
                      mode: str = 'shared', blocks_gpu=None, stage=None,
                      scale_gpu=None, slot=0,
                      ce_residual=False, target_ids=None, head_w=None):
        """
        targets: [C, B, d_model] (chunk 内各位置的 target embedding, 可带 grad)
        S_old:   [B, d_model]
        shared 模式: blocks_gpu [K] + (stage_i8_view 或 None) + scale_gpu
        ce_residual (证伪阶梯 E2): True 时 manual-grad 的残差用真实 CE 梯度
            dL/dS = (softmax(z) - onehot(y)) @ head_w (z = S_new @ head_w.T),
            替代 MSE 残差 (S_new - target). 需 target_ids [C, B] + head_w [V, d].
        返回 (S_final [B, d], S_traj [C, B, d])
        """
        C = targets.shape[0]
        B = S_old.shape[0]
        train_mode = torch.is_grad_enabled() and not self.lock_phase
        if mode == 'exact':
            assert not self.is_cpu_pool, "exact 模式仅支持 GPU 池 (无 staging 预取)"

        # ---- shared: 一次 dequant 服务 C 个位置 ----
        W_bf16 = scale_k = None
        if mode == 'shared':
            K = blocks_gpu.shape[0]
            if stage is not None:  # CPU 池 + 预取: 线程已把 INT8 放进 stage
                with torch.no_grad():
                    self._chunk_bf16[slot].copy_(stage)     # int8 -> bf16 就地
                W_bf16 = self._chunk_bf16[slot]
                scale_k = scale_gpu
            elif self.is_cpu_pool:  # CPU 池无预取 (eval/generate): 同步 gather
                with torch.no_grad():
                    bi = blocks_gpu.cpu()
                    self._chunk_bf16[0].copy_(self.W_pool[bi].to('cuda'))
                    scale_k = self.scale_pool[bi].to('cuda')
                W_bf16 = self._chunk_bf16[0]
            else:                  # GPU 池: 直接 gather + dequant
                with torch.no_grad():
                    self._chunk_i8[0].copy_(self.W_pool[blocks_gpu])
                    self._chunk_bf16[0].copy_(self._chunk_i8[0])
                W_bf16 = self._chunk_bf16[0]
                scale_k = self.scale_pool[blocks_gpu]

        # E6: W_leaf 每 chunk clone 一次 (shared 下 W 对 chunk 内所有位置相同;
        # 62 个 bmm 复用同一 leaf, autograd 自动累积梯度.
        # 不能放进位置循环 -- 每位置 clone 536MB x 62 = 33GB 直接 OOM)
        W_leaf = None
        if self.bptt_pool and mode == 'shared' and train_mode:
            W_leaf = W_bf16.clone().requires_grad_(True)
            self._bptt_leaves.append((W_leaf, blocks_gpu))

        S_traj = []
        R_chunk = torch.zeros(C, self.d_inner, dtype=torch.bfloat16, device='cuda')
        S0_chunk = torch.zeros(C, self.d_model, dtype=torch.bfloat16, device='cuda')
        pos_blocks = []          # exact: 每位置活跃 idx
        union_list = []

        gamma = torch.sigmoid(self.iss_gamma_raw) if self.use_iss else None

        for t in range(C):
            target_t = targets[t]                     # [B, d]
            if mode == 'exact':
                step = self._step_counter
                self._step_counter += 1
                S_gate = S_old.detach().mean(dim=0)
                active_idx = self._compute_gates(step, S=S_gate)
                K_t = len(active_idx)
                if K_t == 0:
                    S_traj.append(S_old)
                    continue
                with torch.no_grad():
                    W_t = self.W_pool[active_idx].to(torch.bfloat16)
                    scale_t = self.scale_pool[active_idx]
                pos_blocks.append(active_idx)
                union_list.append(active_idx)
            else:
                W_t, scale_t, K_t = W_bf16, scale_k, K

            # ---- bmm ----
            # bptt (E6): W_leaf 进 autograd, 真梯度 (S 链也带 grad -> 全 BPTT)
            # 其他: no_grad, W 更新走 chunk 末尾的 manual 机制
            S_for_bmm = S_old.to(torch.bfloat16).unsqueeze(0) \
                .expand(K_t, -1, -1, -1).permute(0, 2, 1, 3) \
                .reshape(K_t, self.d_model, B)
            if W_leaf is not None:
                deltas_bf16 = torch.bmm(W_leaf, S_for_bmm)   # [K, d_inner, B]
                # scale 乘 + K 求和留在 bf16 (fp32 中间张量 62 位置 x 134MB 会 OOM)
                delta_S = (deltas_bf16 * scale_t.bfloat16().view(-1, 1, 1)) \
                    .sum(dim=0).T.float()                    # [B, d_inner]
            else:
                with torch.no_grad():
                    deltas_bf16 = torch.bmm(W_t, S_for_bmm)      # [K, d_inner, B]
                    delta_S = (deltas_bf16.float() * scale_t.view(-1, 1, 1)) \
                        .sum(dim=0).T                            # [B, d_inner]

            # ---- 状态方程 ----
            # adjunct (E5): S_new = γS + g·RMS_norm(embed(x_t)) + g_w·RMS_norm(W@S)
            #   注入项带 autograd (embed 全量训练, 同 E1); W 项 detach 走 manual 更新
            # sum (E4): S_new = γS + g·RMS_norm(W@S + embed(x_t)) — W 项淹没输入
            # 无 (基线): S_new = γS + g·RMS_norm(W@S) — 无输入, unigram 天花板
            if self.input_drive == 'adjunct':
                inj = target_t.float()                       # 带 grad!
                rms_inj = inj.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-6
                rms_w = delta_S.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-6
                gamma = torch.sigmoid(self.iss_gamma_raw) if self.use_iss \
                    else torch.tensor(1.0, device='cuda')
                S_new = gamma * S_old \
                    + self.iss_gain * (inj / rms_inj) \
                    + self.iss_gain_w * (delta_S / rms_w)
                rms = rms_w                                  # W 更新的链式因子用
            elif self.input_drive:  # 'sum' / True
                delta_S = delta_S + target_t.float().detach()
                rms = delta_S.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-6
                delta_hat = delta_S / rms
                gamma = torch.sigmoid(self.iss_gamma_raw) if self.use_iss else None
                S_new = gamma * S_old + self.iss_gain * delta_hat \
                    if self.use_iss else S_old + delta_S
            elif self.use_iss:
                rms = delta_S.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-6
                delta_hat = delta_S / rms
                gamma = torch.sigmoid(self.iss_gamma_raw)
                S_new = gamma * S_old + self.iss_gain * delta_hat
            else:
                rms = None
                S_new = S_old + delta_S

            # ---- S 范数软截断 (F2 纯 GPU) ----
            s_norm = S_new.norm(dim=-1, keepdim=True)
            over = s_norm > self.cfg['S_Norm_Cap']
            scale_f = s_norm / self._s_norm_target
            S_new = torch.where(over, S_new / scale_f, S_new)

            # ---- manual-grad 原料 ----
            if train_mode:
                # W 项进状态的链式因子: adjunct 走 g_w/rms_w, 其他走 g/rms
                if self.input_drive == 'adjunct':
                    chain = self.iss_gain_w.detach() / rms
                elif self.use_iss:
                    chain = self.iss_gain.detach() / rms
                else:
                    chain = None
                with torch.no_grad():
                    if ce_residual and target_ids is not None and head_w is not None:
                        # E2: 真实 CE 梯度做残差 (对 W 的单步线性化,
                        # dL/dΔ ≈ chain·dL/dS, 与 MSE 路径同 GEMM 机制)
                        z = S_new.detach().float() @ head_w.t()        # [B, V]
                        p = torch.softmax(z, dim=-1)
                        b_idx = torch.arange(B, device=z.device)
                        p[b_idx, target_ids[t]] -= 1.0                 # dL/dz
                        residual = p @ head_w                          # dL/dS
                        if chain is not None:
                            residual = residual * chain
                    else:
                        residual = S_new.float() - target_t.float()    # [B, d_inner]
                        if chain is not None:
                            residual = residual * chain
                    # 基线: Σ_b residual[b] ⊗ S_old[0] — 同方向先求和再外积
                    R_chunk[t] = residual.sum(dim=0).to(torch.bfloat16)
                    S0_chunk[t] = S_old[0].to(torch.bfloat16)

            S_traj.append(S_new)
            S_old = S_new

        S_traj = torch.stack(S_traj)                  # [C, B, d]

        # ---- chunk 末尾: 一次更新 + 一次 commit (替代基线的每位置一轮) ----
        if train_mode:
            if self.bptt_pool:
                pass  # E6: 真梯度留在 leaf, 训练循环 backward 后 apply_bptt_grads()
            elif mode == 'shared':
                self._apply_update(blocks_gpu, R_chunk, S0_chunk)
                self._commit_updates(blocks_gpu)
            elif union_list:
                union_idx = torch.unique(torch.cat(union_list))
                K_u = len(union_idx)
                # 掩码矩阵 M [K_u, C]: 块 b 在位置 t 是否活跃
                union_pos = {int(b): i for i, b in enumerate(union_idx.tolist())}
                M = torch.zeros(K_u, C, device='cuda')
                for t, idx_t in enumerate(pos_blocks):
                    M[torch.tensor([union_pos[int(b)] for b in idx_t.tolist()],
                                   device='cuda'), t] = 1.0
                self._apply_update(union_idx, R_chunk, S0_chunk, mask=M)
                self._commit_updates(union_idx)

        return S_old, S_traj

    # ------------------------------------------------------------------
    # E6: backward 后消费 BPTT 真梯度 (经 INT8 commit 机制落盘)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def apply_bptt_grads(self):
        lr = self.cfg['lr_base']
        for W_leaf, idx in self._bptt_leaves:
            g = W_leaf.grad
            if g is None:
                continue
            self.update_buffer[idx] -= (lr * g.float()) \
                .to(self.update_buffer.dtype)
            self._commit_updates(idx)
        self._bptt_leaves.clear()

    # ------------------------------------------------------------------
    # C1: 掩码 GEMM 批量更新 (替代基线每位置 [d_inner,d_model] RMW)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _apply_update(self, idx, R_chunk, S0_chunk, mask=None):
        """
        acc[k] = Σ_t M[k,t] · R[t] ⊗ S0[t]   (mask=None 时 M 全 1)
        update_buffer[idx] -= lr·acc (ISS) 或 lr·acc/scale_k (非 ISS)
        子批 bmm, 每批 acc [SUB, d, d].
        """
        K_u = idx.shape[0]
        lr = self.cfg['lr_base']
        S0T = S0_chunk.T.contiguous()                    # [d_model, C]
        for kb in range(0, K_u, self.SUB):
            sub = idx[kb:kb + self.SUB]
            sb = sub.shape[0]
            if mask is not None:
                m = mask[kb:kb + sb].to(torch.bfloat16)          # [sb, C]
                Rm = R_chunk.T.unsqueeze(0) * m.unsqueeze(1)     # [sb, d_inner, C]
            else:
                Rm = R_chunk.T.unsqueeze(0).expand(sb, -1, -1)
            acc = torch.bmm(Rm, S0T.unsqueeze(0).expand(sb, -1, -1)
                            .transpose(1, 2))                    # [sb, d_inner, d_model]
            upd = lr * acc.float()
            if not self.use_iss:
                upd = upd / self.scale_pool[sub].view(-1, 1, 1).to('cuda')
            self.update_buffer[sub] -= upd.to(self.update_buffer.dtype)

    # ------------------------------------------------------------------
    # commit 覆写: fire.any() 短路 + CPU 池异步写回
    # (fire 全 False 时基线写 W+0=W, 跳过完全等价且省一次 RMW)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _commit_updates(self, idx):
        # int8 加法先回绕溢出 (125+3 -> -128), clamp 救不回 — 基线同款 bug,
        # 这里用 int16 加法再 clamp 修掉 (基线 train_engine 未修, 见报告)
        def _safe_commit(commit_idx, step_i8):
            w = self.W_pool[commit_idx].int() + step_i8.int()
            self.W_pool[commit_idx] = torch.clamp(w, -128, 127).to(torch.int8)

        if self.commit_mode == 'stochastic':
            buf = self.update_buffer[idx]
            fire = torch.rand_like(buf) < buf.abs().clamp(max=1.0)
            if not fire.any():
                return
            step = torch.where(fire, buf.sign().to(torch.int8),
                               torch.zeros((), dtype=torch.int8, device=buf.device))
            self.update_buffer[idx] -= step.to(self.update_buffer.dtype)
            if self.is_cpu_pool:
                if self._prefetcher is not None:
                    self._prefetcher.enqueue_commit(step, idx.cpu())
                else:
                    _safe_commit(idx.cpu(), step.cpu())
            else:
                _safe_commit(idx, step)
        else:
            update_norms = self.update_buffer[idx].float().view(len(idx), -1) \
                .abs().mean(dim=1)
            commit_mask = update_norms > 0.5
            if not commit_mask.any():
                return
            commit_idx = idx[commit_mask]
            rounded = torch.round(self.update_buffer[commit_idx].float()).to(torch.int8)
            self.update_buffer[commit_idx] -= rounded.float().to(self.update_buffer.dtype)
            if self.is_cpu_pool:
                if self._prefetcher is not None:
                    self._prefetcher.enqueue_commit(rounded, commit_idx.cpu())
                else:
                    _safe_commit(commit_idx.cpu(), rounded.cpu())
            else:
                _safe_commit(commit_idx, rounded)


# ======================================================================
# 工厂: 跟 spike_llm.make_pool 同款配置, 换 ChunkedSpikePool
# ======================================================================
def make_chunk_pool(d_model=1024, d_inner=1024, num_blocks=64, top_k=16,
                    cpu_offload=False, scale_coef=1.0, pin_pool=True,
                    ce_residual=False, gamma_per_channel=False,
                    bptt_pool=False, input_drive=False):
    cfg = dict(_mod.CONFIG)
    cfg['d_model'] = d_model
    cfg['d_inner'] = d_inner
    cfg['num_blocks'] = num_blocks
    cfg['Top_K_Active'] = top_k
    cfg['T_total'] = 200
    cfg['T_batch'] = 1
    cfg['gpu_mem_fraction'] = 1.0
    cfg['MicroSleep_Interval'] = 999999
    cfg['Disable_Lock'] = True          # F30: LM 路径不锁
    cfg['use_iss'] = True               # F36
    cfg['iss_gamma_init'] = 0.95
    cfg['iss_gain_init'] = 0.05
    cfg['commit_mode'] = 'stochastic'   # F37
    cfg['cpu_offload'] = cpu_offload
    # 证伪阶梯开关
    cfg['ce_residual'] = ce_residual                # E2
    cfg['iss_gamma_per_channel'] = gamma_per_channel  # E3
    cfg['bptt_pool'] = bptt_pool                    # E6
    cfg['pool_input_drive'] = input_drive           # E4 'sum'/True, E5 'adjunct'
    ref_d = 1024
    cfg['scale_pool_init'] = 0.01 * (ref_d / d_model) ** 0.5 * scale_coef  # F25/F29
    return ChunkedSpikePool(cfg, pin_pool=pin_pool)
