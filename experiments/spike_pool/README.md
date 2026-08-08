# spike_pool — INT8 脉冲残差池训练引擎 (v3-fix)

**分支**：`spike-pool-v3`（基于 `main`）
**承接**：来自一次深度对话推演的设计（v3-final 代码 + 5 个 v3-fix 修复）
**目标**：把 Transformer 的 FFN 拆成数百个 INT8 残差块，用轻量内容寻址 + 宏观调度让 Tensor Core 一直吃饱

## 背景

主线（v49/v50）在做 CWF（capability-weighted forward）+ Soft-Exp 那一套梯度-free 训练。spike_pool 走的是另一条路——

> **把 FFN 拆掉，做成 INT8 残差块池 + 动态门控**。

不是替代品，是另一条探索线。所以独立分支、独立目录。

## 核心设计

| # | 模块 | 关键想法 |
|---|------|---------|
| 1 | 3D INT8 权重池 | `[num_blocks, d_inner, d_model]` 物理连续，L2 缓存友好 |
| 2 | FP32 累加器 | 梯度不直接量化，累超过 0.5 步长才四舍五入到 int8 |
| 3 | 局部反向 | 只对当前激活的 K 个块算梯度（不激活不反传） |
| 4 | 动态学习率 | 老块余弦退火，新块指数升温（**新块永远不超越老块**） |
| 5 | 微睡眠 | 每 100 步唤醒冻结块做低 lr 离线巩固 |
| 6 | 有形大手 | 磨损均衡 + 僵尸清退 + GDP 同质化税收（\|cos_sim\| 双向惩罚） |
| 7 | 末期锁定 | 最后 5% 步数硬阈值门控，**与推理完全对齐**（多数 MoE 脚本不写这段） |
| 8 | S 范数闸门 | 全局状态范数软截断到 128，跟 INT8 量化区间强绑定 |

## v3-fix 修复清单

相对原始 v3-final 代码，5 个会让代码跑不起来或跑歪的硬伤：

| ID | 问题 | 严重度 | 修法 |
|----|------|-------|------|
| F1 | `autograd.grad` 收到无 `requires_grad` 的 `W_active`，**直接崩** | 🔴 阻塞 | `W_active = (... ).detach().requires_grad_(True)` |
| F2 | S 范数截断每步 `.item()` 触发 CPU 同步，**T_total=10k 步就是 10k 次 sync** | 🟡 性能 | 纯 GPU：`torch.where(over, self.S / scale, self.S)` |
| F3 | k_embed 预热用 Python for 循环，**跟"全向量化"承诺冲突** | 🟡 设计 | base + noise_matrix 一次性赋值 |
| F4 | b_gate 只有上升（磨损均衡 +0.5）+ 事件触发下降（GDP 税），**没有对称机制** | 🟠 漂移 | 加 EMA 衰减 `b_gate *= 0.999`（每步） |
| F5 | 微睡眠 scale 归一化改 W_pool，**128M 参数被反复量化** | 🟠 精度 | 只动 `scale_pool`（scale 本来就是干这用的） |

修复点在代码里全部用 `# === v3-fix: <id> ===` 标注，diff 友好。

## 快速开始

```bash
# 已经在 spike-pool-v3 分支上
uv sync                                  # 依赖到位（pyproject.toml 已有 torch==2.9.1 + cu128）
python experiments/spike_pool/train_engine.py
```

⚠️ **注意**：当前 `CONFIG` 默认是 demo 规模（`d_model=4096, num_blocks=128, T_total=10000`），
INT8 池占显存 ~8GB，**需要 ≥12GB VRAM**（RTX 5090 32GB 满载，但笔记本卡会爆）。

跑起来后会看到：

```
🚀 RTX5090SpikePool initialized: 128 blocks, d_model=4096, d_inner=16384
   Total INT8 parameters: 8.59B
🌱 k_embed initialized with S_avg at step 100
🔒 Step 9500: Entering inference alignment phase.
...
✅ Model frozen and saved to ./model_pool.bin
```

## 依赖

- Python ≥ 3.10
- PyTorch ≥ 2.9.1（CUDA 12.8，已在 `pyproject.toml` 里锁了 cu128 索引）
- 单卡 CUDA（设计目标是 RTX 5090 / H100 这种 ~32GB 显存）

不需要 `transformers` / `datasets`——纯 PyTorch 原生 + `numpy`。

## 配置项

`CONFIG` 字典在 `train_engine.py` 顶部，关键旋钮：

| Key | 默认 | 说明 |
|-----|------|------|
| `d_model` | 4096 | 全局状态维度（必须 16 的倍数） |
| `d_inner` | 16384 | FFN 内部维度（一般 4× d_model） |
| `num_blocks` | 128 | 残差块总数 |
| `Top_K_Active` | 32 | 每步最大激活块数（决定 Batch GEMM 形状） |
| `T_total` | 10000 | 总训练步数 |
| `Lock_Ratio` | 0.95 | 末期锁定起始比例（最后 5% 固化） |
| `S_Norm_Cap` | 512 | S 范数硬上限（超则缩放到 128） |
| `Gate_EMA_Decay` | 0.999 | b_gate 每步衰减系数（v3-fix F4） |

## 跟主线的关系

- **不在 main 上**：独立分支 `spike-pool-v3`，可以独立 review
- **不污染 `experiments/v49_pre/`**：v49_pre 是 CWF/Soft-Exp 那一套，spike_pool 是另一条路
- **命名**：spike_pool 而非 v51_spike_pool——v 编号留个余量给主线（v50 已经在 main 上）

---

## 推理引擎（infer_engine.py）

训练端的姊妹项目——把训练生成的 `model_pool.bin` + 元数据加载起来跑前向。

### 核心设计

| # | 模块 | 关键想法 |
|---|------|---------|
| 1 | 三级存储 | Hot (GPU) / Warm (CPU pinned) / Cold (NVMe memmap) 按激活频次切分 |
| 2 | 异步预取 | 后台线程从 memmap 读 → pinned `prefetch_buffer`；主循环 cache hit 走异步 H2D |
| 3 | 异构计算 | 热块在 GPU 做 Batch GEMM（INT8 Tensor Core），温块在 CPU 做小批量矩阵乘（AVX-512 / MKL） |
| 4 | 全 GPU 门控 | 删 S_cpu 镜像，b_gate / k_embed / q_proj 全在 GPU，跨设备同步点归零 |
| 5 | 末期对齐 | 硬阈值（`prob > 0.5`）+ 内容寻址偏置，与训练末期 `lock_phase` 完全一致 |
| 6 | 缓存策略 | prefetch_buffer LRU 上限 256 块，防长跑内存泄漏；冷块首次访问同步兜底 |

### 推理端 v3-fix 修复清单

相对原始 v3-final 推理代码的 5 个硬伤：

| ID | 问题 | 严重度 | 修法 |
|----|------|-------|------|
| F1 | `_ensure_blocks_ready` 同步 NVMe 读 + 同步 H2D，**首次冷块访问卡主循环 50-100ms** | 🔴 阻塞 | 拆成"后台 memmap→pinned"和"主循环 pinned→GPU 异步 H2D"两步；cache miss 仍同步兜底（一次性） |
| F2 | `S_cpu.copy_(self.S.cpu(), non_blocking=True)` 中 `.cpu()` 本身就是同步 D2H，**non_blocking 无效** | 🟠 假同步 | 删 S_cpu 镜像，门控全 GPU（b_gate / k_embed / q_proj 都在 CUDA） |
| F3 | 预取循环 N 次 `.item()` 触发 GPU→CPU 同步 | 🟠 性能 | 维护 CPU 镜像 `_gpu_resident_cpu` / `_cpu_resident_cpu`，每 `resident_sync_interval=10` 步 sync 一次 |
| F4 | 推理端无 S 范数闸门，**自回归生成 100+ 步 S 范数累积爆炸**，INT8 量化精度退化 | 🟠 正确性 | 加 `s_norm_cap=512`（与训练端一致），纯 GPU `torch.where` 缩放 |
| F5 | `self.W_nvme[idx].astype(np.int8)` 冗余（memmap 本身就是 int8，astype 会再拷一份） | 🟡 浪费 | 去掉 |

修复点在代码里全部用 `# === v3-fix: <id> ===` 标注。

### 训练→推理契约（必读）

推理端有几个**隐式假设**，训练端必须满足：

| 假设 | 来源 | 不满足会怎样 |
|------|------|-------------|
| `W_pool` 已按 `activation_stats` 降序重排 | 训练端必须调 `pool.finalize_inference()` 后再保存 | Hot/Warm/Cold 划分乱套，高频块可能落到 NVMe |
| `scale_pool` 已经是训练末期的最终值 | 训练端 `finalize_inference` 已把它跟 W_pool 一起保存 | INT8 反量化系数不对，所有 GEMM 输出偏移 |
| `b_gate` 是末期锁定后的稳态值 | 训练端末期锁定阶段不更新 b_gate | 推理门控概率分布跟训练不一致 |
| `S_Norm_Cap=512` 两端对齐 | 推理端 INFER_CONFIG['s_norm_cap'] = 512 | 推理 S 累积爆炸或训练时序漂移 |

简单说：**没调 `pool.finalize_inference()` 之前保存的 .bin 不能直接喂给推理端**。

### 推理快速开始

```bash
# 已经在 spike-pool-v3 分支上
# 1. 训练（生成 model_pool.bin + .npy）
python experiments/spike_pool/train_engine.py

# 2. 推理（加载 .bin 跑前向）
python experiments/spike_pool/infer_engine.py
```

跑起来后会看到：

```
📂 Loading model from NVMe...
   Loaded W_pool: (128, 16384, 4096), size: 8.59GB
   Storage hierarchy: Hot=32, Warm=44, Cold=52
   Loaded 32 hot blocks to GPU.
   Loaded 44 warm blocks to CPU pinned memory.
✅ Inference engine ready: 32 hot blocks on GPU, 44 warm blocks on CPU, 52 cold blocks on NVMe.
   GPU memory used (W_gpu): 8.59GB
⏱️ 100 steps inference took 245.32ms (2.453ms/step)
🔮 Generating sequence...
   Generated 50 states, final S norm: 127.84
🛑 Inference engine shut down.
```

### 推理端配置项

`INFER_CONFIG` 字典在 `infer_engine.py` 顶部：

| Key | 默认 | 说明 |
|-----|------|------|
| `gpu_hot_ratio` | 0.25 | Hot 集比例（按训练端 `activation_stats` 排序前 25%） |
| `cpu_warm_ratio` | 0.35 | Warm 集比例（25%~60% 区间） |
| `prefetch_window` | 16 | 每步预取候选块数（覆盖 NVMe 延迟） |
| `top_k_active` | 32 | 每步最大激活块数（与训练一致） |
| `s_norm_cap` | 512 | S 范数硬上限（v3-fix F4，与训练一致） |
| `resident_sync_interval` | 10 | CPU 镜像同步间隔（v3-fix F3） |
| `use_cpu_compute` | True | 启用温块 CPU 异构计算（异构模式开关） |
| `cpu_cores` | 8 | torch CPU 线程数 |

---

## TODO / 已知限制

- [ ] **真实数据集**：`__main__` 现在用 `torch.randn` 模拟目标，需要接 dataloader
- [ ] **评估指标**：目前只监控 S 范数和活跃块数，没有下游 loss/perplexity 报表
- [ ] **K=32 是经验值**：Batch GEMM 在 K=32 时 Tensor Core 利用率最高，但没在 16/64 上对照过
- [ ] **autograd.grad 是同步的**：10k 步会有 ~10k 次同步点，profile 过的话可能需要换 torch.func.vmap 或 functorch
- [ ] **量化感知微调**：当前 scale_pool 手动管理，没用 QAT 工具链
- [ ] **多卡**：单卡设计，DP/TP/FSDP 都没接

### 推理端 TODO

- [ ] **真实 benchmark**：`__main__` 现在用 `torch.randn` 模拟初始 S，没接真实 tokenizer 出来的 embedding
- [ ] **预取命中率报表**：prefetch_buffer cache hit rate / NVMe read count 还没监控
- [ ] **冷块首次访问延迟**：cache miss 同步兜底路径有 ~50-100ms 卡顿，要不要做"宁可跳过该块也不阻塞"
- [ ] **Stream 优先级**：H2D 预取 stream 跟主计算 stream 的优先级没设，可能互相挤
- [ ] **evict_cold_blocks 是手动调**：没接自动显存水位监控（`torch.cuda.memory_reserved()`）
- [ ] **多卡 / 张量并行**：单卡设计，TP 拆分 num_blocks 没做
- [ ] **dynamic shapes**：当前所有张量 shape 写死，d_model/d_inner/num_blocks 改了要重启引擎

## 实验日志

- 2026-08-08：v3-final 训练引擎 review，识别 5 个硬伤（F1-F5）
- 2026-08-08：v3-fix 训练引擎落地，commit `091f3c4`
- 2026-08-08：v3-final 推理引擎 review，识别 5 个硬伤（F1-F5，与训练端平行编号）
- 2026-08-08：v3-fix 推理引擎落地，README 显式化"训练→推理契约"
- ...（开了新坑就往下加）

---

跟主线 v49/v50 一样，**先把骨架立稳，再谈优化**。现在这个版本能跑通，能调通，但是不是真的好使，得拿真实任务试。
