# spike_pool 实验线 C — chunk 化 + 双流预取 + CPU pinned 池

**日期**: 2026-09-06
**分支**: `spike-pool-v3`
**动机**: SpikeLLM 训练撞存储墙 — 性能全部浪费在数据搬运上，Tensor Core 喂不饱。

## 存储墙的定量分析

RTX 5090 roofline: ~1.8TB/s HBM / ~200TFLOPS BF16 → 拐点 ~117 FLOP/byte，
即每个权重要被复用 ~60 次才喂得饱 TC。

基线 `forward_step_state` **每个 token 位置**都要走一遍全流程：

| 步骤 | 流量/位置 (d=4096,K=16) | 复用次数 |
|------|------------------------|---------|
| gather INT8 + float→bf16 dequant | ~2.4GB (读268MB+FP32中间+写536MB) | 1 |
| W_active zero_ + bmm | ~1.1GB | 256 (batch 维) |
| manual-grad 外积 + FP32 cast | ~1.6GB | 1 |
| update_buffer RMW + rand_like + commit | ~2.1GB | 1 |
| **每位置 empty_cache()** | 全堆扫描 | - |

×62 位置/step ≈ **350GB+/step**，其中 bmm 以外的全部是零 FLOP 搬运。

关键认识：**INT8 量化对带宽墙中性**（字节和 FLOP 同比例缩），
真正的问题是「每位置重复搬运 + 零复用」。

## 三个改动

### C1. chunk 级摊销
- `shared` 模式: 一次 gather/dequant 服务 C 个位置（C=31 → 摊销 31 倍）
- manual-grad 从「每位置一次 [d_inner,d_model] RMW」变成「每 chunk 一次
  掩码 GEMM」: `acc[k] = Σ_t M[k,t] · R[t] ⊗ S0[t]`，exact 模式下
  M 是位置-块活跃掩码（数学等价于逐位置更新），shared 下 M 全 1
- commit 每 chunk 一次（`fire.any()` 短路省掉空 RMW）
- 彻底移除每位置 `empty_cache()`

### C2. token 门控（预取前提）
chunk 的路由由 chunk 内 token embedding 均值决定（替代 S_old 均值），
因此 **全部 chunk 的路由在 forward 开始前就已知**。
基线路由本来就近似随机（b_gate 除块 0 全 -1e6，content bias 量级可忽略），
换门控源代价很小。`exact` 模式保留逐位置 state 门控做对拍。

### C3. CPU pinned 池 + 预取线程 + 异步 commit 回写
- W_pool INT8 驻留 pinned RAM（`index_select` 进 pinned 中转再 DMA，
  避免 pageable 通道减速）
- 后台线程双缓冲 slot（2×268MB INT8 staging），copy stream 与计算流重叠
- 跨步 lookahead: 训练循环在 step s 结束时下发 step s+1 的拷贝，
  chunk0 的 PCIe 藏进 step s 的 backward
- commit 增量经 pinned staging 异步 D2H + 线程 scatter 回 CPU 池
- update_buffer 强制驻 GPU（更新路径全 GPU）
- **eval/generate (no_grad) 走同步 gather 路径**——预取的 slot 保证
  只对「issue 后按序消费」的训练流成立，eval 插队会撞 slot

## 验证结果 (2026-09-06, RTX 5090, torch 2.9.1+cu128)

### Part A/B: 数值对拍 (tiny: d=256, blocks=16, K=4, B=3, L=24)
双方 threshold commit（排除随机提交的 RNG 分歧）：

```
[A] baseline loss  = 113.11695862
[A] chunk C=1 loss = 113.11695862   rel diff = 0.00e+00   ← bitwise 相等
[A] update_buffer diff: max=2.6e-4 (BF16 求和顺序，不进前向)
[B] chunk exact C=6 rel diff = 0.00e+00              ← union 摊销数学等价
[A.c1/c2] stochastic/threshold commit 通路 PASS
```

### Part C: 吞吐矩阵 (d=4096, blocks=64, K=16, B=256, L=64)

| 变体 | ms/step | speedup |
|------|---------|---------|
| baseline（每位置全流程） | ~8270 | 1.00x |
| exact C=1（只去 empty_cache） | ~8190 | 1.01x |
| exact C=31（union 更新） | ~1590 | 5.21x |
| shared C=8 | ~1230 | 6.75x |
| **shared C=31 (GPU 池)** | **435** | **19.1x** |
| **shared C=31 (CPU pinned 池+预取)** | **534** | **15.5x** |

结论：
1. empty_cache 不是主因（C=1 只 1.01x）；**更新路径摊销才是大头**（exact 5.2x）
2. shared C=31 再拿 3.7x（dequant 摊销 + bmm 纯化）
3. CPU 池只比 GPU 池慢 23%——**池子容量不再受 VRAM 限制**，
   256b@4096 只是小试，512b/1024b 可以直接上（W_pool 在内存条里）

### Part D: 短训 PPL 对比 (120 步, d=4096, blocks=64, B=256, L=64)

```
[baseline      ] loss 8.42→8.34 | FINAL val ppl 3887 | sps 0.13
[chunk C=31 cpu] loss 8.42→8.28 | FINAL val ppl 3825 | sps 1.07   ← 8.46x
```

**学习质量持平（chunk 略好，在噪声内），端到端 8.46x 提速。**
端到端倍数低于 Part C 的 15.5x，因为这里含 optimizer.step / scheduler /
batch 抓取——chunk 化只加速池子路径，embed/head 的 AdamW 开销不变。
chunk val ppl 3825 与历史 256b_4096 1000 步跑法 (ppl 3968) 同档，
确认 shared C=31 路由粗化没有伤害学习。

## 过程中发现的基线 bug（train_engine.py 未修，仅记录）

1. **int8 commit 回绕溢出**（所有 `_commit_updates` 路径）:
   `torch.clamp(W_pool.int8 + rounded.int8, -128, 127)` 里 int8+int8 加法
   **先回绕后 clamp**（126+3 → -127），clamp 救不回。权重靠近 ±128 的元素
   更新方向会翻转。chunk_pool 侧已修（int16 加法再 clamp）。
   对拍中实测 tiny 池 771/262144 元素处在回绕区。
2. **每位置 `empty_cache()`**（`forward_step_state` 末尾）:
   F21 注释说「不释放会累积」，但 F13 预分配后 W_active 是共享 buffer，
   实测去掉后 C=1 吞吐不变（1.01x），说明它只是纯开销。

## 已知偏差（实验接受）

- shared 模式路由粒度粗化（C 个位置共用一组块）——这是带宽-精度 trade 的
  实验变量本身，Part D 验证其学习影响
- CPU 池下 chunk c+2 可能读到 chunk c commit 前的权重（stochastic fire
  元素 ~0.1%，±1 int8 单位）
- manual-grad 求和顺序与基线不同（FP32 求和后转 BF16 vs BF16 逐 sample 累加），
  update_buffer 差 ~1e-3 相对量级，不进前向
- chunk 粒度的 b_gate EMA / burst 统计（`decay**C_eff` 近似）

## 文件

- `chunk_pool.py` — ChunkedSpikePool + ChunkPrefetcher（线程协议见 docstring）
- `spike_llm_chunk.py` — SpikeLLMChunk（LM 包装）+ CLI
- `_verify_chunk.py` — Part A/B/C/D 验证与基准

## 运行

```bash
# 对拍 + commit 单元检查
python experiments/spike_pool/_verify_chunk.py --parts A
# 吞吐矩阵
python experiments/spike_pool/_verify_chunk.py --parts C
# 短训 PPL 对比
python experiments/spike_pool/_verify_chunk.py --parts D --train_steps 120
# 直接训练 (跟 spike_llm.py 同参)
python experiments/spike_pool/spike_llm_chunk.py --d_model 4096 --num_blocks 256 \
    --top_k 16 --batch_size 256 --seq_len 64 --chunk_size 31 --pool_device cpu
```

## 下一步（按预期收益排序）

1. **bmm 内跨位置批量化**: 现在 bmm 仍逐位置调用（S 序列递归依赖），
   62 次/step × 每次 536MB W 读。ISS 递归若展开成 scan 形式或
   chunk 内近似并行（用 chunk 首位置 S 近似），可再消 ~60% bmm 流量
2. **受控偏斜 + 异构分层**: 磨损均衡改「防塌缩」而非「抹平」，
   热块自然驻 GPU、冷块下沉 CPU（PowerInfer 的 Zipf 论证）
3. **CUDA graph 稳态循环**: 生命周期管理（CPU 侧）与 GEMM 稳态分离后，
   GPU 流可以 capture

---

# 证伪阶梯（2026-09-07 追加）— 回答「PPL 4000 是不是走错路」

工具: `_falsify_ladder.py`（E0 标定 + E1-E6 消融，全部同数据同协议同预算）
**方法论**: 不再加速，先标定天花板，再逐层隔离瓶颈。

## E0 标定（同一条 803k token 语料，in-sample）

| 模型 | NLL | PPL |
|------|-----|-----|
| uniform | 8.318 | 4096 |
| unigram | 7.036 | 1137 |
| bigram | 3.545 | 35 |
| **SpikeLLM 原版（1000 步）** | **8.29** | **~3970** |

原版比「只背词频」还差 1.2 nats——不是学得慢，是被锁死。

## 两个根因

1. **架构（致命）**: 递归方程 `S_t = γS + g·norm(W@S)` **没有输入项**。
   `embed(x_t)` 只进 MSE target，从不进状态。状态链是只由 x₀ 启动的
   自治系统，当前 token 信息永远进不了 S → 理论天花板 ≈ unigram 附近。
2. **目标（次要）**: 池子学习信号是 MSE 代理（把 S 推向下一 token 的
   embedding），不是 CE。E2 证明换 CE 残差单因素就有 2x 改善。

## 阶梯结果（300 步，d=4096/blocks=64/K=16/B=256，CE 残差注明）

| 变体 | val PPL | 结论 |
|------|---------|------|
| E1 纯外壳（无池子，直注 input，全 CE） | **32** | 外壳和优化全没问题，已到 bigram 档 |
| E2 池子 only + CE 残差（无输入项） | 1920 | 目标函数是真实瓶颈 |
| E3 池子 + 逐通道 γ | 2714 | 递归容量不是瓶颈 |
| E4 池子 sum 注入（W@S 淹死输入，130x 尺度差） | 1797 | 共享 RMS_norm 不行 |
| **E5 旁挂: S=γS+g·norm(embed)+g_w·norm(W@S) + CE 残差** | **38** | 追平无池子外壳 |
| E6 E5 + BPTT 真梯度（B=128/150 步） | 236* | g_w 0.05→0.013，池子项被静音 |

*E6 用小 batch 短程（BPTT 反传重）。g_w 收缩 = 在此预算下池子项还没
挣到自己的饭钱。

## 诊断过程中的关键教训

1. **E4 尺度分析**: 块 0 随机 int8 × scale 的 W@S 元素 RMS ~1.3，
   embed 元素 RMS ~0.01——混进同一个 RMS_norm 输入方向只剩 1/130。
2. **manual 更新与 optimizer 的时序**: manual 池子更新不受 scheduler
   约束，warmup 期会用未训练 head 的噪声梯度全速写 W_pool。
3. ladder 训练循环一度把 `opt.zero_grad()` 放到 `backward()` 之后
   （E6 挂钩改坏的），E5/E6 的「死平 8.41」全是它造成的假象——
   因子消除法（L0/P1）定位。

## 判决

- **原设计确实错了**，但错在「方程」不在「想法」：输入驱动 + CE 残差
  + 旁挂归一化的池子（E5）从 PPL 4000 → 38，不再有毒。
- **池子尚未证明增值**：E5 (38) 略落后于无池子外壳 (32)，E6 里 g_w
  被收缩。下一步是 2k-5k 步长跑并跟踪 g_w：
  - g_w 上升且 E5 > E1 → 池子有真实容量，路线成立，上 attention 组合
  - g_w → 0 → 池子作为 LM 计算判死；转型 (a) attention 旁挂记忆
    （Titans 形态，E7）或 (b) 回到 FFN 替换的原始语境（spike_pool v3
    的初衷，那里没有「输入通路」问题）
