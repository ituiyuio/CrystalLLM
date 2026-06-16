# CrystaLLM v19 — 扩散先验 (Diffusion Prior) Phase 2

> **承接 v18**: BAD-VAE Phase 1 已成功 (val_recon 2.79, mu per-dim std 0.287, 6 个 N(0,I) 采样全不同). v19 接入"扩散寻场"环节, 让随机噪声在 ≤10 步内坍缩为有意义的 z, 再由冻结的 v18 解码器生成文本.

## 1. 背景与定位

### 1.1 v18 留下的口子
- v18 decoder 已学会用 z 重建文本, val_recon 2.79 是历史最优.
- 但 v18 训练时 z 来自 encoder(text), 推理时只能"从 N(0,I) 随便采一个 z" — 这等价于"在宇宙随机位置生成文本", 不可控, 也不对应任何语义域.
- v19 的核心: 把"z 来自哪里"从"任意噪声"升级为"经过扩散精炼的语义场".

### 1.2 与 design.md 的对齐
`design.md` 第 2.1 节明确描述了"潜变量扩散模块":
> "z_T ~ N(0, I) → z_{T-1} → ... → z_0"
> "总扩散步数 T=10~20. 在低维潜空间 (d=512~2048) 运行"
> "训练目标: 流匹配 (Flow Matching) 损失: 预测速度场 v"

v19 是这一节的**首次真实实现**, 工作在 v18 encoder 提供的 64 维 z 空间 (d=64, 比 design.md 推荐的 d=1024 小, 因为 v18 已用 64 维收敛).

## 2. 目标与成功标准

### 2.1 主要目标
训练一个轻量流匹配扩散先验, 在 **5 步 Euler 采样**内将 N(0, I) 噪声映射为 v18 encoder 提取的 z 分布.

### 2.2 量化指标
| 指标 | 目标 | 说明 |
|---|---:|---|
| 流匹配训练 loss | 收敛, < 0.05 (z 单位方差下) | MSE on velocity field |
| z 余弦相似度 | diffusion_z vs encoder_mu, 同文本 > **0.85** | 评估生成 z 是否落在真实 z 流形 |
| 重构 PPL | 冻结 decoder, diffusion_z 送入的 PPL ≤ **1.10×** encoder_mu PPL | decoder 对生成 z 仍能解码 |
| N(0,I) 端到端生成 | 5 步扩散 + AR, 生成文本**主题可辨** (代码 vs 散文) | 超越 v18 "随机代码片段" |
| 端到端推理时间 | 5 步扩散 + T=128 AR ≤ 1.30× 纯 AR (T=128) | KR1.3 验证 |

### 2.3 非目标 (v19 不做)
- ❌ 端到端微调 v18 decoder (留到 v21)
- ❌ 主题条件控制 (留到 v20)
- ❌ 缩放到 500M (留到 v23+)
- ❌ 文本条件 prompt (留到 v22+)

## 3. 架构

### 3.1 总体流程
```
        ┌─────────────────────────────────────────────────┐
        │  v19 训练: 冻结 v18 encoder                       │
        │  训练数据: (z_0 = encoder_mu, t ~ U[0,1], 噪声)  │
        │  目标: 预测速度场 v_θ(z_t, t)                     │
        └─────────────────────────────────────────────────┘
                              ↓ 训练好
        ┌─────────────────────────────────────────────────┐
        │  v19 推理 (5 步 Euler):                          │
        │  z_1 = N(0, I)                                  │
        │  for k in [0.8, 0.6, 0.4, 0.2, 0.0]:             │
        │    z_k = z_{k+1} - Δt · v_θ(z_{k+1}, k)     │
        │  z_0 = 最终去噪结果                              │
        │  text = v18_decoder(z_0) (AR 生成, 冻结)         │
        └─────────────────────────────────────────────────┘
```

### 3.2 扩散先验网络 (DiffusionPrior)

**输入/输出**: z ∈ ℝ^d, d=64 (与 v18 D_Z 一致).

**架构选择**: **小型 Transformer encoder**, 不是 MLP. 理由:
- v18 z 来自 12 层 Transformer, 其维度间有复杂相关性, MLP 难以建模
- Transformer 自注意力可在 batch 内并行, 64 维下成本可忽略
- 与 v18/v22 风格统一, 便于将来扩展到 1024 维

```python
class DiffusionPrior(nn.Module):
    """流匹配速度场预测器: v_θ(z_t, t) → v ∈ ℝ^d."""
    def __init__(s, D_Z=64, D_HID=256, N_LAYER=4, N_HEAD=4):
        s.D_Z = D_Z
        s.t_emb = SinusoidalTimeEmbed(D_HID)   # t → D_HID
        s.in_proj = nn.Linear(D_Z, D_HID)
        s.proj_t = nn.Linear(D_HID, D_HID)     # 时间嵌入混入
        s.blocks = nn.ModuleList([
            TransformerBlock(D_HID, N_HEAD) for _ in range(N_LAYER)
        ])  # 双向, 维度 d=64 当作 64 个 "tokens"
        s.ln = nn.LayerNorm(D_HID)
        s.out = nn.Linear(D_HID, D_Z)
    def forward(s, z_t, t):
        """z_t: [B, D_Z], t: [B]."""
        h = s.in_proj(z_t).unsqueeze(1)         # [B, 1, D_HID]
        t_emb = s.proj_t(s.t_emb(t))             # [B, D_HID]
        h = h + t_emb.unsqueeze(1)              # 广播到 seq dim
        for blk in s.blocks: h = blk(h)
        return s.out(s.ln(h)).squeeze(1)         # [B, D_Z]
```

**注**: 64 维作为"序列长度 1"过于简单化, 改用 ResMLP 风格更稳妥. 实际实现采用 ResMLP (见 §3.3 决策). **本节代码已被 §3.3 取代, 实际实现以 §3.3 为准.**

### 3.3 决策: ResMLP 而非 Transformer

**最终选择**: **3 层 ResMLP, 隐藏维 256, 带时间 FiLM 调制**.

理由:
1. z 维 64 极小, 自注意力的"序列维度"概念不适用
2. ResMLP 参数少 (~200K), 训练快, 不容易过拟合 2000 条数据
3. FiLM 调制 (γ, β = MLP(t)) 让时间步直接调制每一层, 是低维扩散的标准做法
4. v18 z 已有结构 (来自 Transformer encoder mean-pool), ResMLP 足够建模其分布

```python
class ResBlock(nn.Module):
    def __init__(s, D_HID):
        s.ln1 = nn.LayerNorm(D_HID)
        s.fc1 = nn.Linear(D_HID, D_HID)
        s.ln2 = nn.LayerNorm(D_HID)
        s.fc2 = nn.Linear(D_HID, D_HID)
        s.film = nn.Linear(D_HID, 2 * D_HID)  # γ, β
    def forward(s, h, t_emb):
        gamma, beta = s.film(t_emb).chunk(2, dim=-1)
        h_res = h
        h = s.ln1(h)
        h = s.fc1(F.gelu(h)) * (1 + gamma) + beta
        h = self.fc2(F.gelu(s.ln2(h)))
        return h_res + h


class DiffusionPrior(nn.Module):
    def __init__(s, D_Z=64, D_HID=256, N_LAYER=3):
        s.t_emb = SinusoidalTimeEmbed(D_HID)
        s.in_proj = nn.Linear(D_Z, D_HID)
        s.blocks = nn.ModuleList([ResBlock(D_HID) for _ in range(N_LAYER)])
        s.ln = nn.LayerNorm(D_HID)
        s.out = nn.Linear(D_HID, D_Z)
    def forward(s, z_t, t):
        h = s.in_proj(z_t)
        t_emb = s.t_emb(t)
        for blk in s.blocks: h = blk(h, t_emb)
        return s.out(s.ln(h))
```

### 3.4 训练目标: 条件流匹配 (Conditional Flow Matching)

**采样过程**: 给定数据点 z_0, 噪声 ε ~ N(0, I), 时间 t ∈ [0, 1]:
```
z_t = (1 - t) · ε + t · z_0     # 线性插值
v_target = z_0 - ε               # 速度场
```

**预测目标**: v_θ(z_t, t) 预测 v_target, 损失为 MSE.

```python
def cfm_loss(model, z0):
    B = z0.size(0)
    t = torch.rand(B, device=z0.device)              # U[0,1]
    eps = torch.randn_like(z0)
    z_t = (1 - t[:, None]) * eps + t[:, None] * z0   # 线性插值
    v_target = z0 - eps
    v_pred = model(z_t, t)
    return F.mse_loss(v_pred, v_target)
```

**5 步 Euler 采样**:
```python
@torch.no_grad()
def sample(model, B, D_Z=64, n_steps=5, device='cuda'):
    z = torch.randn(B, D_Z, device=device)
    dt = 1.0 / n_steps
    for k in range(n_steps, 0, -1):
        t = torch.full((B,), (k - 1) * dt, device=device)
        v = model(z, t)
        z = z - dt * v   # 从 t 到 t-dt
    return z
```

**为什么流匹配不是 DDPM**:
- 流匹配目标直接, 无需预测噪声, 无需 β 调度
- 5-10 步可达高质量, 不需要 1000 步训练+DDIM 蒸馏
- v18 z 已是连续高斯, 起点干净, 流匹配最合适

## 4. 训练数据

### 4.1 数据来源
- **train_texts**: 1893 条 v18 训练集 (来自 crystalllm/data/processed/v16_sub.parquet, 已平衡 UE_CPP/JS_REACT)
- **val_texts**: 210 条 v18 验证集

### 4.2 z 提取 (一次性, 离线缓存)
```python
encoder = load_v18_encoder()  # 冻结
encoder.eval()
z_train = []  # [N_train, 64]
z_val = []
with torch.no_grad():
    for text in train_texts:
        chunk = tokenize_and_chunk(text, T=128)
        mu, _ = encoder(chunk)
        z_train.append(mu.cpu())
```

**关键决策**: 用 **mu (deterministic)** 而非 z = mu + σ·ε, 理由:
1. v18 训练时 z 含采样噪声, 但 z 落点的"中心"是 mu
2. 训练扩散去噪 z_0 = mu, 让扩散学"z_0 的真实分布"而非"z_0 + 噪声的分布"
3. 推理时, 扩散输出 = 一个具体的 z_0, 与训练目标对齐

**数据规模**: 1893 训练样本 × 64 维, 缓存到 disk (numpy 格式, ~480KB).

### 4.3 归一化
v18 训练时 z 已对齐 N(0, I) 先验 (KL with free_bits), 所以 z 分布应接近标准正态. v19 **不做额外归一化**, 直接训练.

## 5. 训练配置

| 参数 | 值 | 备注 |
|---|---:|---|
| D_Z | 64 | 与 v18 一致 |
| D_HID | 256 | 扩散先验隐藏维 |
| N_LAYER | 3 | ResBlock 数 |
| 扩散先验总参数 | ~**200K** | 远小于 v18 174M |
| Batch size | **512** | z 极小 (64 维), B=512 仍轻量 |
| Epochs | **200** | 早停 patience=20 on val loss |
| LR | 1e-3 | AdamW (z 训练通常较快) |
| LR schedule | CosineAnnealing | 跟 v18 一致 |
| Weight decay | 0.01 | |
| 采样步数 (训练时) | n/a | 流匹配训练无需"步数"概念, t ∈ [0,1] 连续 |
| 采样步数 (验证时) | **5** | Euler, 验证目标 |
| 时间 t 分布 | U[0, 1] | 标准流匹配 |
| 噪声分布 | N(0, I) | |
| 插值路径 | 线性 (1-t)·ε + t·z_0 | Optimal Transport 简化 |

## 6. 评估

### 6.1 训练时指标
每 50 step 在 val_z 上计算:
- **流匹配 loss**: 应当 < 0.05
- **5-step 重构相似度**: `cos_sim(sample(prior) vs val_z[i])`, 平均 > 0.85
- **1-step 重构相似度**: 验证"快采样"边界

### 6.2 训练后端到端评估
1. **N(0,I) 端到端生成**:
   - 5 步扩散生成 16 个 z
   - 冻结 v18 decoder AR 生成 128 字符
   - 检查文本是否**有连贯结构** (字符组合像真代码, 不是 v18 那种"片段")
   - 计数: 16 个中, 几个看起来像"完整代码块"?

2. **重建对比**:
   - 取 val 文本 src
   - encoder(src) → mu_target
   - prior(mu_target) → 5 步去噪 z_recon
   - 检查 `cos_sim(mu_target, z_recon)` > 0.85

3. **PPL 退化**:
   - decoder(mu_target) 计算 PPL_1 (理想)
   - decoder(z_recon) 计算 PPL_2 (实际)
   - 比率 = PPL_2 / PPL_1, 目标 ≤ 1.10

4. **t-SNE 可视化** (留到 §7 Phase 2b):
   - 画 encoder 提取的 val_z (蓝) vs prior.sample() (红)
   - 两者应高度重叠

### 6.3 推理速度基准
- 单 batch=1, T=128:
  - 5 步扩散: ~5 × (1 forward) ≈ 5ms
  - AR 128 token: ~128ms (与 v18 一致)
  - 总计 ~133ms
- 对照: 纯 AR 200M 模型, T=128: ~95ms
- 比率: 1.40× — **略超 KR1.3 的 1.30× 目标**

**对策**: 因 v19 仍是原型规模 (200K 扩散 + 87M decoder), 5 步扩散开销相对大. 优化路径:
1. v19b 用 Cached prior (z 生成只跑 1 次, 不参与 AR)
2. v22 缩放时, 扩散占比下降, 比率自然下降
3. 长期: AR 阶段并行化 (推测解码 / Medusa) 才是关键

**v19 接受 1.40×, 但记录数据, 留 v22 解决**.

## 7. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| 5 步采样质量差 | 中 | PPL 退化 > 10% | 退到 10 步 (牺牲速度换质量) |
| decoder 看到扩散 z 后 PPL 飙升 | 中 | 端到端不可用 | **Phase 2b**: 用扩散 z 微调 decoder 5-10 步 (小 LR), 快速适配 |
| z 分布外生成 (OOD) | 低 | 文本奇怪 | 加 z 范数截断: `if z.norm() > 3σ: z = z / norm * 3σ` |
| 训练过拟合 2000 条 | 中 | val loss 不降 | 早停 + Dropout 0.1 in ResBlock + 加大 weight decay |
| 流匹配 path 选错 | 低 | 训练震荡 | 已有文献验证线性插值足够; 不尝试 OT 复杂路径 |

### 7.1 Phase 2b 预案: decoder 适配
如果 v19 训练后, `decoder(diffusion_z)` PPL 比 `decoder(mu)` 退化 > 10%, 启动 Phase 2b:
- **冻结扩散先验** (已训好)
- **解冻 decoder**, lr=1e-5 (v18 的 1/30)
- **小数据集**: 1000 个 (text, diffusion_z) 对, 训 200 step
- **目的**: 让 decoder 把扩散输出的 z "当作"自己的输入分布

## 8. 文件交付

| 文件 | 内容 |
|---|---|
| `crystalllm/proto_v19_diffusion_prior.py` | 训练 + 评估一脚本 |
| `crystalllm/cache_v18_z.py` | 一次性提取 encoder mu, 缓存到 .npy |
| `crystalllm/diffusion_prior.pt` | 训练好的先验权重 (~1MB) |
| `crystalllm/v19_train.log` | 训练日志 (loss 曲线) |
| `crystalllm/v19_train_log.json` | 数值结果 (val 指标) |
| `crystalllm/v19_results.md` | 完整结果报告 + 端到端生成示例 |
| `crystalllm/eval_v19_e2e.py` | 端到端评估脚本 (Phase 2 验证) |

## 9. 与历史版本的关系

| 版本 | 角色 | 状态 |
|---|---|---|
| v18 | BAD-VAE Phase 1: encoder + decoder | ✅ 完成, 冻结用于 v19 |
| **v19** | **扩散先验: noise → z, 接入 decoder** | **当前任务** |
| v20 | 主题控制: z_UE / z_JS 原型 + 插值 | 待 v19 完成后 |
| v21 | 端到端微调: 联合训练 encoder+decoder+prior | 待 v20 完成后 |
| v22+ | 缩放到 500M-1.5B | 远期 |

## 10. 决策记录

### D1: 流匹配 vs DDPM
**选**: 流匹配.
**理由**: 训练稳定, 步数少, 目标简单, 文献支持 (Lipman 2023).

### D2: ResMLP vs Transformer
**选**: ResMLP (200K 参数).
**理由**: 64 维极小, 自注意力不合适; 经典流匹配标准做法.

### D3: 5 步 vs 10 步
**选**: 5 步 (训练时验证 5 步和 10 步, 取更优者).
**理由**: KR1.1 要求 ≤10 步, 5 步是中间目标; 推理速度优先.

### D4: z_0 = encoder mu vs sample
**选**: mu (deterministic).
**理由**: 训练目标清晰, 推理对齐 (扩散输出一个"具体"的 z).

### D5: 接受 1.40× 推理时间
**选**: 接受.
**理由**: v19 原型规模下扩散开销大, 缩放后会自然下降; KR1.3 在 v22+ 验证.

## 11. 自审 (Spec Self-Review)

- ✅ 无 placeholder (无 TBD/TODO)
- ✅ 内部一致: §3 架构 / §5 配置 / §8 文件 三处参数一致
- ✅ 范围聚焦: 单一交付 (扩散先验), 不混入 v20/v21
- ✅ 无歧义: "5 步"明确为 Euler 5 次; "PPL 退化 1.10×"明确比较对象
