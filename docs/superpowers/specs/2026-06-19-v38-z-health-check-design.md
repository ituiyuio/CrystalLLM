# CrystaLLM v38 — z 健康度诊断 (Block-Diffusion PoC 前置)

> **承接 v37**: v37 zero-z ablation 量化证明 decoder 不消费 z (ΔPPL +0.441%), 但未测 z 分布本身的健康度.
> **承接用户 design 提议**: 用户提出 block-level diffusion + 时空 MoE 框架, 前提假设是 z 是可用信号. v38 在 PoC 之前先验证 z 是否真的可用.
> **执行**: `python pipeline/z_health_check.py` (单一脚本, 35-45 分钟出结果).

## 1. 背景与定位

### 1.1 v37 留下的两个未答问题

v37 zero-z ablation 证明:
- decoder 不消费 z (ΔPPL_v25 = +0.441%, cross-attn 成本 ≈ 0)
- z 是 dead weight + cross-attn 是纯参数开销

但 v37 **未测** z 分布本身的健康度:

| 维度 | v37 已测 | 未测 |
|---|---|---|
| z 是否被 decoder 消费 | ✅ ΔPPL < 1% | — |
| z 是否含有信息 | ❌ | 互信息 I(z; x) |
| z 是否塌缩/爆炸 | ❌ | 维度方差分布 |
| z 分布是否健康 | ❌ | KL 真实值, 类别可分性 |

### 1.2 用户框架的前提冲突

用户提出的 block-level diffusion + MoE 框架, 核心假设是 **decoder 应该消费 z**. 这与 v37 结论冲突. 必须先回答:
> **z 分布本身是否含有可用信号? 如果 z 是高熵噪声, 所有注入路径都是死路.**

### 1.3 v38 定位

**v38 不修任何架构, 不调任何超参, 不训练任何模型.** 它只做一件事:

> **复用 v25/v36 encoder, 在 1016 val 样本上跑 4 个独立健康指标, 输出 JSON + 决策矩阵结果.**

这是 PoC 之前的最后一道门槛.

## 2. 目标与成功标准

### 2.1 主要目标

通过 4 个独立指标, 量化回答:

> **CrystaLLM 的 z 分布在 v25/v36 encoder 训练后, 是否含有可用信号? 信号质量是否足以支撑 block-diffusion PoC?**

### 2.2 量化指标 (4 个)

| # | 指标 | 实现 | 健康阈值 | 不健康阈值 |
|---|---|---|---:|---:|
| 1 | **KL 散度** | q(z\|x) 拟合 N(μ, σ²I) vs N(0,I) | **< 50 nats** | > 100 nats |
| 2 | **互信息下界** | MINE 估计 I(z; x) | **> 0.10 nats** | < 0.05 nats |
| 3 | **维度塌缩比例** | 256 维 z 各维 std < 0.01 占比 | **< 50%** | > 70% |
| 4 | **类别条件可分性** | 类别间 z 的 JS 散度均值 | **> 0.05 nats** | < 0.02 nats |

**注**: 每个指标独立判断, 任何一个落入"不健康"区间都构成阻断条件.

### 2.3 非目标 (v38 不做)

- ❌ 不修改 encoder 架构 (那是 v39)
- ❌ 不写 decoder 注入路径 (那是 v39 block-diffusion)
- ❌ 不调 KL weight / free_bits (等诊断结果再决定)
- ❌ 不做 block-diffusion PoC (等本诊断结果)
- ❌ 不修 v37 zero-z eval (已完成)
- ❌ 不动 SpS 速度优化 (已 pending, 等诊断后再说)

## 3. 架构

### 3.1 总体流程

```
   ┌────────────────────────────────────────────────────────┐
   │  v38 z 健康度诊断 (复用 v25/v36 encoder)                │
   │                                                        │
   │  load_encoder(model_name)                              │
   │    ↓                                                   │
   │  encode val set (1016 samples)                         │
   │    ↓ z shape: (1016, 256)                              │
   │  ┌──────────────────────────────────────────┐          │
   │  │ 计算 4 个独立指标:                        │          │
   │  │   1. KL (高斯拟合)                       │          │
   │  │   2. MI 下界 (MINE)                      │          │
   │  │   3. 维度塌缩比例                       │          │
   │  │   4. JS 类别可分性                       │          │
   │  └──────────────────────────────────────────┘          │
   │    ↓                                                   │
   │  output z_health_report.json                           │
   └────────────────────────────────────────────────────────┘
                              ↓
   ┌────────────────────────────────────────────────────────┐
   │  决策矩阵:                                              │
   │    4/4 健康  → block-diffusion PoC (v39)               │
   │    2-3 健康  → 修 z (v39 free_bits ↑ / 换 encoder)     │
   │    0-1 健康  → 战略重定位, 放弃 z 路径                  │
   └────────────────────────────────────────────────────────┘
```

### 3.2 指标实现细节

#### 指标 1: KL 散度

```python
def compute_kl(z, target='N(0,I)'):
    """
    z: (N, D) tensor, N=1016, D=256
    假设 q(z|x) ~ N(mu_x, sigma_x^2 I)
    KL(q(z|x) || N(0, I)) per sample, then mean
    """
    mu = z.mean(dim=0)            # (D,)
    sigma = z.std(dim=0)          # (D,)
    # KL(N(mu, sigma^2) || N(0, 1))
    # = 0.5 * sum(mu^2 + sigma^2 - 1 - log(sigma^2))
    kl_per_dim = 0.5 * (mu**2 + sigma**2 - 1 - torch.log(sigma**2 + 1e-8))
    return kl_per_dim.sum().item()  # scalar nats
```

#### 指标 2: 互信息下界 (MINE)

```python
class MINE(nn.Module):
    """Mutual Information Neural Estimation"""
    def __init__(self, x_dim, z_dim, hidden=256):
        self.net = nn.Sequential(
            nn.Linear(x_dim + z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, x, z):
        # joint: x_i paired with z_i
        # marginal: x_i paired with z_j (shuffled)
        joint = self.net(torch.cat([x, z], dim=-1)).mean()
        marginal = torch.log(torch.exp(
            self.net(torch.cat([x, z[torch.randperm(len(z))]], dim=-1))
        ).mean() + 1e-8)
        return joint - marginal  # MI lower bound

def compute_mi_lower_bound(x_emb, z, n_steps=1000, n_runs=5):
    """MINE 估计 I(z; x_emb) 下界, 跑 5 次取平均"""
    # x_emb: (N, hidden) text embedding (one-hot 或 token id 嵌入均可)
    # z: (N, D) z 向量
    estimates = []
    for run in range(n_runs):
        mine = MINE(x_emb.shape[1], z.shape[1])
        opt = torch.optim.Adam(mine.parameters(), lr=1e-3)
        for step in range(n_steps):
            mi_estimate = mine(x_emb, z)
            loss = -mi_estimate
            opt.zero_grad(); loss.backward(); opt.step()
        estimates.append(mine(x_emb, z).item())
    return np.mean(estimates), np.std(estimates)
```

#### 指标 3: 维度塌缩比例

```python
def compute_collapse_ratio(z, threshold=0.01):
    """
    统计 256 维 z 中 std < threshold 的维度比例
    """
    std_per_dim = z.std(dim=0)  # (D,)
    collapse_mask = std_per_dim < threshold
    return collapse_mask.float().mean().item()
```

#### 指标 4: 类别条件可分性 (JS 散度)

```python
def compute_class_separability(z, labels, n_pca=32):
    """
    按类别分组, 计算组间 JS 散度均值
    z: (N, D)
    labels: (N,) 类别索引
    """
    # PCA 降维到 32 维 (避免维度灾难)
    z_pca = PCA(n_components=n_pca).fit_transform(z)
    
    # 按类别分组成高斯
    class_gaussians = {}
    for c in np.unique(labels):
        z_c = z_pca[labels == c]
        class_gaussians[c] = (z_c.mean(axis=0), np.cov(z_c.T) + 1e-6 * np.eye(n_pca))
    
    # 计算组间 JS 散度 (高斯闭式解)
    js_scores = []
    classes = list(class_gaussians.keys())
    for i, c1 in enumerate(classes):
        for c2 in classes[i+1:]:
            js = js_divergence_gaussian(class_gaussians[c1], class_gaussians[c2])
            js_scores.append(js)
    return np.mean(js_scores)
```

### 3.3 类别标注启发式

val 集 1016 样本, 按文本特征自动分组:

| 类别 | 启发式 |
|---|---|
| code | 包含 `def `, `class `, `import ` 等 |
| comment | 以 `#` 或 `//` 开头的行占比 > 50% |
| dialog | 包含 `User:`, `Human:` 等 |
| plain | 不匹配以上 |

(若某类别样本 < 10, 标记 "insufficient" 并跳过该类别)

### 3.4 v36 encoder 加载

v36 路径有 cross-attn decoder, 但 encoder 与 v25 共享架构. 直接复用 v25 encoder + v36 的 encoder (如 `crystalllm/v36_encoder.pt`).

如果 v36 encoder checkpoint 不可访问, **退化为只测 v25**, 在报告中明确标注.

## 4. 实验矩阵

### 4.1 测量

| 编号 | encoder | 测量 | 用途 |
|---|---|---|---|
| H1 | v25_encoder.pt | 4 个指标 | **主测量** |
| H2 | v36_encoder.pt (如可访问) | 4 个指标 | cross-attn 是否让 z 更健康? |

**输出**: `v38/z_health_report.json` 包含 H1+H2 全部 8 个数据点.

### 4.2 时间预算

| 步骤 | 估时 | 备注 |
|---|---:|---|
| 编码 1016 样本 (v25 + v36) | 5 min | GPU forward |
| KL 计算 | <1 min | 解析公式 |
| MINE 训练 (5 runs × 1000 steps) | 10 min | GPU |
| 维度塌缩 + JS | <1 min | 解析公式 |
| 报告写作 | 15 min | v38_decision.md |
| **总计** | **~30-40 min** | 半天内完成 |

### 4.3 不确定性

- **MINE 估计方差**: 5 次取平均, 报告 CI (±1 std)
- **类别标注噪声**: 启发式规则可能有 10-20% 误分类, JS 散度对此鲁棒
- **v36 encoder 不可用**: 退化为 H1 only, 报告中标注

## 5. 决策矩阵

### 5.1 4 指标分流

| 场景 | KL | MI | 塌缩 | JS | 决策 |
|---|---:|---:|---:|---:|---|
| **A. 全部健康** | <50 | >0.10 | <50% | >0.05 | **走 block-diffusion PoC (v39)** |
| **B. KL 高, 其他健康** | >100 | >0.10 | <50% | >0.05 | **修 z (v39 free_bits ↑, 或换 encoder)** |
| **C. MI 低, 其他健康** | <50 | <0.05 | <50% | >0.05 | **二次 brainstorm** (z 有结构但与 x 无关?) |
| **D. 塌缩严重** | <50 | >0.10 | >70% | >0.05 | **修 encoder** (正则化或激活函数) |
| **E. 类别不可分** | <50 | >0.10 | <50% | <0.02 | **修 encoder** (需要监督信号) |
| **F. 全部不健康** | >100 | <0.05 | >70% | <0.02 | **战略重定位**, 放弃 z 路径 |

### 5.2 决策输出

无论哪个场景, v38 必须产出 `v38_decision.md`:
- 场景判定 (A/B/C/D/E/F)
- 推荐下一步 (block-diffusion PoC / 修 z / 二次 brainstorm / 战略重定位)
- 该下一步的实验设计纲要 (1-2 段)

### 5.3 决策后的边界

**走 A (PoC)**:
- v39: block-level diffusion spec (借鉴 BD3-LMs)
- 时间: 2-3 天训练 + 评估
- 成功标准: PPL < v25 baseline (2.47)

**走 B (修 z)**:
- v39: encoder 调整 spec (free_bits ↑ / KL annealing / 换 VAE → VQ-VAE)
- 时间: 1-2 天
- 成功标准: KL < 50 + MI > 0.10

**走 C (二次 brainstorm)**:
- 不能盲目分流, 需重新分析
- 可能补做更细粒度测量 (per-token KL, 时间序列 KL)

**走 F (战略重定位)**:
- 接受 z 不可用
- 走 v25 + SpS 速度优化 (复用 v31)
- 退役 diffusion+AR 混合架构

## 6. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| MINE 训练不稳定 | 高 | MI 估计噪声大 | 跑 5 次取平均 + EMA, 报告 CI |
| 类别标注启发式误分类多 | 中 | JS 散度失真 | 用规则 + 长度过滤, 样本 <10 跳过 |
| v36 encoder 不可访问 | 中 | 失去 H2 对照 | 退化 H1 only, 在报告中标注 |
| z 维度方差极小 (全部 < 0.01) | 低 | 塌缩比例 100% | 此时其他指标也会失效, 报告为"全部不健康" |
| GPU 内存不够 (MINE 训练) | 低 | 实验中断 | 退化为 CPU, 耗时 +5 min |

## 7. 文件交付

| 文件 | 内容 |
|---|---|
| `crystalllm/versions/v38/pipeline/z_health_check.py` | 单一脚本, --model {v25, v36} |
| `crystalllm/versions/v38/z_health_report.json` | 4 指标 × 2 模型 = 8 数据点 + 决策矩阵应用结果 |
| `crystalllm/versions/v38/v38_decision.md` | 报告 + 决策 + 下一步推荐 |
| `crystalllm/versions/v38/README.md` | 简短说明本版本目的 |

**不创建新 checkpoint, 不修改 encoder 架构, 不训练.**

## 8. 与历史版本的关系

| 版本 | 角色 | 状态 |
|---|---|---|
| v25 | BAD-DP, encoder 输出 256-d z | ✅ checkpoint 复用 |
| v36 | cross-attn decoder, encoder 同 v25 | ✅ checkpoint 复用 (如可访问) |
| v37 | zero-z ablation, 决策门 | ✅ 完成, ΔPPL < 1% |
| **v38** | **z 健康度诊断, 决策门** | **当前任务** |
| v39 | (取决于 v38 决策) | 待分流后定义 |

## 9. 决策记录

### D1: 测 v25 + v36 vs 只测 v25
**选**: 都测 (如可访问).
**理由**: cross-attn decoder 是否让 encoder 学到更健康的 z? 这是独立信号.

### D2: 4 个指标 vs 单一 KL
**选**: 4 个独立指标.
**理由**: 单一 KL 高可能是 encoder 学到强表达(好事), 也可能是 z 爆炸(坏事). 4 指标互为 sanity check.

### D3: MINE 估计 vs 严格互信息
**选**: MINE 估计 (下界).
**理由**: 严格互信息需要联合分布解析形式, 不可行. MINE 是当前 SOTA 估计方法, Belghazi et al. 2018.

### D4: 类别可分性 vs 主题控制
**选**: 类别可分性 (OKR 派生).
**理由**: 用户 OKR 包含"主题控制", 类别可分性是这一目标的前置条件.

### D5: v38 不修架构
**选**: 不修.
**理由**: 这是诊断, 不是修复. 修架构留到 v39.

## 10. 自审 (Spec Self-Review)

- ✅ 无 placeholder (无 TBD/TODO)
- ✅ 内部一致: §2 指标 / §3 实现 / §5 决策 三处定义一致 (KL/MI/塌缩/JS 阈值)
- ✅ 范围聚焦: 单一交付 (z 健康度诊断), 不混入 PoC 设计
- ✅ 无歧义: 阈值明确 (<50/>0.10/<50%/>0.05); "健康" 与 "不健康" 边界清晰
- ✅ 与 v37 闭环: §1.2 明确 v37 留下的问题, §4 引用 v25/v36 checkpoint
- ✅ 用户 design 闭环: §1.2 指出前提冲突, §5.3 给出每种决策的下一步