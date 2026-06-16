# CrystaLLM v23 — 扩数据 timeline

> **目标**: 用扩数据 (1893 → N) 让 500M decoder 真正"吃饱", 进一步降 PPL.
> **当前 SOTA**: v22a PPL 4.39, 速度 847ms, KR1.3 0.32× 同规模 AR.
> **状态**: 等待用户提供数据 (本地 2305 已用尽, HuggingFace 网络不通).

## TL;DR — 谁做什么

| 阶段 | 内容 | 责任人 | 估时 | 阻塞 |
|---|---|---|---:|---|
| **P0** | 数据源扩展 + 下载 | 👤 用户 | 1-3 天 | — |
| **P1** | 数据格式验证 + vocab 更新 | 🤖 我 | 1-2 小时 | P0 |
| **P2** | v23 encoder 训 (256z) | 🤖 我 | 30-60 min | P1 |
| **P3** | v23 cached z 提取 | 🤖 我 | 5 min | P2 |
| **P4** | v23 diffusion prior 训 | 🤖 我 | 5-10 min | P3 |
| **P5** | v23 decoder warm-start 训 | 🤖 我 | 30-90 min | P3 |
| **P6** | v23 端到端评估 | 🤖 我 | 10 min | P5 |
| **P7** | v23 报告 + 决策 | 🤖 我 | 30 min | P6 |

**总耗时 (我)**: 2-4 小时, 视数据规模.
**总耗时 (你)**: 1-3 天 (下载/清洗).

## P0 — 数据准备 (用户)

### 数据规模档位

| 档位 | 样本数 | 字符总量 (估) | 训练时间 | 期望 PPL |
|---|---:|---:|---:|---:|
| **v22a (现)** | 1,893 | 12M | (基线) | 4.39 |
| **小扩** | 5,000 | 32M | +30 min | ~4.0 |
| **中扩 (推荐)** | 10,000 | 64M | +60 min | ~3.7 |
| **大扩** | 50,000 | 320M | +3-4 h | ~3.2 |
| **超大** | 100,000 | 640M | +6-8 h | ~3.0 |

**推荐先小扩 (5K) 验证收益曲线**, 再决定是否扩到 10K/50K.

### 数据源候选

| 数据源 | 优点 | 缺点 | 推荐 |
|---|---|---|---|
| **codeparrot-clean-train** | 500K Python 函数, 已清洗 | 需 HF 下载, 网络受限 | ⭐ |
| **codeparrot-clean-valid** | 100K Python 函数, 验证集 | 需 HF 下载 | ⭐ |
| **bigcode/the-stack-smol** | 多语言代码 | 需 HF 下载 | 备选 |
| **GitHub 公开 raw** (e.g. specific repos) | 不需 HF | 需知道具体 repo | 备选 |
| **本地 + 用户补充** | 无网络依赖 | 你手工准备 | 应急 |
| **C4 (Common Crawl)** | 大规模文本 | 太大, 几十 GB, 需下载 | 备选 |

### 数据格式规范 (必须严格)

**v23 接受的数据格式** (与 v22 一致, char-level):

```python
# crystalllm/data/processed/v23_sub.parquet
# 必需列:
#   - text: str (代码/会话文本, 任意长度, 训练时随机切 128 字符窗口)
#   - theme_id: int (可选, 0/1, 用于主题对齐实验; 不用就全填 0)
# 可选列:
#   - session_id, project, rel_path, n_tokens, n_chars
# 无其他必需列
```

**字符 vocab 更新**:
- 现有 char_vocab.json: 2261 entries (3 specials + 2258 chars)
- 新数据可能引入**新字符** (e.g. emoji, CJK, 特殊符号)
- 我会在 P1 阶段合并新字符, 重新生成 vocab
- vocab 扩到 3000-4000 是合理的, decoder 嵌入层会随之扩展 (476M → 477M)

### 数据切分

- **train**: 95% (e.g. 9500 from 10K)
- **val**: 5% (e.g. 500 from 10K)
- 切分 seed: 42 (与之前一致)
- val 必须从新数据中抽, **不要**复用 v16_sub 的 210 (避免数据泄露)

### 数据质量最低要求

- 平均每样本 > 1000 字符 (v22a 中位 7435, 沿用此分布)
- 不要纯空白/单字符重复样本
- 代码/对话/结构化文本优先 (v22a 是会话 + 代码混合)
- 可以保留 markdown 格式 (#, ```, 等)

## P1 — 数据格式验证 + vocab 更新

我会写 `verify_v23_data.py`, 检查:
- parquet 可读, 列名正确
- text 列无空值
- 字符分布, 估算 vocab
- 切分 train/val (95/5)
- 输出 `crystalllm/cached_v23_meta.json` (样本数, 字符数, vocab 增量)

## P2 — v23 encoder 训 (256z)

复用 v22 encoder 架构, 数据换成 v23.

```python
train_v23_encoder.py
- 12L × 768 × 12 + 主题头 256→2 (主题可选)
- 训练: 4000 步, LR 3e-4
- 监督: L_recon + 0.1β*L_KL + 0.5*L_theme (主题可选)
- 估时: 30-60 min (取决于数据量)
```

输出: `v23_encoder.pt` + `v23_encoder_train_log.json`

## P3 — v23 cached z 提取

```python
extract_v23_z.py
- 用 v23 encoder 提取 256 维 z 到 cached_v23_z.npz
- 输出: train_z (N_train, 256), val_z (N_val, 256)
- 估时: 5 min
```

## P4 — v23 diffusion prior 训

复用 v22 prior 架构, D_Z=256.

```python
train_v23_diffusion.py
- D_Z=256, D_HID=512, 6 ResBlock
- 训练: 4000 步, LR 1e-3
- 估时: 5-10 min
```

输出: `v23_diffusion_prior.pt` (best cos_sim 估 >0.95)

## P5 — v23 decoder warm-start

复用 v22 decoder 架构, 从 v22a decoder 权重 warm-start.

```python
train_v23_decoder.py
- 24L × 1280 × 20, 475.65M
- z_to_emb 256→1280 直接复用 v22a (D_Z 都是 256)
- 训练: 4000 步 (vs v22a 2000, 新数据多训)
- LR: 1e-4 (warm-start 慢一点)
- 估时: 30-90 min (取决于数据量, 10K 估 1 hour)
```

输出: `v23_decoder.pt`

## P6 — v23 端到端评估

复用 v22a eval 脚本, 数据换成 v23.

```python
eval_v23_e2e.py
- 三模式 PPL (diffusion_z / encoder_mu / random_z)
- 速度基准 (RTX 5090, batch=1, 100 AR)
- 主题 token 比例 (如启用主题对齐)
- 输出: v23_e2e.json
```

**关键对照**:
- v23 PPL vs v22a PPL (4.39)
- v23 速度 vs 0.32× 同规模 AR
- v23 主题 token 比例 (如启用)

## P7 — v23 报告

`v23_results.md`, 7-10 节:
1. 数据规模/分布/项目分布
2. v23 encoder 训练曲线
3. v23 prior cos_sim
4. v23 decoder 训练曲线
5. 端到端 PPL 对照 (v22a → v23)
6. 速度 KR1.3 验证
7. 主题控制 (如启用)
8. v24 决策树

## 风险与回滚

| 风险 | 应对 |
|---|---|
| 新字符 > 1000 (vocab 爆炸) | 过滤低频字符, 限制 vocab ≤ 5000 |
| decoder 在新数据上过拟合 | 监控 train/val gap, 早停 (patience 500 步) |
| 训练时间超预算 | 减 STEPS 到 2000, 或减数据到 5K |
| PPL 不降反升 | 数据质量低, 检查后回滚到 v22a |
| 网络下载失败 (再试) | 切到本地源 / 用户手工准备 |

## 数据准备 checklist (给用户)

- [ ] 选定数据源 (codeparrot / GitHub / 本地 / 其他)
- [ ] 下载 + 清洗 (PII 过滤, 长度过滤, 字符过滤)
- [ ] 输出 `crystalllm/data/processed/v23_sub.parquet` (含 text 列)
- [ ] (可选) 包含 theme_id 列
- [ ] 通知我数据已就绪, 我跑 P1-P7

## 准备好后告诉我

1. 数据规模 (5K / 10K / 50K / 100K)
2. 数据源 (codeparrot / GitHub / 本地)
3. 是否启用主题对齐 (v22a 失败, **建议不启用**)
4. 训练时间预算 (半天 / 1天 / 2天+)

我会立即执行 P1-P7, 端到端跑完出报告.
