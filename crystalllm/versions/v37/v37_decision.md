# v37 Zero-z Ablation — 决策报告

> **承接 v36**: cross-attn 实验未达 PPL 目标. v37 通过 zero-z ablation 量化 decoder 对 z 的真实依赖度.
> **执行**: `python pipeline/zero_z_eval.py` × 4 (A1-A4) + `evaluation/gen_samples.py` × 4.

## 1. 实验结果

### 1.1 PPL 矩阵

| 编号 | 模型 | z_mode | PPL | Δ vs encoded |
|---|---|---:|---:|---:|
| A1 | v25 | encoded | **2.4605** | (baseline) |
| A2 | v25 | **zero** | **2.4713** | **+0.441%** |
| A3 | v36 | encoded | **2.7982** | (baseline) |
| A4 | v36 | **zero** | **2.8093** | **+0.396%** |

### 1.2 交叉对比

- **v25 ΔPPL (zero vs encoded)**: +0.441%
- **v36 ΔPPL (zero vs encoded)**: +0.396%
- **cross-attn 贡献 (v36_zero - v25_zero)**: +0.3380 PPL
- **cross-attn 成本 (encoded 模式)**: +0.3378 PPL
- **cross-attn 成本 (zero 模式)**: +0.3380 PPL
- **cross-attn 成本差异 (enc - zero)**: -0.000217 PPL — **若接近 0, 说明 cross-attn 成本与 z 完全无关**

### 1.3 生成质量

| 模型 + z_mode | 非空格率 | 代码结构样本数 |
|---|---:|---:|
| v25 + encoded | 78.4% | 3/10 |
| v25 + zero | 84.8% | 2/10 |
| v36 + encoded | 83.6% | 4/10 |
| v36 + zero | 85.2% | 5/10 |

## 2. 决策

**场景**: A (z 死路, 但 cross-attn 有独立贡献)

**核心结论**: z 死路 + cross-attn 自身引入噪声 — 战略重定位路径 (但 cross-attn 也需重新审视)

## 3. 推荐下一步

### 若场景 A (z 死路, ΔPPL_v25 < 1%, cross-attn 贡献 < 0.05)

**走 C 战略重定位**:
- 接受 decoder 不消费 z 的事实
- 重新定义"信息结晶"含义: z 不是生成路线的输入, 而是 SpS 路由 / 数据压缩探针 / 可控性接口
- 或: 放弃混合, v25 + SpS 走速度优化路径 (复用 v31 思路)
- 不再做"让 decoder 用 z"的尝试

### 若场景 B (z 微弱信号, 1-5%)

**二次 brainstorm**:
- v22a 已验证 z 编码完美 (主题 acc 75-94%), 但 decoder 不消费
- 可能中间状态: z 有信息但 decoder 容量饱和 (v22a PPL 范围 0.4% = decoder 忽略 z)
- 需补做更细粒度 ablation: 部分维度 z=0, 维度子集测试

### 若场景 C (z 真有用, >5%)

**走 B v37 prefix-tuning**:
- 设计: z 拆成 M=8 memory tokens, 每层 prefix-tuning
- 比 cross-attn 更轻量, z 信息可选择性使用
- 但 KL=303 仍待修 (v38 路径)

## 4. 与 OKR 的关系

若决策为 C 战略重定位, 需:
- 更新 goal.md: KR1.2 "z 为全局条件" 措辞改为 "z 为可选上下文/可控性接口"
- KR3.1 (主题控制) 重新定义成功标准 (不再要求生成端体现主题, 改为 z 空间可分性)
- M3 1.5B 联合训练目标需重新审视

若决策为 B (prefix-tuning), 写 v37b prefix-tuning spec.
若决策为二次 brainstorm, 列出待补做的 ablation.

## 5. 文件清单

- `crystalllm/versions/v37/pipeline/zero_z_eval.py` — 统一 eval 脚本
- `crystalllm/versions/v37/evaluation/gen_samples.py` — 生成质量脚本
- `crystalllm/versions/v37/v37_e2e.json` — 聚合结果
- `crystalllm/versions/v37/v37_decision.md` — 本报告
- `samples_{v25,v36}_{encoded,zero}.json` — 4 套样本

## 6. 下一步 spec

根据本报告决策, 写下一个 spec (v37b / v38 / 战略重定位 / 二次 brainstorm).