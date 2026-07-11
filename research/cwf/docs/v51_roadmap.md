# CWF v51 路线章程: PDE 战场重启

**日期**: 2026-07-11
**状态**: 草案, 待 v50 不受影响前提下推进
**作者**: Claude (MiniMax-M3) + 用户

---

## 执行摘要 (TL;DR)

CWF 在 **1D 波动方程 + Gaussian packet 短期预测** 任务上, 经 multi-seed 验证,
**碾压实数 Transformer baseline 163x (median ratio 0.0061x, 5/5 seeds 全部重现)**.

这是 CWF 自 v50 关闭 CMT 训练侧以来第一次结构性的胜利, 标志着
**CWF 在"刚性时间轴 + 周期信号"任务上具备真正的归纳偏置优势**.

**v50 不动** (= V49 baseline + Soft-Exp 推理). v51 是独立研究章程, 平行推进.

---

## 1. 证据链

### Exp 31 (single seed)
- 任务: 1D 波动方程 u_tt = c² u_xx, 初始 Gaussian packet, 预测 t=0.5
- 网络: CWFSingleBlock (d=64, ~326k params) vs 2-layer Transformer (~104k params)
- 训练: 1k 步, batch=32, lr=3e-4, MSE loss
- 结果: CWF+MSE = 0.000126 vs Trans+MSE = 0.023253 → **CWF 0.0054x = 184x 优势**

### Exp 31 副实验: 损失函数对照
- CWF + STFT loss = 0.050863 (404x 比 CWF+MSE 退化)
- Trans + STFT loss = 0.030740 (1.3x 比 Trans+MSE 退化)
- **结论**: STFT 损失对所有模型都是负担, 对 CWF 极度有害 (400x vs 1.3x)
- **推翻假设**: "换损失函数能救 CWF" — **真正救 CWF 的是它自己的归纳偏置, 不是新损失**

### Exp 32 (multi-seed 验证)
- 5 seeds: [42, 123, 2024, 7, 11]
- CWF/Trans ratio: min=0.0049x, median=0.0061x, max=0.0073x, std=0.0009x
- **5/5 seeds 全部 CWF ≪ Trans**
- **新增洞察**: CWF std=0.000020 vs Trans std=0.001083 → **CWF 收敛稳定性也是 Trans 的 54x**

---

## 2. 核心洞察

1. **CWF 的复数 + 球面约束天然匹配 PDE 波传播**
   - 1D 波动方程本质是相位演化, CWF 的"球面相位编码"是 PDE 的天然表示
   - 不需要外部损失函数辅助, MSE 已经够

2. **PDE 是相位敏感任务, 不是相位不变任务**
   - 时间偏移 = 轨迹位移 (Lorenz 同样), MSE 才是对的损失
   - STFT/Spectral 损失给"相位不变性" → 对相位敏感任务是毒药
   - **这条红线扩展**: Lorenz, PDE 短期预测, 任何"时间偏移 = 物理位移"的任务 → 全部用 MSE

3. **CWF 对损失函数匹配度极度敏感**
   - 错配损失让 CWF 退化 400x, Transformer 仅 1.3x
   - 含义: CWF 的归纳偏置很强, 但**要求损失函数与归纳偏置对齐**
   - 实操: CWF 必须配"硬"损失 (MSE/L2), 不能配"软"损失 (STFT/spectral magnitude)

---

## 3. v51 路线图 (Ponytail: 一次一实验)

### Phase 1: 1D PDE 全参数扫描 (1-2 周)
**目标**: 定位 CWF 在 PDE 上的优势区间与边界

| 实验 | 任务 | 假设 | falsifiable 阈值 |
|------|------|------|-----------------|
| **exp33** | T_PRED ∈ {0.1, 0.5, 1.0, 2.0} | CWF 在短期预测强, 长期会退化 (类似 Lorenz) | T_PRED > 1.0 时 CWF ratio > 0.5x → 短期胜利而已 |
| **exp34** | 多频率叠加 (k0 ∈ {5, 20, 50}) | 高频信号 CWF 应该更强 (相位结构复杂) | 高频时 ratio 反而恶化 → 1D 胜利是 lucky data |
| **exp35** | 多 Gaussian 包络 (sigma 扫描) | 窄包络 (= 高频分量) CWF 应该更强 | 窄包络退化 → CWF 优势与频率无关 |

**Phase 1 决策点**:
- 全赢 (3/3 ratio < 0.5x) → 进入 Phase 2
- 任一输 → 归档 1D PDE, 写"短期预测边界"论文

### Phase 2: 高维 PDE + Schrödinger (2-3 周)
**目标**: 验证 CWF 在"相位结构更复杂"的战场仍然胜出

| 实验 | 任务 | 假设 | falsifiable 阈值 |
|------|------|------|-----------------|
| **exp36** | 2D 波动方程 (衍射 + 干涉) | 2D 多方向传播, CWF 球面相位优势放大 | 2D 时 ratio > 1.0x → 1D 胜利是结构巧合 |
| **exp37** | Schrödinger 方程 (自由粒子 + 谐振子) | 复数场天然, closure 约束天然 | Schrödinger ratio ≈ 1.0x → CWF 优势与方程形式无关 |
| **exp38** | Burgers 方程 (激波形成) | 非线性 PDE, 相位结构被破坏 | Burgers 上 ratio > 1.0x → CWF 优势只在线性 PDE |

**Phase 2 决策点**:
- 全赢 (3/3 ratio < 0.5x) → 进入 Phase 3
- 任一输 → 定位"CWF 优势域" (线性 vs 非线性, 短波 vs 长波)

### Phase 3: 周期长序列 + STFT 该出现的战场 (3-4 周)
**目标**: 探索 STFT 损失真的有效的领域, 找 CWF + STFT 的合作场景

| 实验 | 任务 | 假设 | falsifiable 阈值 |
|------|------|------|-----------------|
| **exp39** | 音频合成 (短句语音 STFT 域) | 周期性 + 相位可平移, STFT 主场 | CWF+STFT vs Trans+STFT ratio ≈ 1.0x → 这里两者同水平 |
| **exp40** | 量子态演化 (VQE 简化版) | 概率幅演化, 复数场, closure | CWF ratio < 0.5x 且 STFT 增益 > 0 → STFT 终于有用 |

**Phase 3 决策点**:
- 找到 STFT 真有效的领域 → CWF + STFT 是新组合拳
- 没找到 → STFT 永远不是 CWF 的好搭档 (这条规则写进 CWF 红线)

---

## 4. 风险与反例

### 4.1 Transformer baseline 可能只是步数不够
- exp31/32 都用 1k 步, Transformer 还在震荡 (loss 0.015-0.025)
- 假说: 5k-10k 步 Transformer 也可能收敛
- **对冲**: exp33 改 5k 步对照, 看 Transformer 是否能追平

### 4.2 CWF 在长 T_PRED 可能退化为 memorizer
- 1D 短期预测 CWF 强, 但 Lorenz/PDE 长期演化混沌性会破坏归纳偏置
- **对冲**: exp33 长 T_PRED 是必跑测试

### 4.3 单架构胜利可能不泛化
- exp31/32 只用 CWFSingleBlock (一个 block), 其他 CWF 变体 (multi-block, holomorph) 没测
- **对冲**: Phase 2 用 multi-block CWF + 对照 baseline

### 4.4 v50 干扰
- v50 = V49 + Soft-Exp, 主线推进中
- v51 不能抢 GPU/时间
- **对冲**: v51 全部 CPU 可跑, 不占用 v50 GPU 资源

---

## 5. 与 v50 的关系

**v50 不动**. v50 是产品级 LM 路线 (V49 baseline + Soft-Exp 推理), 与 v51 平行.

| 维度 | v50 | v51 |
|------|-----|-----|
| 目标 | 1.2B LM PPL < 2.36 | CWF 在 PDE 战场结构性胜利 |
| 战场 | char-level / BPE 语言 | 1D/2D PDE, Schrödinger |
| 损失 | Soft-Exp inference only | MSE (Phase 1-2), STFT (Phase 3) |
| 状态 | active, 主线 | 草案, 待 exp33 启动 |

---

## 6. 决策记录 (Decision Log)

| 日期 | 决策 | 依据 |
|------|------|------|
| 2026-07-11 | CWF 在 1D PDE 短期预测真翻盘 (163x, 5/5 seeds) | exp31 + exp32 |
| 2026-07-11 | 推翻"换损失救 CWF" 直觉, 损失函数不是瓶颈 | exp31 STFT 副实验 |
| 2026-07-11 | CWF 必须配"硬"损失 (MSE), 不能配"软"损失 (STFT) | CWF 错配损失退化 400x |
| 2026-07-11 | v51 路线启动, Phase 1 = 1D PDE 全参数扫描 | Ponytail: 一次一实验 |
| 2026-07-11 | v50 不动 | 用户决定 |

---

## 7. 待办与未决

- [ ] exp33: 多 T_PRED 扫描 (1k 步 / 5 seeds × 4 T_PRED, ~3h CPU)
- [ ] exp34: 多频率叠加 (1k 步 / 3 freqs × 5 seeds, ~3h CPU)
- [ ] exp35: 多 sigma 扫描 (1k 步 / 3 sigmas × 5 seeds, ~3h CPU)
- [ ] exp36-38: Phase 2 全部待 Phase 1 完成
- [ ] exp39-40: Phase 3 待 Phase 2 完成
- [ ] **未决**: CWF multi-block 版本是否在 PDE 上比 single-block 更强?
- [ ] **未决**: Transformer 5k-10k 步对照是否追平?

---

**附录**:
- 证据: `research/cwf/experiments/exp31_pde_smoke/results/exp31_results.json`
- 证据: `research/cwf/experiments/exp32_multiseed/results/exp32_results.json`
- 代码: `research/cwf/experiments/exp31_pde_smoke/exp31_pde_smoke.py`
- 代码: `research/cwf/experiments/exp32_multiseed/exp32_multiseed.py`
- CWF 原型: `research/cwf/prototype/cwf_minimal.py::CWFSingleBlock`