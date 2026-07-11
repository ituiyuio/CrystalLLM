# v51 Phase 4 草案: 文本-波空间映射 (Text ↔ Wave Mapping)

**日期**: 2026-07-11
**状态**: 设计草案, 待 Phase 1-3 完成后启动
**优先级**: 远期 (v51 Phase 4)

---

## 1. 核心问题

**22 轮 CMT 失败的是某种特定映射**（字符 → 复数相位编码 → MSE next-token）,
不是"波推理 + 文本"的所有可能映射都失败.

**问题**: 文本能否映射到 CWF 的优势区间（连续场 + PDE 演化 + 复数相位）?
答案取决于**映射保留什么结构**.

---

## 2. 三种合法映射

### 映射 A: 频域表示 (Frequency Domain Encoding)

**核心思想**: 字节流做 FFT / STFT, 在频域做预测.

```
字节流 (raw bytes) ──FFT/STFT──> 频域向量 ──CWF──> 预测下一段频域 ──IFFT──> 字节
```

**保留结构**:
- 字节流的统计频率分布（字符级 n-gram 频率的连续表示）
- 短程相关（STFT 窗口内的局部频率模式）
- 长程周期（语言韵律 / 重复模式）

**CWF closure 用处**:
- 频域向量本来就是复数 → CWF 直接吃
- 球面约束 → 频域能量守恒（Parseval）

**对应任务**:
- byte-level next-byte 预测
- char-level next-char 预测
- 句子级频域填充（text infilling）

**falsifiable 阈值**:
- 在 v28 char-level 数据上, CWF-频域 vs V49 baseline 的 PPL 比
- 比值 < 0.5x → 频域 + CWF 真有戏, 进入 Phase 4 实测
- 比值 > 1.0x → 频域化也没救, 归档

**风险**:
- 频域丢失 token 边界信息
- FFT 假设全局平稳性, 文本非平稳

---

### 映射 B: 波形编码 (Waveform Modulation)

**核心思想**: 每个字符映射为短正弦脉冲（OOK / FSK / PSK 调制）, 文本序列
变成一段连续波形, CWF 学这个波形传播.

```
字符序列 ──调制──> 连续波形 ──CWF──> 预测波形 ──解调──> 字符序列
```

**保留结构**:
- 字符边界（脉冲位置）
- 字符身份（频率/相位）
- 局部顺序（脉冲间距）

**CWF closure 用处**:
- 波形是实数 → 需要映射到复数场（Hilbert 变换 / 解析信号）
- 球面约束 → 波形振幅归一化

**对应任务**:
- 字符级序列预测
- 句子级波形重建
- 多字符并行传输

**falsifiable 阈值**:
- 字符重建准确率 > 90% → 调制方案可行
- 调制方案能在长序列 (>1k) 保持稳定 → CWF 学到了长程传播
- 否则 → 调制方案失败, 归档

**风险**:
- 退化为"学个码本", CWF 优势不明显
- 字符↔波形映射的物理意义弱

---

### 映射 C: Embedding 频域化 (Embedding Spectralization)

**核心思想**: token embedding (d=512 实数) reshape 成 1D 信号, 做 FFT,
在频域做 attention/演化, 类似 FNet 但用 CWF.

```
token embeddings (B, S, d) ──reshape──> (B, S, d) 信号 ──FFT──> 频域 ──CWF──> 预测 ──IFFT──> embedding
```

**保留结构**:
- token 边界（time 维度保留）
- embedding 内部结构（频域维度）
- 多尺度模式（embedding 不同频率分量）

**CWF closure 用处**:
- 频域复数场 → CWF 直接吃
- 球面约束 → embedding 范数稳定

**对应任务**:
- LM next-token 预测（标准）
- 任何 Transformer encoder 替换场景

**falsifiable 阈值**:
- 在 v28 char-level 数据上, CWF-Embedding-频域 vs V49 baseline 的 PPL 比
- 比值 < 0.8x → embedding 频域化有效
- 比值 > 1.0x → 频域化引入噪声, 归档

**风险**:
- FNet (Google 2020) 已尝试过纯 FFT 替换 attention, 效果不如 Transformer
- CWF 比 FNet 强的关键在 closure 约束, 但频域范数归一化可能不需要 closure
- **可能只是重新发明 FNet 但更复杂**

---

## 3. 三种映射对比

| 维度 | A: 频域表示 | B: 波形编码 | C: Embedding 频域化 |
|------|-------------|--------------|---------------------|
| 输入域 | 字节流 | 字符 → 脉冲 | token embedding |
| CWF 吃的数据 | 复数频域向量 | 实数波形（需 Hilbert） | 复数频域向量 |
| 保留结构 | 频率分布 | 调制位置 | embedding 内部 |
| 失败风险 | token 边界丢失 | 退化为码本 | 重新发明 FNet |
| 与 22 轮 CMT 区别 | 完全不同 | 完全不同 | 类似但更结构化 |
| 推荐度 | ⭐⭐⭐ (最干净) | ⭐⭐ (可能太巧妙) | ⭐ (已有 FNet) |

---

## 4. 与 22 轮 CMT 失败的关键区别

| 失败原因 | 22 轮 CMT | Phase 4 映射 |
|----------|-----------|--------------|
| 字符 → 复数映射 | 直接（无物理对应） | 频域/波形（有物理意义）|
| 损失函数 | MSE next-token（错配） | 待定（看任务）|
| 训练数据 | char-level (v28) | 待定（看任务）|
| 任务 | next-token | 频域/波形预测 |

**关键修正**: 22 轮 CMT 的失败不能阻止 Phase 4, 因为映射方式根本不同.

但**也不能保证 Phase 4 一定赢**——三种映射各自有不同的失败模式.

---

## 5. Phase 4 路线图 (远期)

### Phase 4.1: 映射 A smoke (推荐先做)
- 字节流 → STFT → CWF → 频域预测 → IFFT → 字节
- 在 v28 上跑 1k 步 / 5 seeds, 对比 V49 baseline
- **预期时间**: 2h CPU
- **falsifiable**: PPL 比 < 0.5x → 进 Phase 4.2; > 1.0x → 归档

### Phase 4.2: 映射 B smoke (如果 A 没赢)
- 字符 → FSK 调制 → 波形 → CWF → 波形预测 → 解调 → 字符
- 在 v28 上跑 1k 步
- **预期时间**: 3h CPU
- **falsifiable**: 字符重建 > 90% → 进 4.3; < 50% → 归档

### Phase 4.3: 映射 C smoke (A 或 B 赢才做)
- token embedding → FFT → CWF → embedding 预测
- **预期时间**: 4h CPU
- **falsifiable**: vs V49 baseline PPL < 0.8x → 真融合

### Phase 4 决策点:
- 至少一个映射赢 → 写"文本-波推理"论文
- 全输 → 把"文本 + 波推理"加入 CWF 红线（不适合）

---

## 6. 风险与对冲

### 6.1 重蹈 22 轮覆辙
- 即使映射方式不同, 仍可能在某些细节上重蹈覆辙
- **对冲**: 每个 Phase 4 实验前, 写明"与 22 轮 CMT 的区别", 防止循环

### 6.2 FNet 历史包袱
- 映射 C 接近 FNet, 而 FNet 已证明不如 Transformer
- **对冲**: Phase 4.3 必须包含 FNet baseline 对照, 不能跳过

### 6.3 v50 干扰
- v50 = V49 + Soft-Exp, 主线推进中
- Phase 4 不能抢 v50 资源
- **对冲**: Phase 4 全部 CPU 可跑, v50 GPU 不动

### 6.4 跨任务泛化失败
- Phase 4.1-4.3 都基于 v28 char-level, 可能不泛化
- **对冲**: 任一映射赢后, 必须扩到 v23 BPE 数据集验证

---

## 7. 与 v51 其他 Phase 的关系

| Phase | 任务 | 状态 |
|-------|------|------|
| Phase 1 (1D PDE) | CWF 真翻盘 (163x) | 已验证 |
| Phase 2 (2D PDE + Schrödinger) | 维度扩展 | 待启动 |
| Phase 3 (音频) | STFT 主场 | 远期 |
| **Phase 4 (文本-波映射)** | **本文档** | **远期空槽** |

**Phase 4 必须在 Phase 1 完成且 CWF 仍在赢才启动**.

---

## 8. 决策记录

| 日期 | 决策 | 依据 |
|------|------|------|
| 2026-07-11 | "文本无法映射到 CWF 优势区间"是错的 | 用户的质疑 |
| 2026-07-11 | 22 轮 CMT 失败的是特定映射, 不是所有映射 | 复盘失败原因 |
| 2026-07-11 | Phase 4 = 文本-波映射, 三种合法映射 (频域/波形/embedding) | 本文档 |
| 2026-07-11 | Phase 4 启动条件: Phase 1-3 完成且 CWF 仍在赢 | Ponytail 一次一实验 |

---

## 9. 待办与未决

- [ ] Phase 1: 1D PDE 全参数扫描 (exp33-35) — 先做
- [ ] Phase 2: 2D PDE + Schrödinger (exp36-38) — Phase 1 后
- [ ] Phase 3: 音频合成 (exp39-40) — Phase 2 后
- [ ] **Phase 4.1: 映射 A smoke (频域表示) — Phase 3 后**
- [ ] **Phase 4.2: 映射 B smoke (波形编码) — Phase 4.1 后**
- [ ] **Phase 4.3: 映射 C smoke (Embedding 频域化) — Phase 4.1 或 4.2 赢后**
- [ ] **未决**: 映射 A/B/C 哪个最值得做?
- [ ] **未决**: Phase 4 失败的话, 文本 + CWF 红线如何写?

---

**附录**:
- 相关: `research/cwf/docs/v51_roadmap.md` (主章程)
- 相关: `research/cwf/experiments/exp25-30` (CMT 失败历史)
- 相关: `research/cwf/experiments/exp31_pde_smoke/` (Phase 1 起点)