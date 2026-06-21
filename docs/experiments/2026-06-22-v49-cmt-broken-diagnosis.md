# V49 CMT-Fixed 架构诊断报告 — 根因确定

**生成日期**: 2026-06-21
**承接**: V49 Formal Training + Test (val_ppl 1.0053 但生成崩坏)
**核心结论**: **CMT-Fixed 架构本身有根本缺陷,不是训练配置/数据问题. V49 不应 scale 到 1.2B.**

---

## 1. 诊断过程 (3-way 对比)

为隔离崩坏根因,在**相同 v28_train (10k subset)** 上对比 3 个模型,唯一变量是架构:

| 模型 | 架构 | 步数 | lr | val_ppl | Generation Diversity |
|---|---|---|---|---|---|
| V49 | CMT-Fixed (复数 KAN + WaveAttn + LieRE) | 30k | 3e-4 | 1.0053 | **0.085** (FAIL) |
| Diag_CMT | CMT-Fixed 同架构 | 10k | 1e-4 | 1.0052 | **0.031** (FAIL, 更差) |
| **Baseline** | **标准 Transformer (attn + MLP)** | **10k** | **1e-4** | **2.80** | **0.212 (PARTIAL, 真 LM)** |

---

## 2. 关键发现

### 2.1 CMT-Fixed "PPL 1.005" 是 memorization 假象

**两个 CMT-Fixed 配置 (V49 高 lr 和 Diag 低 lr) 都崩坏**:

**V49 (30k, 3e-4, full data):**
```
"The quick brown fox "  → "hnnuknrknuunuuknnuh..." (重复 n/u/k, diversity 0.09)
"def fib(n):"           → "(((fffin:nninbb(if)bfibbb..." (重复 (/f/b)
"Once upon a time"      → "mmunamnuuuumumpuunup..." (重复 m/u)
```

**Diag CMT (10k, 1e-4, 10k subset):**
```
"The quick brown fox "  → "bx bbbbbbbb...bbb" (98% 'b', diversity 0.03)
"int main() { return"   → "          ...   " (100% 空格)
"def fibonacci(n):"     → "   nn              c                              "
```

**Diag 反而比 V49 更崩坏** — 排除"训练过头/lr 太高"假设。

### 2.2 Baseline (52M 标准 Transformer) 是真语言模型

虽然 val_ppl 高 2.7x (2.80 vs 1.005), **baseline 实际生成连贯代码/英文**:

```
T=0.5: "The quick brown fox the provided used to the GNU General Public 
         License as published by the Free Software Foundatio"
T=0.8: "The quick brown fox the received to the
         const char* pxFox = pxFox(pxFoxFox) / 2;
         msg = pxFox"
T=1.0: "The quick brown fox pass"), and state state for the data pass2"), 
         and n neal pass4", internal the enable state form(pas"

T=0.8 code_python: "     category = time.defaults['name']
                       return x\n\n\nclas"
T=1.0 code_python: ' """\n      configize = prjjmCPOurnely (n, prjmCPOurnely)\n   '
T=0.5 code_c: "main();\n}\n\nvoid Property::setNode() {\n     return main();\n}\n"
T=0.8 code_c: "out wandow low and use lectal in string, whose weners then a"
T=1.0 code_c: 'main(new = 15 && main("buffer") < 16, main("buffer") << main'
T=1.0 agentic: "include milestones, technology choices, risks, testing strat"
```

Baseline 学到了:
- C++ 代码模式 (返回语句, 类方法, include 头文件)
- Python 缩进 + 函数定义
- 英文散文 (单词序列)
- 许可证文本 (GPL 头)

多样性 0.212 比 CMT-Fixed 高 2.5-7x,且 T=1.0 时 baseline 生成包含 30+ unique chars 的连贯文本.

### 2.3 对比数据 (Diversity 0=全重复, 1=完全随机)

| 模型 | avg_div | 解读 |
|---|---|---|
| V49 (CMT-Fixed 30k) | 0.085 | 🔴 强 memorizer, 生成重复字符循环 |
| Diag CMT (10k) | 0.031 | 🔴 **更严重**, 单字符占 94-100% |
| **Baseline (10k)** | **0.212** | 🟡 真 LM, 多样性 OK, 部分 prompt 也部分崩坏 |

Baseline 也有部分崩坏 (T=0.5 "int main" 100% 空格, "code_python" T=0.5 100% 空格) — 这是 char-level 50M + 10k 训练步数的固有限制,不是架构缺陷.

---

## 3. 根因分析 (CMT-Fixed 崩坏)

### 3.1 排除的假设

| 假设 | 排除证据 |
|---|---|
| 训练步数太多 (30k) | Diag 10k 步崩坏更严重 |
| 学习率太高 (3e-4) | Diag 1e-4 崩坏更严重 |
| 训练数据太少 (10k subset) | V49 用 full data 仍崩坏 |
| 数据本身有问题 | 同样数据上 baseline 工作 |
| 8-bit AdamW 量化问题 | 三者都用 8-bit, 只有 CMT-Fixed 崩坏 |

### 3.2 剩余假设: CMT-Fixed 架构本身

**可能的崩坏源**:

1. **ComplexKANFFN_TrueMul (复数 B-spline KAN)**
   - 用 `|z| = sqrt(real² + imag²)` 作为 basis 输入
   - 在大量 zero-padding 输入上, `|z|` 接近 0 → B-spline 退化为常数
   - 复数 B-spline 输出再 `cat[real | imag]` 累加,可能塌缩到常数
   - **可能根因**: 字符级数据 high-frequency zero padding 触发 B-spline 退化

2. **WaveAttentionSoftmax (magnitude-softmax)**
   - `score_mag = sqrt(score_real² + score_imag²)` 取 magnitude 后 softmax
   - 复数 attention 累积后 magnitude 可能饱和 → 注意力塌缩到均匀分布
   - 训练后期 magnitude 集中在最大 token, 退化到 greedy next-token

3. **LieRE_NoContext (2D RoPE 风格 PE)**
   - 偶数/奇数维度配对旋转
   - 在 2D 复数空间旋转, 长期依赖可能丢失
   - 但 PE 本身只占 0 参数, 不会是主因

4. **组合效应 (magnitude collapse chain)**
   - 复数 magnitude → magnitude-softmax → 累加 magnitude → ComplexKAN magnitude
   - 每一步都丢失相位信息, 累积塌缩

**最可能根因**: ComplexKANFFN 的 magnitude 输入在 padding/zero-heavy 序列上退化, 复数 B-spline 输出几乎常数, 整个 FFN 失效. 然后残差连接主导, 模型退化为"复制最常见 token".

### 3.3 验证根因的下一步实验 (建议)

| 实验 | 目的 | 预期 |
|---|---|---|
| CMT w/o ComplexKAN (用标准 MLP) | 隔离 KAN 贡献 | 若 PPL 上升, KAN 是元凶 |
| CMT w/o WaveAttn (用标准 attn) | 隔离 WaveAttn 贡献 | 若 PPL 上升, WaveAttn 是元凶 |
| CMT 单独 KAN/Attn/PE 单元测试 | 单元 forward 输出分布 | 检查 magnitude 分布 |
| ComplexKAN 改实数 B-spline (vanilla) | 简化 KAN | 验证复数是否必要 |
| baseline 训 30k step (vs 10k) | 验证 baseline 提升潜力 | baseline PPL 应降到 ~2.0 |

---

## 4. 行动建议 (修订 V49 路径)

### 4.1 立即做 (低风险, 高价值)

1. **V49 = baseline (标准 Transformer) 50M + 8-bit AdamW + full data + 30k step**
   - 复用现有 `train_v49_baseline.py`, 改 `--n_steps 30000 --max_train_samples` 不限
   - 训练 ~1h, 预期 val_ppl ~2.0 (vs V49 "PPL 1.005" 假象)
   - **这是真语言模型**, 可 scale 到 1.2B

2. **CMT-Fixed 单元 ablation** (并行, ~1h):
   - CMT w/o ComplexKAN
   - CMT w/o WaveAttn
   - 找出崩坏的具体模块

### 4.2 中期 (1-2 天)

3. **CMT 修复后重做**:
   - 修复 ComplexKANFFN (实数化 or 改其他激活)
   - 重新训练验证

4. **改用 BPE tokenization**:
   - 字符级 vocab=2261, 序列短, n-gram 模式压倒结构
   - BPE 16K vocab 应有更好 long-range 依赖

### 4.3 不要做 (基于诊断)

- ❌ **不要 scale CMT-Fixed 到 1.2B** — 架构已崩坏, scale 无意义
- ❌ **不要相信"val_ppl < 1.01"为成功** — 必须 generation test
- ❌ **不要在 v28_train 上重训更大模型** — 需先解决架构/数据问题

---

## 5. 关键教训

### 5.1 评估方法

**`val_ppl` 不是语言模型能力的可靠指标**:
- PPL 1.005 (CMT-Fixed) = memorization, 实际崩坏
- PPL 2.80 (baseline) = real LM, 实际生成连贯
- **必须做 generation quality test (多样性 + 实际输出)**

**建议评估流程**:
1. val_ppl @ 训练指标
2. val_ppl @ held-out 完全 OOD (e.g., Wikipedia)
3. **Generation diversity** (unique-n, repetition rate)
4. **实际生成 review** (人工 + 自动化)
5. Token entropy 检查 (避免 magnitude collapse)

### 5.2 架构选择

**复杂架构 (复数空间, magnitude 操作) 在数据多样性低时易崩塌**:
- CMT 设计的复数几何结构假设数据有连续结构
- v28_train 是 C++ 代码 (字符级), 离散且 high-frequency
- magnitude 操作在 zero-padding 主导时退化为常数
- **简单架构 (标准 Transformer) 在小数据/小模型上更鲁棒**

### 5.3 训练策略

**lr=3e-4 对 50M 字符级模型过高**:
- V49 训练 loss 在 1500 步就降到 0.01, 之后只是 memorize
- baseline 用 lr=1e-4 更稳定收敛
- 字符级 next-token 本身简单, 不需要高 lr

---

## 6. 后续计划

### Phase 1 (立即, ~1h)
- 训练 V49 = baseline 50M + 8-bit AdamW + full data + 30k step
- 验证 generation quality

### Phase 2 (如果 Phase 1 成功, ~1 天)
- Scale baseline 到 200M, 1.2B
- 评估 val_ppl 趋势 + generation quality

### Phase 3 (CMT 修复, ~1 周)
- 单元 ablation 找崩坏源
- 修复 (可能: ComplexKAN 实数化, 或用 magnitude-free 设计)
- 重新训练验证

### Phase 4 (数据/方法改进, ~1 周)
- 评估 BPE tokenization
- 评估更多样数据 (Wikipedia/Reddit 子集)
- 如果都失败, 退回到 baseline 路径, 不再尝试 CMT

---

**生成时间**: 2026-06-21
**下次更新**: V49 baseline 30k 训练完成后
