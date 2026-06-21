# Wave Function Transformer 实现与对比 — 严格 vs 实用

**生成日期**: 2026-06-21
**承接**: V49 诊断 (CMT-Fixed 崩坏) + 数学分析
**核心结论**: **严格波函数 Transformer 训练困难; 去 norm 约束后改善但仍劣于 baseline. Born rule + 酉变换 + modReLU 联合约束限制学习能力.**

---

## 1. 实现的架构 (Option A)

完整实现严格波函数 Transformer,组件:

| 组件 | 实现 | 物理意义 |
|---|---|---|
| Linear layers | `Cayley transform` U = (I-A)(I+A)^{-1} | 严格 unitary |
| Attention scores | `\|⟨q,k⟩\|²` (Born rule) | 量子测量概率 |
| Activation | `modReLU`: ReLU(\|z\|)·e^(i·phase) | 保留相位的非酉 |
| Readout | `BornRuleHead`: P(v) = \|⟨v\|ψ⟩\|² / Σ | Born rule |
| Normalization | `WaveFunctionNorm`: ‖ψ‖² = 1 | 量子态约束 |

数学验证:
- Cayley: ‖U^H U - I‖_max = 3.6e-7 ✓
- |ψ|² = 1 严格保持 ✓
- 复数 dtype 保留 phase ✓

**代码**: `experiments/v49_pre/wave_transformer.py` (415 lines, 完整可跑)

---

## 2. 4-way 对比 (相同 v28_train 10k subset)

| 模型 | 架构 | val_ppl | Gen Diversity | 状态 |
|---|---|---|---|---|
| CMT-Fixed (30k) | 复数 KAN+WaveAttn+LieRE (magnitude collapse) | 1.0053 | 0.085 | 🔴 Memorizer |
| **Baseline (10k)** | **标准 Transformer** | **2.80** | **0.212** | **🟡 真 LM** |
| Wave Function (strict, 5k) | Cayley+Born+modReLU+‖ψ‖²=1 | **36.30** | - | 🔴 **失败** (无学习) |
| Wave Function (no-norm, 5k) | Cayley+Born+modReLU (无 ‖ψ‖²=1) | **16.36** | - | 🟡 学习中但劣于 baseline |

**关键发现**:
- **严格 norm=1 严重损害学习**: 36 PPL (基本是猜)
- **去 norm 约束后** PPL 16 (有改善但远差于 baseline 2.80)
- **CMT-Fixed 的 "PPL 1.005" 是 memorization 假象**,但仍优于 Wave Function 的真学习 (因为 memorization 把训练集背下来了)
- **Baseline 在 5k-10k 步内达到 PPL 2.80**,远超两个 Wave Function 变体

---

## 3. 数学失败原因分析

### 3.1 总 norm=1 为什么损害学习

Wave function 约束 ‖ψ‖² = 1 是个**强约束**:
- 1 个标量约束(总 norm) + N-1 个自由参数(相位 + 相对 magnitude)
- 对比标准 transformer: N² 个自由参数(线性层)
- 自由度被严重限制

**梯度分析**:
- ∂‖ψ‖²/∂ψ = 2ψ^T (线性)
- ∂WaveFunctionNorm/∂z 在 ‖z‖=1 附近是 well-defined,但远离 1 时梯度被强制拉回,模型无法"放大"重要方向
- Born rule |⟨v|ψ⟩|² ≤ |v|²|ψ|² = 1,无法用 magnitude 区分 token,只能依赖 phase

### 3.2 modReLU 为什么差

modReLU: z' = ReLU(|z| + b) · e^(i·phase)
- 对 |z| ≈ 0 的维度, ReLU(0+b) 是常数 → 失去 phase 信息
- 比 GELU/ReLU 严格更差(在 0 附近)
- 标准 GELU 允许小梯度通过,modReLU 在 b=0 时完全截断

### 3.3 Cayley unitary 为什么过严

Cayley 给出 unitary U, 这是**双射**且保持 norm:
- ∂U/∂A 在 ‖A‖ 较小时是 well-conditioned
- 但 unitarity 是 hard constraint: U^H U = I 必须精确成立
- 标准 linear 没有这个约束,可以学任意映射

**对 next-token 任务**:
- 注意力需要"软选择"多个 key
- 酉 Q, K, V 旋转后,|⟨q,k⟩|² 是固定的,不能 sharp
- 标准 softmax 允许任意尖锐度

### 3.4 Born rule 的局限

Born rule: P(v) = |⟨v|ψ⟩|²
- 这是**双线性**的,不是 softmax
- 缺少 softmax 的"竞争"机制: 如果 vocab 中有 2261 个 token,Born rule 把概率分给所有 |⟨v|ψ⟩² > 0 的
- Softmax 有"exponential competition": 优势 token 的概率指数级压过劣势

**具体例子**:
- Standard softmax: P(token) ∝ exp(z·v_token) — 可以 sharp
- Born rule: P(token) ∝ |⟨v_token|ψ⟩|² — 最多 ‖v‖² |ψ|² ≤ 1,无 sharp 机制

→ Born rule 自然不擅长 next-token 的"one-hot-ish" 分布

---

## 4. Wave Function 的有效用途

虽然 Option A 严格版在 next-token 上失败,**波函数 Transformer 的核心思想在以下任务上仍然有效**:

### 4.1 可能有效的领域
- **物理模拟** (真正的 Schrödinger 方程演化) — 物理上波函数就是 norm=1
- **量子化学** (分子基态) — 实际波函数是 ‖ψ‖=1
- **密度估计** (continuous) — Born rule 是自然的概率 measure
- **干涉敏感任务** (如相位恢复, 光学) — 需要 phase preservation

### 4.2 不适合 next-token
- Next-token 是**离散的、尖锐的**分布 (1 of V)
- Born rule 的 |⟨v|ψ⟩|² 在 V=2261 时太"均匀"
- Softmax 的指数机制更适合
- 字符级 vocab=2261 + PPL≈1 的 next-token,**目标分布几乎是 one-hot**,Born rule 极难匹配

---

## 5. 实用建议 (Option B / C 风格)

如果要"波函数 inspired" 但实用:

### 5.1 Option B: Born rule attention + 实用 FFN

```python
class PracticalWaveBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4):
        # 复数 attention (Born rule scores) ✓ 严格
        self.attn = WaveFunctionAttention(dim, n_heads)
        # 标准 GELU FFN (complex linear) — 实用
        self.ffn = StandardComplexFFN(dim, dim * mlp_ratio)
        # 标准 LayerNorm — 实用
        self.ln1 = ComplexLayerNorm(dim)
        self.ln2 = ComplexLayerNorm(dim)
    
    def forward(self, psi):
        psi = psi + self.attn(self.ln1(psi))
        psi = psi + self.ffn(self.ln2(psi))
        return psi  # 无 norm 约束
```

### 5.2 Option C: 简化为 RoPE + 标准 attention

实际上, **RoPE 已经是相位编码**:
- RoPE 在相邻 (x_even, x_odd) 上做 2D 旋转 = 复数乘法
- 标准 attention softmax 配合 RoPE 已经有 phase 敏感性
- 不需要 explicit complex

→ 标准 Transformer + RoPE **已经是隐式波函数 Transformer**

---

## 6. 实验对比总结

| 模型 | 物理真实性 | 训练性 | next-token 表现 |
|---|---|---|---|
| 严格波函数 (Option A) | ⭐⭐⭐⭐⭐ | 🔴 差 | PPL 36 (失败) |
| 实用波函数 (Option B) | ⭐⭐⭐ | 🟡 中 | 需测试 |
| RoPE + 标准 (Option C) | ⭐⭐ | 🟢 好 | **baseline PPL 2.80** ✓ |
| CMT-Fixed (复数 magnitude) | ⭐ (错) | 🟢 好但 memorizer | PPL 1.005 (假象) |

**结论**: **波函数 Transformer 的核心思想 (phase preservation) 已经在 RoPE 中实现**. 显式 complex + Born rule + unitary 联合约束对 next-token 任务过严. 实用选择: **标准 Transformer + RoPE** (Option C).

---

## 7. 产出文件

- 严格实现: `experiments/v49_pre/wave_transformer.py` (415 lines)
  - `WaveFunctionTransformer` (24.7M params for d=384, n_layers=6)
  - `UnitaryLinear` (Cayley transform)
  - `WaveFunctionAttention` (Born rule)
  - `WaveFunctionFFN` (modReLU)
  - `BornRuleHead`
  - `WaveFunctionNorm` (total ‖ψ‖² = 1)
  - `ModReLU`
- 训练脚本: `experiments/v49_pre/train_wave.py`
- 训练结果:
  - `experiments/v49_pre/results/_wave_5k.json` (strict, PPL 36)
  - `experiments/v49_pre/results/_wave_no_norm_5k.json` (no norm, PPL 16)
- 关键测试:
  - Cayley unitarity: ‖U^H U - I‖_max = 3.6e-7 ✓
  - Total norm=1 严格保持 ✓
  - 完整 forward + backward + generation ✓

---

## 8. V49 后续路径 (更新)

| 路径 | 状态 | 行动 |
|---|---|---|
| CMT-Fixed (复数 magnitude collapse) | ❌ 不可用 | 跳过, 已在诊断中确认崩坏 |
| Baseline (RoPE + 标准) | ✅ 真 LM, PPL 2.80 | **V49 = baseline 50M + 8-bit AdamW + 30k step** |
| 严格波函数 (Cayley + Born + norm) | ❌ PPL 36 | 跳过 (不实用 for next-token) |
| 实用波函数 (Born attention + 标准 FFN) | ⏳ 未测试 | 需实验验证 |
| RoPE + 标准 | ✅ baseline 路径 | 维持 |

**最强建议**: V49 维持 baseline 路径 (PPL 2.80, 真 LM), 不上 CMT 也不上严格波函数.

---

**生成时间**: 2026-06-21
**下次更新**: 如有 Option B 实验结果
