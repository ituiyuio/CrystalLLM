# CWF × 文本脉冲波 (FSK Text-to-Wave) Smoke 实验

**Spec**: `docs/superpowers/specs/2026-07-15-脉冲波-text-wave-design.md`
**Branch**: `cwf-manifesto` (new experiment lives under `research/cwf/experiments/exp33_fsk_text_smoke/`)
**Date**: 2026-07-15
**Status**: Draft (pending user review)

---

## 1. 背景

### 1.1 双重锚点

**CWF Phase 1 PDE 翻盘** (exp31/32, 2026-07-11):
- 1D 波动方程 + Gaussian packet + MSE 短期预测
- CWF vs Transformer ratio = 0.0061x (163x 优势), 5/5 seeds 重现
- 推翻"换损失救 CWF" 直觉, 证明 CWF 归纳偏置天然匹配 PDE 波传播

**v50 LM 路线** (locked, 2026-06-23):
- V49 baseline 1.2B + Soft-Exp 推理 (+48.6%)
- 22 轮 CMT 失败 (char-level + 复数 + next-token 错配)
- v51 章程独立推进, v50 不动

**未决问题**: CWF 的 PDE 优势能否迁移到文本?

### 1.2 关键洞察

22 轮 CMT 失败的根因是"**离散 token ↔ 复数场**"的错配 — char-level next-token 既不是相位敏感任务, 也不连续可微。v51 Phase 4.2 提出: **只要文本能映射到 CWF 的优势域 (连续波 + MSE 短期预测)**, 22 轮失败不能阻止新映射的成功。

**核心设计**: 字符 → FSK 调制 → 连续复数波 → CWF 传播 → 解调回字符。**物理结构上** 文本与 PDE 任务同构, 复用 exp31/32 几乎所有部件。

### 1.3 与 v51 Phase 4.2 映射 B 的关系

本文档即 v51 Phase 4.2 的最小 smoke 实现。完整 roadmap 见 `research/cwf/docs/v51_phase4_text_wave.md` §2 映射 B + `research/cwf/docs/v51_roadmap.md` §5 Phase 4.2。

**roadmap 顺序**: Phase 1 (1D PDE 全参数扫描) → Phase 2 (2D PDE) → Phase 3 (音频) → Phase 4.1 (频域) → **Phase 4.2 (本文档)**。

**本次 spec 偏离 roadmap**: 用户在 2026-07-15 决定 "从脉冲波方向再做一次努力", 选择跳过 Phase 1-3 + Phase 4.1, 直接打 Phase 4.2 smoke。理由: 物理结构匹配度高, 复用 exp31/32 几乎全部代码, 失败成本 ≤ 1 天 CPU。

---

## 2. 目标

**Primary (GO)**: 证明 CWF 在 "FSK 调制文本波 + 短时 shift-by-1 预测" 任务上, 字符重建准确率 > 90% (来自 v51_phase4 §2 映射 B 阈值)。

**Secondary (PASS)**: CWF + MSE 字符重建 > Transformer + MSE 字符重建, 且比值 < 0.5x (CWF 至少 2x 优势)。

**Non-goals**:
- 长序列依赖 (smoke 用 32 char 短段, 不测 1k+ 字符)
- 完整 v28 (256) 词汇表 (smoke 用 top-32 ASCII 子集)
- 真实 LM 部署 (PPL < 2.36 之类, v50 范畴)
- 替换 v50 Soft-Exp 推理

**Falsifiable 阈值**:
- char accuracy > 90% → **GO**, 写 Phase 4.2 完整版, 扩 vocab 到 64/128/256
- char accuracy 50%-90% → **PARTIAL**, 看 CWF/Trans ratio; 优势 ≥ 2x → 进 4.3, 否则归档
- char accuracy < 50% → **DEAD**, 归档 "CWF + 文本脉冲波" 路线, 写失败 postmortem

---

## 3. 设计

### 3.1 端到端数据流

```
text[0:8] (8 chars from v28 top-8 ASCII subset)
    ↓
encode: ψ[c*T_CHAR + n] = exp(i · 2π · char_id[c] · n / T_CHAR)  for n in [0, 8)  # FSK baseband
    ↓
complex waveform: ψ_in  shape (B, S=64), complex
    ↓
CWF block: ψ_out = CWFSingleBlock(ψ_in)        # 复用 exp31 (d=S=64, skip projection)
    ↓
decode: char_id[c] = argmax_k |IFFT(ψ_out[c*T_CHAR:(c+1)*T_CHAR])|  # per-slot freq bin (size 8)
    ↓
predicted text[1:9] (8 chars)
```

### 3.2 关键参数

| 参数 | 值 | 理由 |
|------|----|------|
| `VOCAB_SIZE` | 8 (top-8 ASCII 频率) | **Nyquist 限制**: T_CHAR=8 → 8 个可分辨 freq bins; 覆盖英文 ~50% (空+e+t+a+o+i+n+s) |
| `S` (CWF 网格) | 64 | 与 exp31/32 完全一致, 复用架构 (d=S=64, skip projection) |
| `N_CHARS` | 8 字符/段 | S / T_char = 64 / 8 = 8 |
| `T_CHAR` | 8 网格点/字符 | T_CHAR ≥ VOCAB_SIZE 满足 Nyquist; 1 周期/字符 = 1 cycle in 8 points |
| `FSK_TYPE` | baseband 复数 FSK | 无需 Hilbert 变换; 解析信号天然; 实部/虚部 = cos/sin |
| `TEXT_SHIFT` | 1 字符 (shift-by-1) | 与 next-token 预测同构, 简单清晰 |
| `V28_SUBSET` | top-8 ASCII 频率 (空 e t a o i n s) | v28 字符合法子集; char 频次 rank 来自标准英文统计 |

**Nyquist 约束推导**:
- per-slot 信号长度 = T_CHAR
- per-slot FFT 频率分辨率 = T_CHAR (complex Nyquist)
- 唯一可区分的 char 数 ≤ T_CHAR
- 因此 VOCAB_SIZE ≤ T_CHAR
- 本 spec: VOCAB=8, T_CHAR=8 (贴边, ponytail)

### 3.3 FSK 编码细节

```python
def fsk_encode(char_ids: LongTensor[B, N_CHARS]) -> ComplexTensor[B, S]:
    """
    char_ids: (B, 8) char indices in [0, 8)
    output: (B, 64) complex waveform
    """
    B, N = char_ids.shape
    assert N == N_CHARS == 8
    waveform = torch.zeros(B, S, dtype=torch.complex64)
    for c in range(N_CHARS):
        freq_bin = char_ids[:, c].float()  # (B,) in [0, 8)
        # 在 slot c 内 (8 points) 放 1 周期, 频率 = freq_bin / T_CHAR
        # 周期长度 T_CHAR = 8, 故相位 = 2π · freq_bin · n/8 for n in [0, 8)
        n = torch.arange(T_CHAR, dtype=torch.float32)  # (8,)
        phase = 2 * pi * freq_bin.unsqueeze(-1) * n.unsqueeze(0) / T_CHAR  # (B, 8)
        waveform[:, c*T_CHAR:(c+1)*T_CHAR] = torch.complex(
            torch.cos(phase), torch.sin(phase)
        )
    return waveform
```

**等价理解**: 整个波形长度 S=64 点。per-slot (长度 8) 的 IFFT 后, 在 freq bin = char_id[c] 处有冲激峰值。Decoder 用 per-slot IFFT (长度 8) 直接读出 char_id。

### 3.4 CWF 架构 (零改动)

复用 `research/cwf/prototype/cwf_minimal.py::CWFSingleBlock`:
- 输入: complex tensor (B, S, 2) (real/imag 两通道)
- 内部: complex linear + Lie group rotation + modReLU (closure-preserving)
- 输出: complex tensor (B, S, 2)

**关键**: exp31 的 `CWFPredictor` 用 `Linear(S, d) + Linear(d, S)` 包裹, 把 (B, S) 复信号映射到 (B, d=64) latent 再回来。我们直接用 CWFSingleBlock (d=64) 处理 (B, S=64) 复信号, 跳过 input/output projection (S == d, identity mapping)。

```python
class CWFFSKPredictor(nn.Module):
    def __init__(self, d: int = 64):
        super().__init__()
        self.d = d  # = S, no projection needed
        self.block = CWFSingleBlock(d=d, hidden_mult=2)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """
        psi: (B, S=64) complex
        returns: (B, S=64) complex
        """
        # (B, S) complex → (B, 1, S, 2) for CWFSingleBlock API
        psi_ri = torch.stack([psi.real, psi.imag], dim=-1)  # (B, S, 2)
        psi_ri = psi_ri.unsqueeze(1)  # (B, 1, S, 2)
        psi_out_ri, _ = self.block(psi_ri)
        psi_out_ri = psi_out_ri.squeeze(1)  # (B, S, 2)
        return torch.complex(psi_out_ri[..., 0], psi_out_ri[..., 1])
```

**Transformer baseline** (复用 exp31 `TransformerPredictor`, S=64):
- 1D Transformer encoder, 2 层, 4 head, d=64
- 输入: real/imag 拼接成 (B, S, 2) real tensor
- 输出: (B, S, 2) real tensor → complex

### 3.5 训练

**复用 exp31 几乎全部**:
- 优化器: Adam, lr=3e-4
- 训练步数: 1000 (smoke)
- Batch size: 32
- 损失: `F.mse_loss(psi_pred.real, psi_target.real) + F.mse_loss(psi_pred.imag, psi_target.imag)`
- Seeds: [42, 123, 2024, 7, 11] (与 exp32 一致, 5 seeds)

**数据生成** (smoke, 不读 v28 文件):
- 训练: 随机 8-char 序列, char_id ~ Uniform(0, 8)
- 测试: 1000 个独立 8-char 序列 (与训练 disjoint)
- 目标: target = text shifted by 1 (text[1:9])

**Ponytail 优化**:
- 单 seed 1k 步预计 ~30s CPU (S=64 同 exp31, FSK 编码更便宜)
- 5 seeds × 2 模型 = 10 跑, 总计 ~5 min CPU
- 全程不读 v28, 数据在内存生成, **无 IO 瓶颈**

### 3.6 评估

**Primary metric**: char reconstruction accuracy
- per-slot IFFT (size T_CHAR=8): `char_id_pred[c] = argmax_{k ∈ [0, 8)} |IFFT(ψ_pred[c*8:(c+1)*8])|[k]`
- accuracy = mean(predicted == target) over all 8 chars and 1000 test samples

**Secondary metrics**:
- waveform MSE (与 exp31 一致, real+imag 各算)
- CWF/Trans accuracy ratio (CWF 优势量化)

**判决逻辑** (`verdict`):
```python
if cwf_acc >= 0.90:
    verdict = "GO - CWF 文本脉冲波可行, 进 Phase 4.3"
elif cwf_acc >= 0.50 and cwf_acc / trans_acc < 0.5:
    verdict = "PARTIAL - CWF 优势够, 进 Phase 4.3 (扩展 vocab)"
elif cwf_acc >= 0.50:
    verdict = "NEUTRAL - 重建够, 但优势不足, 归档"
else:
    verdict = "DEAD - 重建都不到 50%, 路线归档"
```

---

## 4. 文件结构

```
research/cwf/experiments/exp33_fsk_text_smoke/
├── exp33_fsk_text_smoke.py     # 主实验 (~250 行)
├── results/
│   └── exp33_results.json      # 自动保存
└── README.md                   # 1 段说明: 与 exp31/32 区别
```

**复用文件** (不复制):
- `research/cwf/prototype/cwf_minimal.py::CWFSingleBlock`
- exp31 `CWFPredictor` 的接口模式 (S==d, skip projection)
- exp32 的 multi-seed 循环 + ratio 计算

**ponytail 注释**:
- `ponytail: FSK 编码是 deterministic, 无学习参数, 隔离在 utility 函数`
- `ponytail: 数据在内存生成, 1k 步 5 seeds 总耗时 < 10 min CPU`
- `ponytail: 不动 v50 任何文件, 不读 v28 原始数据`

---

## 5. 风险与对冲

| 风险 | 概率 | 对冲 |
|------|------|------|
| 32 char shift-by-1 太简单 (无长程依赖) | 高 | smoke 通过 ≠ 真实 LM; 后续 Phase 4.3 必须扩到 256 char 长段 |
| FSK baseband 在 CWF 输出后失真 | 中 | exp31 CWF 输出在 norm < 0.1; IFFT 可能不解析; 加 amplitude rescale (×10) |
| top-32 ASCII 子集 ≠ v28 完整 (256) | 高 | smoke 验证后扩 vocab 到 64 (S=128) / 128 (S=256) 测泛化 |
| Transformer baseline 5k 步追平 CWF | 中 (exp31 已观察) | smoke 限制 1k 步, ratio 阈值改用 median across seeds |
| S=64 远小于真实 LM (1k+ 字符) | 高 | 短期目标, 不在 smoke 范围; Phase 4.3 处理 |

---

## 6. 决策记录 (Decision Log)

| 日期 | 决策 | 依据 |
|------|------|------|
| 2026-07-15 | 跳过 Phase 1-3 + 4.1, 直接打 Phase 4.2 smoke | 用户决定: "从脉冲波方向再做一次努力" |
| 2026-07-15 | 调制方案 = FSK | 用户从 OOK/FSK/PSK 中选 FSK (推荐) |
| 2026-07-15 | 字符↔网格 = 整段文本 = 1 CWF 输入 | 用户从 3 选项中选推荐 (复用 exp31) |
| 2026-07-15 | 词汇表 = char-level v28 (smoke 缩到 top-8 ASCII) | 用户从 char/BPE 中选 char (推荐) |
| 2026-07-15 | 损失 = 纯波形 MSE | 用户从 3 选项中选推荐 (与 v51 "CWF 配硬损失" 规则一致) |
| 2026-07-15 | 实操参数 (初版) = vocab=32, T_char=2, S=64 | 用户全照推荐 |
| 2026-07-15 | **实操参数修正** = vocab=8, T_char=8, S=64 (Nyquist 约束) | spec 自审发现 T_char<VOCAB 物理上不可能, ponytail 选最简工作配置 |
| 2026-07-15 | Phase 4.2 优先于 Phase 1 PDE 全参数扫描 | 用户意图 (本次 spec 偏离 v51_roadmap 顺序) |

---

## 7. 验证清单 (Definition of Done)

- [ ] `exp33_fsk_text_smoke.py` 跑通 5 seeds × 2 模型 (CWF + Trans) < 10 min CPU
- [ ] `exp33_results.json` 包含 per-seed accuracy, ratio, verdict
- [ ] CWF median char accuracy > 0.50 (起码不像纯噪音)
- [ ] verdict 三种之一 (GO / PARTIAL / NEUTRAL / DEAD) 自动打印
- [ ] 不修改 v50 任何文件
- [ ] 不读 v28 原始数据 (内存生成)
- [ ] ponytail 注释 3 处 (FSK 编码 / 数据生成 / 不动 v50)
- [ ] README.md 1 段说明与 exp31/32 区别

---

## 8. 后续路径 (post-smoke)

**如果 GO** (>90% char accuracy):
1. 扩 vocab: 8 → 16 (S=64, T_CHAR=16, N_CHARS=4) → 32 (S=128, T_CHAR=32, N_CHARS=4)
2. 扩长度: 8 char → 16 char → 64 char (long-range dependency)
3. 进 Phase 4.3: 真实 LM 评估 (PPL on v28)

**如果 PARTIAL** (50-90% + CWF 优势 ≥ 2x):
1. 调查失败字符 (哪些 char_id 重建失败)
2. 调 amp rescale (×1, ×10, ×100)
3. 调 char freq spacing (Δω)

**如果 NEUTRAL** (50-90% + 无优势):
1. 检查 Transformer baseline 是不是也失败 (是的话任务本身难, 不是 CWF 问题)
2. 考虑加 learned channel coding (char_id → freq 不是 1:1 映射)

**如果 DEAD** (<50%):
1. 写 postmortem: 为什么 CWF 在 FSK+text 上不行
2. 候选解释: 8 char 太短 / FSK 频谱不在 CWF 归纳偏置覆盖域 / MSE 损失错配
3. 归档 "CWF + 文本脉冲波" 路线, v50 不动

---

## 9. 附录

### 9.1 复用与新写

| 组件 | 来源 | 改动 |
|------|------|------|
| CWFSingleBlock | `prototype/cwf_minimal.py` | 0 改动 |
| TransformerPredictor | `exp31_pde_smoke.py` | 改输入 (B,S)→(B,S,2) real |
| FSK 编码 | 新写 (`fsk_encode` 函数) | ~15 行 |
| FSK 解码 | 新写 (`fsk_decode` 函数) | ~10 行 |
| 数据生成 | 新写 (内存随机) | ~10 行 |
| 训练循环 | `exp31.train_one` | 复用 90%, 改 loss 接口 |
| 评估 | 新写 (char accuracy) | ~30 行 |

### 9.2 参考文档

- `research/cwf/docs/v51_phase4_text_wave.md` §2 映射 B (本文档依据)
- `research/cwf/docs/v51_roadmap.md` §3 Phase 1-4 (roadmap 顺序)
- `research/cwf/experiments/exp31_pde_smoke/` (架构 + 训练循环模板)
- `research/cwf/experiments/exp32_multiseed/` (multi-seed 验证模板)
- `research/cwf/prototype/cwf_minimal.py` (CWFSingleBlock)
- `docs/superpowers/specs/2026-06-24-cwf-resurrection-design.md` (spec 格式参考)

### 9.3 与 v51 roadmap §5 Phase 4.2 的差异

| roadmap 写法 | 本 spec 写法 | 差异原因 |
|--------------|--------------|----------|
| Phase 1-3 + 4.1 后启动 | 跳过 Phase 1-3 + 4.1 | 用户决定 (2026-07-15) |
| 字符重建 > 90% / < 50% 阈值 | 同 | 完全一致 |
| CPU 可跑, 不占 GPU | 同 | 完全一致 |
| 验证后扩 v23 BPE | 暂未列入, 见 §8 后续路径 | smoke 优先, 后续再说 |

---

**用户审阅**: 请确认以下 5 点无误后, 我会进入 writing-plans 阶段, 写出实施计划:
1. 数据流 §3.1 与 FSK 编码 §3.3 符合预期
2. CWF 架构 §3.4 零改动思路 OK
3. 训练 §3.5 + 评估 §3.6 阈值 OK
4. 文件结构 §4 (单文件 ~250 行) OK
5. 决策记录 §6 + 验证清单 §7 完整
