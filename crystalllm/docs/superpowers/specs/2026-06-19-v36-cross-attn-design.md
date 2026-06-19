# CrystaLLM v36 — BAD-DP v2 (Cross-Attention Standard Decoder)

> **Q: 把 BAD-DP decoder 的"z 作为单 token 拼接"改为"每层 cross-attention to z"，能从 v25 (PPL 2.47) 突破吗？**
> **A: 待验证。预期 PPL < 2.30, KL < 200, 生成非空格率 > 90%。**

## TL;DR

| 项 | v25 (BAD-DP) | **v36 (BAD-DP v2 + cross-attn)** | 变化 |
|---|---|---|---|
| 架构 | z→embedding → pos 0 拼接, self-attn only | z→K/V, 每 block cross-attn, 不再 prepend | 架构升级 |
| 参数量 | 476M | **~570M** (+94M, 24 blocks × ~4M) | +20% |
| PPL | 2.47 | **< 2.30 (目标)** | -7%+ |
| 速度 | 828ms | **< 1500ms** (估 +50%) | 可接受 2x 开销 |
| KL | ~250 (估) | **< 200** | z 被真使用 |
| 非空格率 | ~70% (估) | **> 90%** | 防坍缩 |
| Warm-start | — | **from v25** (非 v28.5) | 避开坍缩遗传 |
| 数据 | v24 19K | v24 19K (不变) | 控制变量 |
| 训练时间 | 12 min | ~15-18 min | +30% |

## 1. 背景与动机

### 1.1 v35 揭示的根本问题

v35 报告 (commit eab720d) 揭示：
- v28.5 verifier (PPL 2.39) **从零生成坍缩到空格**，只能续写 prefix
- v31 SpS "95.5% 接受率"是**空格对空格的 trivial 匹配**，框架从未真正工作
- v35 drafter 学到了真模式（6-8 unique tokens），但被坍缩 verifier 拒绝

### 1.2 BAD-DP 架构的固有缺陷

v28.5 报告 (Section 4.4) 已指出：
```
decoder 只看 z, 不看 prefix, 限制了模型容量上限
KL 持续 295-350 nats → z 信息未被充分利用
```

**根因分析**：当前 BAD-DP 的 z 注入方式：
```python
input = cat([z_emb, bos_emb, x_emb], dim=1)  # z 作为 pos 0 单 token
input = input + pos_emb
for block: input = block(input)              # 仅 self-attn
logits = head(ln_f(input))[:, 1:T+1]         # 丢掉 z 位置的输出
```

- z 被压缩到 1 个 token 的 embedding 中
- 512 位置的 attention 必须"绕回" pos 0 才能用 z
- 信息瓶颈在 `z_to_emb: 256→1280` 单次投影
- 训练时 z 的梯度信号被 512 位置的 self-attn 稀释

### 1.3 v36 修复方向

**核心思路**：让 z 在每个 block 都作为 K/V 可被查询，不是只作为序列的第一个 token。

```
BAD-DP (v25):     z_emb → pos 0 → self-attn only
BAD-DP v2 (v36):  z_kv_proj → K/V → self-attn + cross-attn
```

预期：
1. **KL 显著下降**（z 被有效消费）
2. **PPL 改善**（decoder 能学到 z 和 prefix 的对应关系）
3. **生成不坍缩**（z 提供强信号，避免从零生成塌到空格）

## 2. 架构详细变更

### 2.1 新 Decoder 结构

**DecoderCrossAttn**：
```python
class DecoderCrossAttn(nn.Module):
    def __init__(s, d_z=D_Z):
        super().__init__()
        s.d_z = d_z
        s.tok = nn.Embedding(V, DEC_EMBD)
        s.pos = nn.Embedding(T + 2, DEC_EMBD)  # T=512 → 514
        s.blocks = nn.ModuleList([
            BlockCrossAttn(DEC_EMBD, DEC_HEAD, d_z) for _ in range(DEC_LAYER)
        ])
        s.ln_f = nn.LayerNorm(DEC_EMBD)
        s.head = nn.Linear(DEC_EMBD, V, bias=False)
        s.tok.weight = s.head.weight
    def forward(s, z, x):
        B, T = x.shape
        bos = s.tok(torch.tensor([BOS_ID], device=x.device)).expand(B, 1, -1)
        inp = torch.cat([bos, s.tok(x)], dim=1)              # 不再 prepend z
        inp = inp + s.pos(torch.arange(T + 1, device=x.device))
        for b in s.blocks: inp = b(inp, z)                   # z 传给每个 block
        return s.head(s.ln_f(inp))                           # logits[:, :T]
```

**关键变化**：
- ❌ 移除 `z_to_emb` 投影（被 k_proj/v_proj 替代）
- ❌ 不再将 z_emb prepended 到输入序列
- ✅ z 直接传入每个 block 作 cross-attn 的 K/V

### 2.2 BlockCrossAttn 子层

```python
class BlockCrossAttn(nn.Module):
    """self-attn + cross-attn(z) + mlp"""
    def __init__(s, N_EMBD, N_HEAD, D_Z):
        super().__init__()
        s.nh = N_HEAD; s.head_dim = N_EMBD // N_HEAD
        # Self-attn (warm-start from v25)
        s.ln1 = nn.LayerNorm(N_EMBD)
        s.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        s.proj = nn.Linear(N_EMBD, N_EMBD)
        # Cross-attn (NEW, 随机初始化)
        s.ln_cross = nn.LayerNorm(N_EMBD)
        s.q_cross = nn.Linear(N_EMBD, N_EMBD)
        s.k_cross = nn.Linear(D_Z, N_EMBD)
        s.v_cross = nn.Linear(D_Z, N_EMBD)
        s.proj_cross = nn.Linear(N_EMBD, N_EMBD)
        # MLP (warm-start from v25)
        s.ln2 = nn.LayerNorm(N_EMBD)
        s.mlp = nn.Sequential(nn.Linear(N_EMBD, 4*N_EMBD), nn.GELU(),
                              nn.Linear(4*N_EMBD, N_EMBD))
    def forward(s, x, z_kv):
        B, T, C = x.shape
        # self-attn (unchanged)
        h = s.ln1(x)
        qkv = s.qkv(h).reshape(B, T, 3, s.nh, s.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = x + s.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True)
                         .transpose(1, 2).contiguous().view(B, T, C))
        # cross-attn (NEW)
        h_c = s.ln_cross(x)
        q_c = s.q_cross(h_c).reshape(B, T, s.nh, s.head_dim).permute(0, 2, 1, 3)
        k_c = s.k_cross(z_kv).reshape(B, 1, s.nh, s.head_dim).permute(0, 2, 1, 3)
        v_c = s.v_cross(z_kv).reshape(B, 1, s.nh, s.head_dim).permute(0, 2, 1, 3)
        y_c = F.scaled_dot_product_attention(q_c, k_c, v_c)  # 无 causal mask
        x = x + s.proj_cross(y_c.transpose(1, 2).contiguous().view(B, T, C))
        # MLP
        x = x + s.mlp(s.ln2(x))
        return x
```

### 2.3 参数量变化

| 项 | 数量 | 说明 |
|---|---|---|
| q_cross | D² = 1.638M | per block |
| k_cross | D_Z × D = 0.328M | per block |
| v_cross | D_Z × D = 0.328M | per block |
| proj_cross | D² = 1.638M | per block |
| **每 block 新增** | **~3.93M** | |
| 24 blocks 总 | ~94.4M | 新增 |
| 移除 z_to_emb | -0.33M | |
| **净增** | **~94M** | |
| **总参数** | **~570M** | vs v25 476M |

### 2.4 速度影响估算

| 项 | v25 | v36 (估) |
|---|---|---|
| Self-attn | 24 layers | 24 layers (不变) |
| Cross-attn | 0 | 24 layers × ~5-10ms |
| 总推理开销 | 828ms | **1000-1100ms** |
| 增量 | — | +20-30% |

## 3. 训练策略

### 3.1 Warm-Start 加载

**起点**：v25_decoder.pt（PPL 2.47, 真正生成有意义代码）

| 来源权重 | 用途 |
|---|---|
| tok / head (V=2261) | 完全复用 |
| pos (T=512, 514) | 完全复用 |
| blocks[i].ln1, qkv, proj | 完全复用 (self-attn) |
| blocks[i].ln2, mlp | 完全复用 (MLP) |
| blocks[i].ln_cross, q_cross, k_cross, v_cross, proj_cross | **随机初始化** (He) |
| ~~z_to_emb~~ | 移除 (被 k_cross/v_cross 替代) |

**加载代码**：
```python
v25_state = torch.load("v25_decoder.pt", weights_only=False)["decoder"]
new_decoder = DecoderCrossAttn().to(DEVICE)
new_state = new_decoder.state_dict()
loaded, skipped, fresh = 0, 0, 0
for k, v in v25_state.items():
    if k == "z_to_emb.weight" or k == "z_to_emb.bias":
        skipped += 1; continue
    if k in new_state and v.shape == new_state[k].shape:
        new_state[k] = v; loaded += 1
    else:
        fresh += 1  # 保留 new_state 中对应 key 的随机初始化
new_decoder.load_state_dict(new_state)
```

**预期统计**：loaded ≈ 290, skipped ≈ 2 (z_to_emb), fresh ≈ 96 (4 weights × 24 blocks)。

**为什么不从 v28.5 warm-start**：
v28.5 本身已坍缩到空格分布，warm-start 会继承这个坏行为。从 v25 (真正能生成) 出发，cross-attn 子层随机初始化时，self-attn 部分能保证 decoder 仍正常生成。

### 3.2 训练超参

| 参数 | 值 | 来源 |
|---|---|---|
| B | 4 | v25 |
| T | 512 | v25 |
| LR | 1e-4 | v25 |
| STEPS | 4000 | v25 |
| Optimizer | AdamW, wd=0.1, β=(0.9, 0.95) | v25 |
| Schedule | Cosine | v25 |
| W_KL | 0.1 | v25 |
| KL anneal | 1000 steps | v25 |
| Free bits | 1.0 nat | v25 |
| Grad clip | 1.0 | v25 |
| Seed | 42 | v25 |

**数据**：复用 v24_train.parquet (19,307) + v24_val.parquet (1,016)。
**缓存 z**：复用 cached_v24_z.npz（D_Z=256, 与 v25 decoder 同源）。

### 3.3 预期训练时间

~15-18 min（vs v25 12 min, 增加 30% 因为参数多了 94M）。

### 3.4 训练时观察重点

1. **KL 必须下降**：cross-attn 应让 z 信息被消费，KL 显著低于 v28.5 的 295-350 nats
2. **训练 best PPL < 2.0**：证明模型容量足够
3. **测试 PPL 接近 best**：不严重过拟合

## 4. 评测与成功 / 失败 标准

### 4.1 评测指标（5 项）

| # | 指标 | 含义 | 通过阈值 |
|---|---|---|---|
| 1 | **PPL** | 端到端预测能力 | **< 2.30** (vs v25 2.47, -7%) |
| 2 | **生成质量 (非空格率)** | 真实生成能力 | **> 90%** 非空格 token |
| 3 | **KL 收敛** | z 是否被使用 | **KL < 200** |
| 4 | **速度** | 推理效率 | **< 1500ms** |
| 5 | **样本检查** | 视觉确认 | 至少 1 个样本含 import/def/class |

**指标 2 是关键防线**：v35 揭示 v28.5 verifier 根因就是"从零生成坍缩到空格"。本次实验必须验证 v36 **不重蹈覆辙**。

### 4.2 成功判定（全部满足）

- [ ] PPL < 2.30
- [ ] 非空格率 > 90%
- [ ] KL < 200
- [ ] 速度 < 1500ms
- [ ] 至少 1 个生成样本含 import/def/class

### 4.3 失败模式 + 后续决策

| 失败模式 | 判定 | 下一步行动 |
|---|---|---|
| PPL 不下降 (< 2.40) | cross-attn 无效 | 改为 prefix-tuning (z→M tokens) |
| 生成仍坍缩 | 架构不够 | 检查训练数据 + 加 BOS 启动样本 |
| KL 仍 > 250 | z 仍难利用 | 加 cross-attn 频率（每 2 层一次 → 每层） |
| PPL 提升但速度 > 2000ms | 开销过大 | 减 cross-attn 维度（D_EMBD → 768） |

### 4.4 对 v31 SpS 框架的影响

v36 若成功，**v31 verifier 从 v28.5 换为 v36**，重做 v31 SpS 评测：
- v36 不会坍缩到空格，SpS 接受率不再"假象"
- 真实接受率应基于"非空格 token"统计
- 接受率指标本身也需修（v35 报告指出）

## 5. 文件清单

### 5.1 新增文件

| 文件 | 用途 |
|---|---|
| `crystalllm/train_v36_decoder.py` | 训练脚本（warm-start v25 + cross-attn） |
| `crystalllm/eval_v36_e2e.py` | 端到端评测（PPL, KL, 速度） |
| `crystalllm/debug_v36_gen.py` | 生成质量调试（非空格率 + 样本输出） |
| `crystalllm/v36_decoder.pt` | ~570M 模型 (~2.2 GB) |
| `crystalllm/v36_decoder_train_log.json` | 训练日志 |
| `crystalllm/v36_e2e.json` | 5 项指标 JSON |
| `crystalllm/v36_results.md` | 实验报告 |
| `crystalllm/docs/superpowers/specs/2026-06-19-v36-cross-attn-design.md` | 本设计文档 |

### 5.2 不修改的现有文件

- `v25_decoder.pt` / `train_v25_decoder.py` / `eval_v25_e2e.py`（只读）
- `v28_5_*`, `v31_*`, `v35_*`（历史对照）
- `data/processed/*`（数据不动）

## 6. 实验时间表

| 阶段 | 工作 | 时间 | 产出 |
|---|---|---|---|
| 1. 准备 | 写 train_v36_decoder.py + 调试 warm-start 加载 | 30-45 min | 脚本可跑通 |
| 2. 训练 | 4000 step on RTX 5090 | ~15-18 min | v36_decoder.pt + log |
| 3. PPL 评测 | 跑 eval_v36_e2e.py | ~3 min | v36_e2e.json |
| 4. 生成质量 | 跑 debug_v36_gen.py，10 个样本检查 | ~5 min | 非空格率 + 样本 |
| 5. 报告 | 写 v36_results.md | 15 min | 实验报告 |
| **合计** | | **~1.5-2 小时** | |

## 7. 风险点

| 风险 | 影响 | 预案 |
|---|---|---|
| Warm-start 形状不匹配 | 加载失败 | 严格 shape check，提前验证 |
| KL 数值异常 | 评测无意义 | KL 基于 (mu, logvar)，不受 decoder 架构影响 |
| 速度爆炸 (>2000ms) | 实用价值低 | 减 cross-attn 维度或减层数 |
| Cross-attn 不收敛 | PPL 不降 | 增大 LR、调整 init |

## 8. 总结

v36 是 v35 揭示的根本问题的**架构层修复**：

1. **BAD-DP 改为 cross-attention BAD-DP**：z 不再是单 token，而是每层可查询的 K/V
2. **warm-start 自 v25**：避开 v28.5 坍缩遗传
3. **控制变量**：规模、数据、超参全继承 v25
4. **5 项评测指标**：PPL + 非空格率 + KL + 速度 + 样本检查
5. **明确成功阈值**：PPL < 2.30 + 非空格率 > 90% + KL < 200

**核心一句话**：把"z 作为 pos 0 拼接"改成"z 作为每层 K/V"，让 decoder 在每个 block 都能直接消费 z 信息，不再依赖自注意力"绕回"。

---

**下一步**：v36 成功后，v31 SpS 框架的 verifier 从 v28.5 换为 v36，重做投机解码评测，验证 SpS 真实加速（不再有"空格对空格"假象）。