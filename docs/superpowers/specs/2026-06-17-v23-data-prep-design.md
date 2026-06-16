# CrystaLLM v23 — 100G 数据准备设计

**日期**: 2026-06-17
**作者**: Claude (brainstorming with user)
**状态**: Draft (待用户复审)
**对应 git**: 基于 v22a `e62bfea`，作为 v23 (M8 扩数据) 的输入

---

## 1. 背景与动机

CrystaLLM 经历了 v1-v6 (prefix-LM) + v10-v22a (BAD-DP) 两条路线，v22a 在 **1893 真实会话 / 9.2M tokens** 上达到 **val PPL 4.39**。TIMELINE §8.1 将 v23 列为"扩数据 1893→N"，目标"PPL < 4.0 (5K) / < 3.5 (10K) / < 3.2 (50K)"。

本地 2305 sessions 已用尽，v23 需要**外部 100G 数据**作为训练补充。本 spec 定义从 5 个 ModelScope 数据源（70% agentic + 20% code + 10% wiki）到单 parquet 的完整数据准备流水线。

---

## 2. 目标

**主目标**：把 100G 外部数据 + 现有本地 16MB v23 数据 → 单 parquet → 喂 v23 decoder warm-start 训练，**val PPL < v22a 的 4.39**。

**次目标**：
- 流水线 5 步独立可重入（任意 step 中断可从该 step 续跑）
- 磁盘峰值 < 200GB（不含 SDK 临时缓存）
- 100% char-level 兼容（不重训 vocab，保留 vocab=2261）
- val 集 = v22a 1893+210 永远不动（论文可比性）

**非目标**：
- 不动 v22a 任何已训权重
- 不引新依赖（仅 `modelscope` / `datasketch` / `pyarrow` / `tqdm` / `pandas`）
- 不强求 1 epoch 看完 100G（接受 12 min 训练看 ~33M chars）

---

## 3. 数据域与配额

| 域 | 源 | 大小 (chars) | 占比 | 获取方式 |
|---|---|---:|---:|---|
| **agentic** | `armand0e/claude-fable-5-claude-code` | 待探 | 70% 总配额一部分 | 全量下载 |
| agentic | `Glint-Research/Fable-5-traces` | 待探 | 70% 总配额一部分 | 全量下载 |
| agentic | `lazarus19/Vibe-Coding-Claude-Fable-5` | 待探 | 70% 总配额一部分 | 全量下载 |
| agentic | local `sessions.parquet` (2305) | 9.16M | 70% 总配额一部分 | 已有 |
| **code** | `swift/github-code` (Python + C++) | 20 GB | 20% | streaming, 凑够即停 |
| **wiki** | `swift/wikipedia` (zh + en) | 10 GB | 10% | streaming, 凑够即停 |
| **eval** | `ZhipuAI/humaneval-x` | 164 doc | n/a | 全量下载, 仅作 pass@1 sanity |

**配额基准**：**清洗后字符**（与 char-level 训练一致）。最终分配按 character 比例，**非**按文档数。

**`HumanEval-X` 角色**：仅作 v23 训后 zero-shot pass@1 sanity check，**不入 PPL**。

---

## 4. 架构（5 步流水线 + 1 baseline 锚点）

```
┌──────────────────────────────────────────────────────────────┐
│ 锚点: v22a val (1893+210) — 不动, 作 PPL 比较 anchor         │
├──────────────────────────────────────────────────────────────┤
│ Step 0 (新建): discover_v23_schema.py                        │
│   每个源下 100MB → 记录 schema/avg_len/sample_n              │
│   → data/schema_v23/{source}.json                            │
│   报警: empty > 10% / min_len < 10                           │
├──────────────────────────────────────────────────────────────┤
│ Step 1a (新建): download_v23_agentic.py                      │
│   3 Fable 5 源 + HumanEval-X → data/raw_v23/agentic/*.jsonl  │
│   线程池并行 3 worker, ModelScope SDK cache_dir resume       │
├──────────────────────────────────────────────────────────────┤
│ Step 1b (新建): download_v23_streaming.py                    │
│   github-code (Py+C++) 边下边清洗 → 凑够 20GB 停             │
│   wikipedia (zh+en) 边下边清洗 → 凑够 10GB 停                │
│   临时缓存: D:/tmp_v23_dl/ (step 4 后删)                     │
├──────────────────────────────────────────────────────────────┤
│ Step 2 (新建): clean_v23_data.py                             │
│   过滤 control chars / 不可打印 unicode / 统一换行            │
│   长度过滤: 10 ≤ len ≤ 50_000 chars                          │
│   → data/clean_v23/{agentic,code,wiki}/*.jsonl               │
├──────────────────────────────────────────────────────────────┤
│ Step 3 (新建): dedup_v23_data.py                             │
│   SHA-1 (前 200 chars) 精确去重 + datasketch MinHash 近重     │
│   num_perm=128, ngram=5, threshold=0.85                      │
│   跨域交叉去重 (local↔Fable 5: 0.90, 其他: 0.85)            │
│   LSH sqlite 后端, CHUNK_SIZE=1M doc, LeanMinHash 落盘        │
│   → data/dedup_v23/{agentic,code,wiki}/*.jsonl               │
│   → data/dedup_v23/report.json (per-domain 统计)             │
├──────────────────────────────────────────────────────────────┤
│ Step 4 (新建): pack_v23_data.py                              │
│   4 路输入 [agentic, code, wiki, local_v23]                   │
│   按字符配额 70/20/10 采样                                    │
│   T=512 greedy bin-packing (短文本拼, <sep> 隔)               │
│   → data/processed/extended_v23.parquet                      │
│   字段: text / domain / source / n_docs / n_chars             │
├──────────────────────────────────────────────────────────────┤
│ Step 5 (新建): proto_v23_decoder.py + eval_v23_e2e.py       │
│   warm-start from proto_v22_decoder.pt (24L×1280×20, 475M)   │
│   MAX_SEQ_LEN 128→512, batch_size=8                          │
│   8000 step × 2 phase (anneal 1e-4 → fine 5e-5)               │
│   val 强制 v22a_val.parquet (PPL<4.39 验收)                  │
│   HumanEval-X pass@1 sanity check                            │
│   → results_v23.tsv + proto_v23_decoder.pt                   │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. 详细设计

### 5.1 Step 0 — Schema 探查（必跑）

**为什么必需**：streaming 源不先知道 schema 就写清洗 = 赌博。100MB 探查 < 2 min，能避免数小时返工。

**脚本**：`discover_v23_schema.py`

**接口**：
```python
python discover_v23_schema.py --source swift/github-code --lang Python --sample-mb 100
python discover_v23_schema.py --source swift/wikipedia --lang en --sample-mb 50
```

**输出**：`data/schema_v23/{source}.json`：
```json
{
  "source": "swift/github-code",
  "lang": "Python",
  "fields": ["code", "repo_name", "path", "language", "size"],
  "field_types": {"code": "str", "size": "int64"},
  "sample_n": 1247,
  "avg_text_len": 1823,
  "median_text_len": 542,
  "max_text_len": 158293,
  "min_text_len": 12,
  "empty_text_n": 0
}
```

**断言**：
- `empty_text_n / sample_n > 0.10` → 报警
- `min_text_len < 10` → 报警

---

### 5.2 Step 1a — Agentic 全量下载

**脚本**：`download_v23_agentic.py`

**3 个 Fable 5 源**（基于 Step 0 探查决定是否全下）：
```python
AGENTIC_SOURCES = [
    ("armand0e/claude-fable-5-claude-code", None, "train"),
    ("Glint-Research/Fable-5-traces", None, "train"),
    ("lazarus19/Vibe-Coding-Claude-Fable-5", "default", "train"),
]
```

**实现要点**：
- 线程池 3 worker 并行下（不进程，避免 OOM）
- ModelScope SDK `cache_dir` = `~/.cache/modelscope/`，自动 resume
- tqdm 进度写 stderr
- 每 5 min flush 磁盘配额日志

**输出格式**（每行 JSON）：
```json
{"text": "...", "source": "claude-fable-5-claude-code", "doc_id": "abc123"}
```

**错误处理**：
| 异常 | 处理 |
|---|---|
| `ConnectionError` | 退避 5s 重试 3 次，仍失败 → 记 `data/raw_v23/agentic/FAILED.txt` |
| `DiskFullError` | 立即停 + 报警，不清空已下载 |
| 单 source 失败 | 不影响其他 source |

---

### 5.3 Step 1b — 流式配额下载（2TB 约束）

**脚本**：`download_v23_streaming.py`

**接口**：
```python
python download_v23_streaming.py \
    --code-quota-chars 20_000_000_000 \
    --wiki-quota-chars 10_000_000_000
```

**配额停止逻辑**（code 示例）：
```python
char_count = 0
for doc in ds.streaming_iter():
    if filter_lang(doc, allow={"Python", "C++"}):
        cleaned = clean_text(doc["code"])  # 立即粗清洗
        if 10 <= len(cleaned) <= 50_000:
            write_jsonl(cleaned)
            char_count += len(cleaned)
            if char_count >= target_char_quota:
                break
```

**不落盘原始 jsonl**（避免 2TB 风险）：
- 边下 → 立即粗清洗 → 累积到 `data/raw_v23/code/streaming_*.jsonl`
- 累积到 100MB 切一个文件
- 临时目录 `D:/tmp_v23_dl/` 存 SDK 缓存，**step 4 完成后**才删

**Wikipedia 子集**：
```python
WIKI_SOURCES = [
    ("swift/wikipedia", "zh", "train"),
    ("swift/wikipedia", "en", "train"),
]
# 凑够 10GB, zh:en 按可用比例 1:9
```

**断点续传**：`--resume-from-file data/raw_v23/code/streaming_0001.jsonl`，统计已写字符数作为新起点。

**错误处理**：
| 异常 | 处理 |
|---|---|
| `OSError: [Errno 28] No space left` | 立即停 + 提示清理 `D:/tmp_v23_dl/` |
| 流中断 (timeout) | 退避 10s 重试 5 次 |
| `EmptyDatasetError` (wiki 某子集空) | 跳过该子集，继续其他 |

---

### 5.4 Step 2 — 字符级清洗

**脚本**：`clean_v23_data.py`

**清洗规则**：
```python
def clean_text(text: str) -> str | None:
    if not isinstance(text, str): return None
    # 去 control chars (保留 \n \t)
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)
    # 不可打印 unicode
    text = ''.join(c for c in text if c.isprintable() or c in '\n\t')
    # 统一换行
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # 长度过滤
    if len(text) < 10 or len(text) > 50_000:
        return None
    return text
```

**输出**：`data/clean_v23/{agentic,code,wiki}/*.jsonl`（schema 同 raw_v23，多 `clean_len` 字段）

**验收**：`总 char 损耗 < 5%`（清洗不破坏）

---

### 5.5 Step 3 — MinHash 去重（关键风险点）

**脚本**：`dedup_v23_data.py`

**MinHash 参数**：
```python
MINHASH_CONFIG = {
    "num_perm": 128,         # 概率误差 ~9%, 阈值 0.85 足够
    "ngram": 5,              # 字符 5-gram
    "threshold": 0.85,       # Jaccard > 0.85 视为近重
    "shingle_method": "char",
}
EXACT_HASH_PREFIX_LEN = 200 # SHA-1 只对前 200 字符
```

**内存预算**（关键）：
- 100B chars / 5-gram = 20B shingles
- 50M 文档 × MinHash 128 perm × 4 byte = **25.6 GB**
- + SHA-1 表 ≈ 10GB
- **总计 ~36GB RAM**（32GB PC 会 OOM）

**解决方案**：**LeanMinHash + sqlite LSH**：
```python
from datasketch import MinHashLSH, LeanMinHash
lsh = MinHashLSH(threshold=0.85, num_perm=128, storage_config={
    "type": "sqlite",
    "basename": "data/dedup_v23/lsh.db"
})
# LSH 索引 < 2GB 内存, LeanMinHash 落盘 25GB
```

**分块处理**：
```python
CHUNK_SIZE = 1_000_000  # 每块 100 万文档
for chunk in chunks_iter():
    # 1) multiprocessing.Pool 8 workers 算 MinHash
    # 2) LSH query 找近重
    # 3) 标记删除, 写 chunk_done.json
```

**跨域阈值**：

| 源 A | 源 B | 阈值 | 理由 |
|---|---|---|---|
| Fable 5a/b/c 互相 | — | 0.85 | 名字相似可能同源 |
| local `sessions.parquet` | Fable 5 全部 | **0.90** | 提防 Fable 5 反向包含本地旧任务 |
| github-code | Fable 5 | 0.95 (近严格) | 代码可能相似但 agentic 上下文不同 |
| github-code | wikipedia | 0.95 | wiki 偶尔有代码示例 |
| wikipedia | Fable 5 | **不去重** | 域差异大，Jaccard 自然低 |

**4 步去重流水线**：
1. `exact_hash_dedup` → `keep_ids_A.json`
2. `intra_domain_minhash` → `keep_ids_B.json`
3. `cross_domain_minhash` → `keep_ids_C.json`
4. `take_intersection` → `data/dedup_v23/{domain}/*.jsonl`

每步幂等可重跑。

**验收报告**：`data/dedup_v23/report.json`：
```json
{
  "input_chars_total": 105000000000,
  "output_chars_total": 73500000000,
  "dedup_ratio": 0.30,
  "intra_domain_removed": 12000000000,
  "cross_domain_removed": 4500000000,
  "exact_dup_removed": 15000000000,
  "per_domain": {
    "agentic": {"input": 73500000000, "output": 55000000000},
    "code":    {"input": 21000000000, "output": 14700000000},
    "wiki":    {"input": 10500000000, "output":  3800000000}
  }
}
```

**报警**：dedup_ratio ∉ [0.10, 0.50]

---

### 5.6 Step 4 — 配额采样 + T=512 打包

**脚本**：`pack_v23_data.py`

**4 路输入**：
- `data/dedup_v23/agentic/*.jsonl`（含 local_v23 副本）
- `data/dedup_v23/code/*.jsonl`
- `data/dedup_v23/wiki/*.jsonl`
- `data/processed/v23_train.parquet`（已有，agentic 子集）

**T=512 packing**：
```python
PACK_LEN = 512

def pack_documents(docs: list[str]) -> list[list[str]]:
    """贪心打包: 当前 bin 没满就塞下一个 doc, 满了就开新 bin."""
    bins = []
    cur_bin, cur_len = [], 0
    for doc in docs:
        if cur_len + len(doc) + 1 > PACK_LEN:  # +1 for <sep>
            bins.append(cur_bin)
            cur_bin, cur_len = [doc], len(doc) + 1
        else:
            cur_bin.append(doc)
            cur_len += len(doc) + 1
    if cur_bin:
        bins.append(cur_bin)
    return bins
```

**关键设计**：
- `<sep>` 复用 `<eos>` token
- 不插 `<bos>`（dataloader 自己加）
- 跳过 < 50 chars 的退化 pack

**配额计算**：
- 总目标 ~70B chars（agentic 70% × 100B + 实际去重后调整）
- agentic 70% / code 20% / wiki 10% 比例
- 按字符配额采样，**不**按文档数

**Parquet schema**：
```python
{
    "text": str,         # pack 后的字符串, 长度 = 512
    "domain": str,       # "agentic" | "code" | "wiki" | "local_v23"
    "source": str,       # 具体源名
    "n_docs": int,       # 这个 pack 装了几个 doc
    "n_chars": int,      # 原始字符数
}
```

**写盘配置**：
```python
df.to_parquet(
    "data/processed/extended_v23.parquet",
    engine="pyarrow",
    compression="snappy",
    index=False,
    row_group_size=10_000,
)
```

**估大小**：~25GB on disk（snappy 压缩）

---

### 5.7 Step 5 — 训练接入

**脚本**：`proto_v23_decoder.py` + 适配 `train.py` / `eval_v23_e2e.py`

**架构**：完全复用 v22a decoder (24L×1280×20, 475M)，**不引入新变量**。

**Warm-start**：
```python
decoder_v23 = build_decoder_from_v22a_config()  # 24L×1280×20
state = torch.load("proto_v22_decoder.pt")
# v22a z_to_emb 是 256→1280, v23 复用 256 维 z → 同一 layer
decoder_v23.load_state_dict(state, strict=True)
```

**训练 schedule**：

| 阶段 | 步数 | LR | 目的 |
|---|---|---|---|
| warm-start | 0 | — | 复用 v22a 权重 |
| phase 1 (anneal) | 2000 | 1.0e-4 | 大 batch 看新数据分布 |
| phase 2 (fine) | 4000 | 5.0e-5 | 收敛到 PPL<4.39 |
| phase 3 (val) | — | — | 跑 v22a val PPL 验收 |

**总训练时间估**：8000 step × ~90ms/step (T=512 翻 4x) = **12 min**（同 v22a）

**`train.py` 关键改动**：
```python
# 原
TRAIN_PATH = "data/processed/v22a_train.parquet"  # 1893 doc
VAL_PATH   = "data/processed/v22a_val.parquet"    # 210 doc
MAX_SEQ_LEN = 128

# v23 新
TRAIN_PATH = "data/processed/extended_v23.parquet"  # 137M pack
VAL_PATH   = "data/processed/v22a_val.parquet"     # **不变, anchor**
MAX_SEQ_LEN = 512
```

**`val` 不动的原因**：v22a PPL=4.39 是论文核心数字，v23 必须**同口径**比较。

**评估输出**（必填）：
| 指标 | 阈值 |
|---|---|
| v22a val PPL | **< 4.39** (主指标) |
| 端到端 PPL (encoder → 256z → 5步 prior → decoder) | 比率 < 1.10 |
| 速度 (5+100 AR tokens) | < 1000ms |
| HumanEval-X pass@1 | > 0%（无 baseline） |
| 按 domain 分桶 PPL | 报告 agentic/code/wiki 各自 |

---

## 6. 错误处理（3 原则）

**原则 1：每个 stage 独立可重入**
- `--stage {1,2,3,4,5}` 显式指定起点
- 检查点写到 `{stage_name}.checkpoint.json`

**原则 2：失败立即停 + 报警，不静默重试**
- 任何 crashed → `data/logs_v23/crashes.log`
- 报警含 stage / 命令 / stack 前 30 行 / 已处理文档数

**原则 3：磁盘/内存硬限制 → 立即停**
- 磁盘余量 < 5GB → 停 + 提示清理
- RSS > 30GB → 停 + 减小 `CHUNK_SIZE`
- GPU OOM → 减小 `batch_size` 重试 1 次，仍 OOM 则停

---

## 7. 测试

### 7.1 单元测试（`tests/test_v23_data_prep.py`）
- `test_clean_text_strips_control_chars`
- `test_clean_text_normalizes_newlines`
- `test_minhash_threshold_filters_near_dups`
- `test_pack_documents_respects_pack_len`
- `test_discover_schema_rejects_empty_dataset`
- `test_quota_sampling_respects_ratios`

### 7.2 集成测试（`tests/test_v23_pipeline_smoke.py`）
- `test_smoke_full_pipeline_with_tiny_data`：100 文档端到端

### 7.3 验收测试（`tests/test_v23_acceptance.py`）
- `extended_v23.parquet` 存在 + 字段完整
- char-level vocab 仍是 2261
- dedup ratio ∈ [0.10, 0.50]
- val 仍是 v22a 1893+210
- train.py 能 dry-run 100 step 不 crash

---

## 8. 验收 Checklist

### 8.1 阶段验收

| Stage | 必须达成 |
|---|---|
| 0 (discover schema) | 4 源 schema 报告完整，no empty dataset |
| 1a (agentic 下载) | 3 Fable 5 源 + HumanEval-X 文件存在，size > 0 |
| 1b (stream 下载) | code ≥ 18GB chars, wiki ≥ 9GB chars (留 10% buffer) |
| 2 (clean) | 总 char 损耗 < 5% |
| 3 (dedup) | dedup_ratio ∈ [0.10, 0.50], per-domain 报告完整 |
| 4 (pack) | `extended_v23.parquet` 可读，137M pack ±20% |
| 5 (train) | val PPL < 4.39, 8000 step 12 min |

### 8.2 最终验收（v23 完成）

- [ ] `data/processed/extended_v23.parquet` 入仓
- [ ] `proto_v23_decoder.pt` 保存，shape 与 v22a 一致
- [ ] `eval_v23_e2e.py` 跑通 4 维 KR，输出 `results_v23.tsv`
- [ ] `crystalllm/TIMELINE.md` 更新 v23 行
- [ ] 论文素材包：`v23_data_stats.md`（数据统计 + dedup 报告 + 配额图）

### 8.3 失败回滚

| 情况 | 回滚方案 |
|---|---|
| v23 PPL 没破 4.39 | 保留 v22a 不动，v23 论文写"扩量未改善，作为反例分析" |
| 训练 NaN | LR 1.0e-4 → 5.0e-5 重试 |
| 内存 OOM | `CHUNK_SIZE` 1M → 500K, `num_perm` 128 → 64 |
| 下载某源失败 | 该源标 "skipped", 70% agentic 按比例重分 |
| disk full | 立即停，删 `D:/tmp_v23_dl/`, 从最近 checkpoint 续跑 |

---

## 9. 边界 case 清单

| Case | 期望处理 |
|---|---|
| Fable 5 某源只有 100 文档 | 该源跳过，70% agentic 按比例重分剩余源 |
| github-code 抽不到 20GB | 用尽即停，code 份额按实际抽到量报告 |
| wikipedia zh 子集不存在 | 跳过 zh，仅 en，wiki 总配额减半 |
| MinHash 后 agentic 总字符 < 49GB | 报警，继续训练（paper 写"配额未达成"） |
| T=512 拼出 pack_len=1 的退化 pack | 跳过 < 50 chars 的 pack |
| v22a 1895 里有 doc 在 Fable 5 中也存在 | 跨域去重时删 Fable 5 副本，保 local（v22a 是 anchor） |
| HumanEval-X 评估 0% pass@1 | 正常，475M 训量下不指望 code 能力 |
| 训练中途断电 | `train.py` 每 500 step save checkpoint, `--resume-from` 重启 |

---

## 10. 估时汇总

| 阶段 | 估时 | Buffer |
|---|---|---|
| Stage 0 schema 探查 | 30 min | 1.5x |
| Stage 1a Fable 5 下载 | 4-8 h | 1.5x |
| Stage 1b streaming 下载 | 4-8 h | 1.5x |
| Stage 2 clean | 1-2 h | 1.3x |
| Stage 3 dedup | 4-8 h | 1.5x |
| Stage 4 pack | 1-2 h | 1.3x |
| Stage 5 train | 12 min × N | 2x |
| Stage 5 eval | 30 min | 1.5x |
| **总墙钟** | **15-30 h** | (1.5x 累计) |

实际工程时间：2-3 天（白天黑夜跑，cron 安排）。

---

## 11. 与现有 v23 build 脚本的关系

已存在 `crystalllm/build_v23_data.py`，做**本地 2305 sessions 滑窗切分**（W=5000 → 9K 样本, 16MB parquet）。

**本 spec 的关系**：
- 该脚本对应 **Step 0 后的"本地 anchor"**，其 `v23_train.parquet` 直接作为 Step 4 的 4 路输入之一
- 不修改 `build_v23_data.py`，仅在新 spec 的 Step 4 中读取其输出
- 最终 v23 paper 应说明 "1893 sessions local + 100G external combined"

---

## 12. 开放问题（已通过 brainstorming 解决）

| 问题 | 决策 |
|---|---|
| 数据域 | 70% agentic + 20% code + 10% wiki |
| Fable 5 源大小未知 | **先探后采**（Step 0 必跑） |
| HumanEval-X 角色 | 仅 pass@1 sanity，**不入 PPL** |
| 配额基准 | **按清洗后字符** |
| Tokenizer | 保持 char-level 2261 |
| T | 512 |
| 限速 | 不限速，cron 由 user schedule |
| v23 PPL 目标 | **< v22a 4.39（不爆即可）** |
| Val 集 | v22a 1893+210 永远不动 |
| 失败回滚 | PPL 没破 4.39 → 保留 v22a 不动，作为反例分析 |

---

**下一步**: spec 复审 → writing-plans skill 写实现计划 → 进入 v23 实施。
