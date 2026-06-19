# nanochat — Karpathy nanochat 简化移植

> 单 GPU / 单文件 / 单目录的 GPT 预训练脚手架，从 [Karpathy nanochat](https://github.com/karpathy/nanochat) cherry-pick 并精简而来。

## 文件

| 文件 | 作用 | 用法 |
|---|---|---|
| `prepare.py` | 数据准备：下载 shards + 训练 BPE tokenizer + 缓存到 `~/.cache/autoresearch/` | `python prepare.py --num-shards 8`（仅下载 8 个 shard 做测试） |
| `train.py` | GPT 预训练入口：单 forward、AdamW、cosine schedule、MFU 报告 | `uv run train.py` |

## 与 CrystaLLM 主项目的关系

本目录是**独立可运行的最小脚手架**，用作：
1. **基础设施参考**：FA3 / rotary / GQA / KV-cache 等实现细节可借鉴到 CrystaLLM decoder
2. **性能 baseline**：与 CrystaLLM 的扩散定位 + AR 方案做参数/速度/质量的横向对比
3. **学习路径**：阅读这份 1000 行的"教科书式"实现能快速建立 LLM 训练的全局观

训练时 `train.py` 通过 `from prepare import ...` 引用 `prepare.py`，**两个文件必须放在同一目录**。

## 与上游的差异

- 删除多 GPU 分布式（DeepSpeed-style）逻辑，只保留单 GPU 训练循环
- 删除 wandb / tensorboard 集成，只保留本地 stdout + JSON 日志
- 删除 SFT / RLHF 等下游任务模块，只保留 pretrain
- 保留：RoPE、GQA、μP init、cosine schedule、MFU 计算、BPE tokenizer

## 致谢

基于 Karpathy 的 [nanochat](https://github.com/karpathy/nanochat) 项目精简而来。原项目使用 MIT 协议，本仓库保留上游版权声明；本目录下的文件不修改原协议。
本目录的目的是**教学 + 借鉴**，不是为了替代上游。

## 协议说明

- `train.py` 和 `prepare.py`：源自 [karpathy/nanochat](https://github.com/karpathy/nanochat)，**MIT License**（保留上游版权）
- `README.md`：本仓库原创，**Apache License 2.0**
- 整个仓库其余代码：**Apache License 2.0**

## 依赖

```
uv add torch kernels rustbpe tiktoken pyarrow requests
```