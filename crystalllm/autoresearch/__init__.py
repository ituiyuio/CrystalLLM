# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""autoresearch — CrystaLLM 训练研究脚手架

按功能划分的 6 个子包：

- pipeline/      数据准备（build / collect / download / extract / cache / dedup / clean / pack）
- training/      训练脚本（train.py 入口 + train_v* 版本 + proto_v* 原型 + v*_model.py 模型定义）
- evaluation/    评估脚本（eval_v* 评测 + debug_v* 调试 + check_* 检查 + smoke_v* 冒烟）
- benchmarks/    基准测试（bench_*/benchmark_*/speed_*）
- tests/         内联 sanity 测试（test_v*_speed.py / test_v36_model.py / test_v36_warmstart.py 等）
- nanochat/      Karpathy nanochat 简化移植（train.py + prepare.py，独立子项目）

注：除了 ``nanochat/`` 子包是从 Karpathy nanochat cherry-pick 简化移植外，
    其它子包都是 CrystaLLM 项目自身的实验脚本，围绕"扩散定位 + 自回归解码"范式独立编写。
    命名上借鉴了 nanochat 的"短平快训练循环"思想。
"""