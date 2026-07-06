"""Exp05 scientific controls — addresses Challenge 1 (capacity) & Challenge 3 (seed).

Control 1 (capacity): real d=64 Stage A. 若 complex d=32 (119K params) 仍优于
  real d=64 (~240K, 2× params), 则 Stage A 的 2.8× 优势不是容量, 是相位.
Control 2/3 (seed): complex/real Stage B seed=123, 加载 d=32-s42 tokenizer.
  若 step-1000 complex-beats-real 复现, 则 0.155 nat 不是单 seed 噪声.

Challenge 2 (初始 loss 对等) & Challenge 4 (梯度衰减) 已从现有数据证伪, 不重跑.
"""
import sys
import shutil
import argparse
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from research.cwf.experiments.exp05_wave_autoencoder.wave_autoencoder import (
    load_data, build_model, run_stage, RESULTS_DIR,
)


def make_args(mode, stage, d, seed):
    return argparse.Namespace(
        mode=mode, seed=seed, stage=stage, steps=3000, d_model=d,
        modes=16, n_layers=2, seq_len=256, stride=4, batch_size=32,
        peak_lr=3e-4, warmup=100, log_every=200,
        eval_steps=[200, 500, 1000, 2000, 3000],
    )


def backup_orig(name):
    src = RESULTS_DIR / f"{name}.json"
    dst = RESULTS_DIR / f"{name}_d32_s42.json"
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)
        print(f"[backup] {src.name} -> {dst.name}")


def main():
    train_ids, val_ids = load_data()

    # 备份 d=32 s42 原始结果 (控制实验会覆盖 stage 输出)
    for name in ["complex_stagea", "complex_stageb", "real_stagea", "real_stageb"]:
        backup_orig(name)
    for mode in ["complex", "real"]:
        src = RESULTS_DIR / f"tokenizer_{mode}.pt"
        dst = RESULTS_DIR / f"tokenizer_{mode}_d32_s42.pt"
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)
            print(f"[backup] {src.name} -> {dst.name}")

    # --- Control 1: real d=64 Stage A (容量过校正) ---
    print("\n### CONTROL 1: real d=64 Stage A (capacity over-correct) ###")
    args = make_args("real", "a", d=64, seed=42)
    torch.manual_seed(42)
    model = build_model(args)
    run_stage(model, "a", args, train_ids, val_ids,
              ckpt_path=str(RESULTS_DIR / "tokenizer_real.pt"))
    shutil.move(str(RESULTS_DIR / "real_stagea.json"),
                str(RESULTS_DIR / "control_real_stagea_d64_s42.json"))
    shutil.move(str(RESULTS_DIR / "tokenizer_real.pt"),
                str(RESULTS_DIR / "control_tokenizer_real_d64_s42.pt"))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 恢复 d=32 s42 real tokenizer 供 seed-variance Stage B 使用
    shutil.copy(str(RESULTS_DIR / "tokenizer_real_d32_s42.pt"),
                str(RESULTS_DIR / "tokenizer_real.pt"))

    # --- Control 2: complex Stage B seed=123 ---
    print("\n### CONTROL 2: complex Stage B seed=123 (load d=32 s42 tokenizer) ###")
    args = make_args("complex", "b", d=32, seed=123)
    torch.manual_seed(123)
    model = build_model(args)
    run_stage(model, "b", args, train_ids, val_ids,
              ckpt_path=str(RESULTS_DIR / "tokenizer_complex.pt"))
    shutil.move(str(RESULTS_DIR / "complex_stageb.json"),
                str(RESULTS_DIR / "control_complex_stageb_s123.json"))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Control 3: real Stage B seed=123 ---
    print("\n### CONTROL 3: real Stage B seed=123 (load d=32 s42 tokenizer) ###")
    args = make_args("real", "b", d=32, seed=123)
    torch.manual_seed(123)
    model = build_model(args)
    run_stage(model, "b", args, train_ids, val_ids,
              ckpt_path=str(RESULTS_DIR / "tokenizer_real.pt"))
    shutil.move(str(RESULTS_DIR / "real_stageb.json"),
                str(RESULTS_DIR / "control_real_stageb_s123.json"))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 恢复 d=32 s42 Stage B 结果作为 canonical (被 control 2/3 覆盖了)
    for name in ["complex_stageb", "real_stageb"]:
        src = RESULTS_DIR / f"{name}_d32_s42.json"
        dst = RESULTS_DIR / f"{name}.json"
        if src.exists():
            shutil.copy(src, dst)
            print(f"[restore] {src.name} -> {dst.name}")

    print("\n[done] all controls complete")


if __name__ == "__main__":
    main()
