#!/usr/bin/env python3
"""organize_to_versions.py — 把 autoresearch/{training,evaluation,pipeline,benchmarks}/ 下的 .py
按版本号+功能分类移动到 versions/{ver}/{func}/。

用法:
    python crystalllm/scripts/organize_to_versions.py          # 执行移动
    python crystalllm/scripts/organize_to_versions.py --dry-run # 只打印计划，不移动

设计要点:
  1. 版本号从文件名第一个 _v<数字> token 提取（按 _ 分词）
  2. 195 / 215 三位数映射为 v19_5 / v21_5（目录约定）
  3. 子版本（v19b/v15_3/v34c）回落主版本（v19/v15/v34），不新建目录
  4. 无版本号进 versions/_common/{func}/
  5. 跳过 autoresearch/{nanochat,tests,__pycache__} 和所有 __init__.py
  6. 幂等：目标已存在则跳过，不覆盖

幂等: 重跑安全，已移动的文件不会再处理。
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

# 路径: scripts/organize_to_versions.py -> crystalllm/scripts/ -> crystalllm/
SCRIPT_DIR = Path(__file__).resolve().parent
CRYSTALLM = SCRIPT_DIR.parent
AUTORESEARCH = CRYSTALLM / "autoresearch"
VERSIONS = CRYSTALLM / "versions"

# autoresearch 功能子目录 -> versions 下目标功能子目录
FUNC_DIRS = {
    "training": "training",
    "evaluation": "evaluation",
    "pipeline": "pipeline",
    "benchmarks": "benchmarks",
}

SKIP_FILES = {"__init__.py"}


def extract_version(filename: str) -> str | None:
    """从文件名提取版本号。

    规则:
      - 第一个 _v<数字>[a-d]? token 即为版本
      - 195/215 三位数（末位为 5）→ v19_5 / v21_5
      - 无 _v<数字> token → None（无版本号）

    例:
      proto_v2.py                → 'v2'
      proto_v195_pure_ar.py      → 'v19_5'
      proto_v19b_decoder_ft.py   → 'v19b'
      eval_v34b_shared.py        → 'v34b'
      debug_v29_two_verifiers.py → 'v29'
      prototype.py               → None
    """
    name = filename[:-3] if filename.endswith(".py") else filename
    for tok in name.split("_"):
        if not tok.startswith("v"):
            continue
        m = re.match(r"^v(\d+)([a-d])?$", tok)
        if m is None:
            continue
        base, suffix = m.group(1), m.group(2) or ""
        # 195 → 19_5: 三位数末位 5 表示 .5 子版本
        if len(base) == 3 and base[2] == "5":
            base = base[:2] + "_" + base[2]
        return "v" + base + suffix
    return None


def resolve_version_dir(ver: str) -> str | None:
    """把提取的版本映射到有效目录；无法映射时返回 None（调用方放 _common）。

    versions/ 已有目录: v3-v22, v19_5, v20a, v21_5, v22a, v23_BAD,
                         v23-v36, v26_5, v28_5, v34a, v34b, v34d。
    规则:
      - 目录已存在 → 直接返回
      - 整数版本（v2、v10）但目录不存在 → CREATE 新目录（返回 ver）
      - 子版本（v19b、v34c）→ 回落主版本；主版本也不存在 → None
    """
    if (VERSIONS / ver).is_dir():
        return ver
    # 整数版本（无后缀字母/数字）→ 创建新目录
    if re.match(r"^v\d+$", ver):
        return ver
    # 子版本：尝试回落主版本
    m = re.match(r"^(v\d+)([a-d]|_\d+|_BAD)?$", ver)
    if m and m.group(2):
        base_ver = m.group(1)
        if (VERSIONS / base_ver).is_dir():
            return base_ver
    return None  # 无法映射 → 放进 _common/


def plan_moves() -> list[tuple[Path, Path, str]]:
    """返回 (源, 目标, 版本标签) 列表，但不执行移动。"""
    plan = []
    for func_dir, target_func in FUNC_DIRS.items():
        src_dir = AUTORESEARCH / func_dir
        if not src_dir.is_dir():
            continue
        for py_file in sorted(src_dir.glob("*.py")):
            if py_file.name in SKIP_FILES:
                continue
            ver = extract_version(py_file.name)
            if ver is None:
                target_dir = VERSIONS / "_common" / target_func
                ver_label = "_common"
            else:
                actual = resolve_version_dir(ver)
                if actual is None:
                    # 无法映射到已知版本目录（如 v34c → v34 但 v34 不存在）
                    target_dir = VERSIONS / "_common" / target_func
                    ver_label = "_common"
                else:
                    target_dir = VERSIONS / actual / target_func
                    ver_label = actual
            plan.append((py_file, target_dir / py_file.name, ver_label))
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不移动")
    parser.add_argument("--verbose", "-v", action="store_true", help="打印每个文件的去向")
    args = parser.parse_args()

    if not AUTORESEARCH.is_dir():
        print(f"错误: {AUTORESEARCH} 不存在", file=sys.stderr)
        return 1
    if not VERSIONS.is_dir():
        print(f"错误: {VERSIONS} 不存在", file=sys.stderr)
        return 1

    plan = plan_moves()
    moved, skipped, errors = [], [], []

    for src, dst, ver_label in plan:
        if dst.exists():
            skipped.append((src, dst, "目标已存在"))
            continue
        if args.dry_run:
            print(f"[DRY] {src.relative_to(CRYSTALLM)} → {dst.relative_to(CRYSTALLM)}")
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            moved.append((src, dst, ver_label))
            if args.verbose:
                print(f"[MV]  {src.relative_to(CRYSTALLM)} → {dst.relative_to(CRYSTALLM)}")
        except Exception as e:
            errors.append((src, str(e)))

    by_ver = Counter(v for _, _, v in moved)
    print()
    print("=" * 50)
    if args.dry_run:
        print(f"[DRY-RUN] 将移动 {len(plan) - len(skipped)} 个文件")
    else:
        print(f"已移动: {len(moved)}")
        print(f"跳过:   {len(skipped)}（目标已存在）")
        print(f"失败:   {len(errors)}")

    if errors:
        print("\n失败:")
        for src, e in errors:
            print(f"  {src}: {e}")
        return 2

    if not args.dry_run and moved:
        print("\n按版本统计:")
        for v, c in sorted(by_ver.items(), key=lambda x: (x[0] == "_common", x[0])):
            print(f"  {v:<10} {c} 个")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())