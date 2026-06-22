"""pytest 共享路径配置.

确保 tests/ 子目录的测试能 import 项目模块 (与现有 test_exp_runner.py 等保持兼容).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))