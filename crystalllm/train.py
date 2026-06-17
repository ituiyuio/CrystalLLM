"""train.py — CrystaLLM decoder 训练入口 (V23_MODE flag)

默认 v22a 模式: 走 v22a 数据 + 128 字符
V23 模式 (env CRYSTALLM_V23=1): 走 extended_v23 数据 + 512 字符
val 强制用 v22a_val.parquet 做 anchor (跨版本可比)

用法:
    python train.py                    # v22a 模式
    CRYSTALLM_V23=1 python train.py    # v23 模式
"""
import os
from pathlib import Path

# === V23_MODE flag (env var 切换) ===
V23_MODE = os.environ.get("CRYSTALLM_V23", "0") == "1"

DATA = Path("crystalllm/data/processed")

if V23_MODE:
    TRAIN_PATH = DATA / "extended_v23.parquet"
    VAL_PATH = DATA / "v22a_val.parquet"  # 强制 anchor
    MAX_SEQ_LEN = 512
    INIT_FROM = Path("crystalllm/v22_decoder.pt")
    OUT_PATH = Path("crystalllm/v23_decoder.pt")
else:
    TRAIN_PATH = DATA / "v22a_train.parquet"
    VAL_PATH = DATA / "v22a_val.parquet"
    MAX_SEQ_LEN = 128
    INIT_FROM = None
    OUT_PATH = Path("crystalllm/v22a_decoder.pt")


# === 主入口 (实际训练委托给对应脚本) ===
def main():
    print(f"[train] V23_MODE={V23_MODE}")
    print(f"[train] TRAIN_PATH={TRAIN_PATH}")
    print(f"[train] VAL_PATH={VAL_PATH}")
    print(f"[train] MAX_SEQ_LEN={MAX_SEQ_LEN}")
    if V23_MODE:
        # 委托给 proto_v23_decoder.py (Step 5)
        import subprocess
        subprocess.check_call(["python", "crystalllm/proto_v23_decoder.py"])
    else:
        # 委托给 train_v22_decoder.py
        import subprocess
        subprocess.check_call(["python", "crystalllm/train_v22_decoder.py"])


if __name__ == "__main__":
    main()
