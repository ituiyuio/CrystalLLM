"""
Momo 终端 monitor: 另开一个终端跑这个, 实时看训练状态。
读 D:/CrystaLLM/model_pool_progress.json + nvidia-smi, 每秒刷新一次。
"""
import os
import sys
import time
import json
import subprocess
from pathlib import Path

# === v4-fix: F20c — Windows GBK stdout reconfigure (emoji 兜不住) ===
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROGRESS_FILE = Path("D:/CrystaLLM/model_pool_progress.json")
REFRESH_S = 1.0


def read_progress():
    """读 progress JSON, 返回 None 如果文件不存在"""
    try:
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def nvidia_smi():
    """返回 (util%, mem_used_mib, mem_total_mib) 或 Nones"""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            stderr=subprocess.DEVNULL, timeout=2
        ).decode().strip()
        parts = [p.strip() for p in out.split(',')]
        return {
            'util': int(parts[0]),
            'mem_used_mib': int(parts[1]),
            'mem_total_mib': int(parts[2]),
            'temp_c': int(parts[3]) if parts[3] != '[Not Supported]' else 0,
            'power_w': float(parts[4]) if parts[4] != '[Not Supported]' else 0.0,
        }
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def fmt_eta(seconds):
    if seconds is None or seconds < 0:
        return "--:--"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def fmt_bytes(mib):
    if mib >= 1024:
        return f"{mib/1024:.1f}G"
    return f"{mib}M"


def draw_bar(pct, width=20):
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def clear():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def main():
    print(f"🔍 Mavis monitor — watching {PROGRESS_FILE}")
    print(f"   Press Ctrl+C to quit")
    print(f"   Refreshing every {REFRESH_S}s")
    time.sleep(1.5)

    last_progress = None
    try:
        while True:
            clear()
            prog = read_progress()
            gpu = nvidia_smi()

            # Header
            print("═" * 60)
            print(f"  🚀 Mavis Training Monitor  •  {time.strftime('%H:%M:%S')}")
            print("═" * 60)

            # GPU section
            if gpu:
                bar = draw_bar(gpu['util'], 20)
                mem_pct = 100 * gpu['mem_used_mib'] / gpu['mem_total_mib'] if gpu['mem_total_mib'] > 0 else 0
                print(f"\n  GPU RTX 5090")
                print(f"  util: [{bar}] {gpu['util']:3d}%   temp: {gpu['temp_c']}°C   power: {gpu['power_w']:5.0f}W")
                print(f"  mem:  {fmt_bytes(gpu['mem_used_mib']):>6s} / {fmt_bytes(gpu['mem_total_mib'])}  ({mem_pct:.0f}%)")
            else:
                print(f"\n  GPU: nvidia-smi 读取失败")

            # Training section
            print(f"\n  {'─'*56}")
            if prog is None:
                print(f"  ⏳ 等待训练启动 (model_pool_progress.json 不存在)")
                print(f"      提示: 训练启动后每 100 步写一次进度")
            else:
                # Compute progress
                pct = 100.0 * prog['step'] / max(1, prog['total'])
                bar = draw_bar(pct, 30)
                sps = prog['sps']
                ms_per_step = 1000.0 / sps if sps > 0 else 0

                print(f"  📈 Training Progress")
                print(f"  step: {prog['step']:>6d} / {prog['total']:<6d}  [{bar}] {pct:5.1f}%")
                print(f"  speed: {sps:.2f} steps/s   ({ms_per_step:.1f} ms/step)")
                print(f"  elapsed: {fmt_eta(prog['elapsed_s'])}   ETA: {fmt_eta(prog['eta_s'])}")

                # State
                print(f"\n  State")
                print(f"  S_norm:        {prog['s_norm']:.4f}")
                print(f"  active blocks: {prog['active_logical']:3d} / {prog['num_blocks']}")
                print(f"  Top_K:         {prog['top_k']}")
                print(f"  T_batch:       {prog['t_batch']}")

                # Progress delta
                if last_progress and prog['step'] > last_progress['step']:
                    delta_step = prog['step'] - last_progress['step']
                    delta_time = prog['elapsed_s'] - last_progress['elapsed_s']
                    if delta_time > 0:
                        recent_sps = delta_step / delta_time
                        print(f"\n  Δ (since last refresh): {delta_step} steps in {delta_time:.1f}s "
                              f"= {recent_sps:.2f} steps/s")

            last_progress = prog if prog else last_progress

            print(f"\n{'─'*60}")
            print(f"  next refresh in {REFRESH_S}s   |   Ctrl+C to quit")
            time.sleep(REFRESH_S)

    except KeyboardInterrupt:
        print("\n👋 monitor stopped")


if __name__ == "__main__":
    main()
