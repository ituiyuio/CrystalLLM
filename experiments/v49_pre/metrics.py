"""训练 metrics 采集: tokens/sec, peak memory."""
import time
from typing import Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MetricsCollector:
    """训练过程中的 metrics 采集器."""

    def __init__(self):
        self.tokens_processed = 0
        self.elapsed_time = 0.0
        self.peak_memory_mb = 0.0
        self._start_time: Optional[float] = None

    def start(self):
        """开始计时."""
        self._start_time = time.time()

    def record_step(self, tokens: int):
        """记录一个 step 的 tokens 数."""
        if self._start_time is None:
            raise RuntimeError("MetricsCollector.start() not called")
        self.tokens_processed += tokens
        self.elapsed_time = time.time() - self._start_time

    def update_peak_memory(self):
        """更新 peak GPU memory (MB)."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            self.peak_memory_mb = 0.0
            return
        current = torch.cuda.max_memory_allocated() / (1024 * 1024)
        self.peak_memory_mb = max(self.peak_memory_mb, current)

    @property
    def tokens_per_sec(self) -> float:
        if self.elapsed_time == 0:
            return 0.0
        return self.tokens_processed / self.elapsed_time

    def to_dict(self) -> dict:
        """导出为 dict."""
        return {
            "tokens_per_sec": self.tokens_per_sec,
            "peak_memory_mb": self.peak_memory_mb,
            "total_tokens": self.tokens_processed,
            "elapsed_time": self.elapsed_time,
        }


def format_metrics(metrics: dict) -> str:
    """格式化 metrics 为可读字符串."""
    parts = []
    if "tokens_per_sec" in metrics:
        parts.append(f"tokens/sec: {metrics['tokens_per_sec']:.2f}")
    if "peak_memory_mb" in metrics:
        parts.append(f"peak_mem: {metrics['peak_memory_mb']:.2f} MB")
    if "val_ppl" in metrics:
        parts.append(f"val_ppl: {metrics['val_ppl']:.4f}")
    if "wall_clock_sec" in metrics:
        parts.append(f"wall_clock: {metrics['wall_clock_sec']:.1f}s")
    return " | ".join(parts)
