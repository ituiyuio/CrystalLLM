import time
import pytest
from experiments.v49_pre.metrics import MetricsCollector, format_metrics


def test_metrics_collector_initial_state():
    """初始状态应为空."""
    mc = MetricsCollector()
    assert mc.tokens_processed == 0
    assert mc.elapsed_time == 0.0
    assert mc.peak_memory_mb == 0.0


def test_metrics_collector_record_step():
    """record_step 应累积 tokens_processed 和 elapsed_time."""
    mc = MetricsCollector()
    mc.start()
    time.sleep(0.01)  # 10ms
    mc.record_step(tokens=512)
    mc.record_step(tokens=512)
    assert mc.tokens_processed == 1024
    assert mc.elapsed_time >= 0.01


def test_format_metrics_returns_string():
    """format_metrics 应返回格式化字符串."""
    metrics = {"tokens_per_sec": 1000.0, "peak_memory_mb": 1024.5, "val_ppl": 2.5}
    result = format_metrics(metrics)
    assert "tokens/sec: 1000.00" in result
    assert "peak_mem: 1024.50 MB" in result
    assert "val_ppl: 2.5000" in result
