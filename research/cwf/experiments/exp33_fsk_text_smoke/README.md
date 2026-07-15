# Exp 33: CWF × FSK Text-Wave Smoke

**Date**: 2026-07-15
**Status**: Active (v51 Phase 4.2 smoke)
**Spec**: `docs/superpowers/specs/2026-07-15-脉冲波-text-wave-design.md`

## What it does

Maps text to FSK-modulated complex waveforms, runs them through CWF single block, decodes back. Tests whether CWF can learn the "shift-by-1" structure of an 8-char sequence (the foundation for any autoregressive LM in this representation).

## How to run

```bash
# 5 seeds × 1000 steps (full, ~5 min CPU)
python research/cwf/experiments/exp33_fsk_text_smoke/exp33_fsk_text_smoke.py

# Quick smoke (2 seeds × 100 steps, ~60s)
python research/cwf/experiments/exp33_fsk_text_smoke/exp33_fsk_text_smoke.py --seeds 42 123 --steps 100

# Run tests
python -m pytest research/cwf/experiments/exp33_fsk_text_smoke/tests/ -v
```

## Verdict thresholds (Nyquist-aware, 88.4% theoretical ceiling)

| Char accuracy | Verdict | Action |
|---------------|---------|--------|
| > 80% | GO | 扩 vocab (8→16→32), 写 Phase 4.2 完整版 |
| 50-80% + ratio < 0.5 | PARTIAL | 进 Phase 4.3 (扩展) |
| 50-80% + ratio ≥ 0.5 | NEUTRAL | 归档 |
| < 50% | DEAD | 归档 "CWF + 文本脉冲波" 路线 |
