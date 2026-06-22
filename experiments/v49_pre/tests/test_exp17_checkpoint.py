"""Unit tests for Exp 17 checkpoint save/load helper."""
import os
import tempfile
import torch
import torch.nn as nn

from experiments.v49_pre.exp17_checkpoint import (
    save_phase_transition_ckpt,
    load_phase_transition_ckpt,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)


def test_save_and_load_roundtrip():
    """Save model state, load into new instance, verify state matches."""
    model = TinyModel()
    with torch.no_grad():
        model.linear.weight.fill_(0.42)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "tiny_step1000.pt")
        save_phase_transition_ckpt(model, path, step=1000, config_label="A0", val_ppl=2.5)
        new_model = TinyModel()
        loaded_meta = load_phase_transition_ckpt(new_model, path)
        assert torch.allclose(new_model.linear.weight, model.linear.weight)
        assert loaded_meta["step"] == 1000
        assert loaded_meta["config"] == "A0"
        assert loaded_meta["val_ppl"] == 2.5


def test_save_includes_imag_energy_field():
    """Saved checkpoint should support optional imag_energy_ratio field."""
    model = TinyModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "tiny_step2000.pt")
        save_phase_transition_ckpt(
            model, path, step=2000, config_label="A0", val_ppl=1.5, imag_energy_ratio=5966.86,
        )
        loaded_meta = load_phase_transition_ckpt(model, path)
        assert "imag_energy_ratio" in loaded_meta
        assert abs(loaded_meta["imag_energy_ratio"] - 5966.86) < 1e-6
