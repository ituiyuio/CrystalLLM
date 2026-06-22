"""Phase-transition checkpoint save/load helper for Exp 17.

Saves model state_dict + minimal metadata to enable later evaluation
without re-training. Metadata includes step, config_label, val_ppl,
and optional CMT-specific fields (imag_energy_ratio).
"""
import torch
import torch.nn as nn


def save_phase_transition_ckpt(
    model: nn.Module,
    path: str,
    step: int,
    config_label: str,
    val_ppl: float,
    imag_energy_ratio: float = None,
) -> None:
    """Save model + metadata to a single .pt file.

    Args:
        model: PyTorch model (state_dict will be saved)
        path: target file path (will overwrite)
        step: training step at which checkpoint was taken
        config_label: "A0" | "A1" | "A2" | "V49_50M"
        val_ppl: validation PPL at this checkpoint
        imag_energy_ratio: optional, only for CMT (input/output imag magnitude ratio)
    """
    metadata = {
        "step": step,
        "config": config_label,
        "val_ppl": val_ppl,
    }
    if imag_energy_ratio is not None:
        metadata["imag_energy_ratio"] = imag_energy_ratio
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def load_phase_transition_ckpt(model: nn.Module, path: str) -> dict:
    """Load checkpoint into model. Returns metadata dict.

    Args:
        model: PyTorch model (state_dict will be overwritten)
        path: path to .pt file

    Returns:
        metadata dict with keys: step, config, val_ppl, [imag_energy_ratio]
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    return ckpt["metadata"]
