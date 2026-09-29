"""Device selection: ``auto`` picks cuda -> mps -> cpu."""

from __future__ import annotations

import torch


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lightning_accelerator(name: str = "auto") -> str:
    """Map our device names onto a Lightning ``accelerator`` string."""
    if name != "auto":
        return {"cuda": "gpu"}.get(name, name)
    dev = resolve_device("auto").type
    return {"cuda": "gpu"}.get(dev, dev)
