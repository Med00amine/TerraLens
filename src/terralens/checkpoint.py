"""Loading the self-describing checkpoints written by `terralens.train`."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from terralens.models import build_model

REQUIRED_KEYS = frozenset(
    {"architecture", "num_classes", "class_names", "img_size", "mean", "std", "state_dict"}
)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    """Load a checkpoint safely (`weights_only=True`) and validate its contents."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    missing = REQUIRED_KEYS - set(ckpt)
    if missing:
        raise ValueError(f"{path} is not a TerraLens checkpoint (missing {sorted(missing)})")
    return ckpt


def model_from_checkpoint(ckpt: dict, device: str | torch.device = "cpu") -> nn.Module:
    """Rebuild the network described by `ckpt` and load its weights, in eval mode."""
    model = build_model(ckpt["architecture"], ckpt["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
