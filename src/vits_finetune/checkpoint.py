"""Checkpoint save/load for VITS fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save model/optimizer state plus training progress to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'epoch': epoch,
    }
    if extra is not None:
        payload['extra'] = extra
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load a checkpoint written by ``save_checkpoint`` into ``model`` (and ``optimizer``)."""
    path = Path(path)
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint
