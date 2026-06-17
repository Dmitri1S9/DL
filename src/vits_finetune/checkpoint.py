"""Checkpoint save/load for VITS fine-tuning.
"""

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
    discriminator: torch.nn.Module | None = None,
    disc_optimizer: torch.optim.Optimizer | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save model/optimizer (+ optional discriminator) and training progress."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'epoch': epoch,
    }
    if discriminator is not None:
        payload['discriminator'] = discriminator.state_dict()
    if disc_optimizer is not None:
        payload['disc_optimizer'] = disc_optimizer.state_dict()
    if extra is not None:
        payload['extra'] = extra
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    discriminator: torch.nn.Module | None = None,
    disc_optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load a checkpoint written by ``save_checkpoint`` into the given modules."""
    path = Path(path)
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if discriminator is not None and 'discriminator' in checkpoint:
        discriminator.load_state_dict(checkpoint['discriminator'])
    if disc_optimizer is not None and 'disc_optimizer' in checkpoint:
        disc_optimizer.load_state_dict(checkpoint['disc_optimizer'])
    return checkpoint
