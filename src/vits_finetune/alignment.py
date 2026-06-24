"""Monotonic Alignment Search (MAS) for VITS training"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_NEG_INF = -1e9


def neg_cross_entropy(
    z_p: torch.Tensor, prior_mean: torch.Tensor, prior_log_stddev: torch.Tensor
) -> torch.Tensor:

    precision = torch.exp(-2 * prior_log_stddev)  # (B, flow_size, T_text)
    const_term = torch.sum(
        -0.5 * math.log(2 * math.pi) - prior_log_stddev, dim=1, keepdim=True
    )
    mean_sq_term = torch.sum(-0.5 * prior_mean**2 * precision, dim=1, keepdim=True)
    per_text_term = (const_term + mean_sq_term).transpose(1, 2)  # (B, T_text, 1)

    z_sq_term = -0.5 * torch.matmul(
        precision.transpose(1, 2), z_p**2
    )  # (B, T_text, T_frames)
    cross_term = torch.matmul((prior_mean * precision).transpose(1, 2), z_p)
    return per_text_term + z_sq_term + cross_term


@torch.no_grad()
def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

    value = neg_cent * mask
    batch_size, num_text, num_frames = value.shape
    device, dtype = value.device, value.dtype

    # direction[b, i, j]: True ("stay" on token i) vs False ("advance" from i-1).
    direction = torch.zeros(
        (batch_size, num_text, num_frames), dtype=torch.bool, device=device
    )
    # running[b, i]: best cumulative log-likelihood of frames seen so far, ending on token i.
    running = torch.zeros((batch_size, num_text), dtype=dtype, device=device)
    text_index = torch.arange(num_text, device=device, dtype=dtype).unsqueeze(0)

    for frame in range(num_frames):
        advance_from = F.pad(running, (1, 0), value=_NEG_INF)[:, :-1]
        stay = running >= advance_from
        best_prev = torch.where(stay, running, advance_from)
        direction[:, :, frame] = stay
        reachable = text_index <= frame
        running = torch.where(
            reachable,
            best_prev + value[:, :, frame],
            torch.full_like(running, _NEG_INF),
        )

    # Default padding to "stay" so the traceback can't wander once it reaches
    # the last valid frame of a (shorter, padded) example in the batch.
    direction = torch.where(mask.bool(), direction, torch.ones_like(direction))

    path = torch.zeros((batch_size, num_text, num_frames), dtype=dtype, device=device)
    token = (mask[:, :, 0].sum(dim=1).long() - 1).clamp(min=0)
    batch_index = torch.arange(batch_size, device=device)
    for frame in range(num_frames - 1, -1, -1):
        path[batch_index, token, frame] = 1.0
        token = (token + direction[batch_index, token, frame].long() - 1).clamp(min=0)

    return path * mask
