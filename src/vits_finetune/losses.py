from __future__ import annotations

import torch


def recon_loss(predicted_mel, target_mel):
    return torch.abs(predicted_mel - target_mel).mean()


def kl_loss(
    z_p: torch.Tensor,
    posterior_log_stddev: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_log_stddev: torch.Tensor,
    z_mask: torch.Tensor,
) -> torch.Tensor:
    prior_var = torch.exp(2 * prior_log_stddev)
    kl = (
        prior_log_stddev
        - posterior_log_stddev
        - 0.5
        + 0.5 * (z_p - prior_mean) ** 2 / prior_var
    )
    return (kl * z_mask).sum() / z_mask.sum()


def discriminator_loss(
    real_outs: list[torch.Tensor], fake_outs: list[torch.Tensor]
) -> torch.Tensor:
    loss = 0.0
    for real_out, fake_out in zip(real_outs, fake_outs, strict=True):
        loss += torch.mean((real_out - 1) ** 2) + torch.mean(fake_out**2)
    return loss


def generator_adversarial_loss(fake_outs: list[torch.Tensor]) -> torch.Tensor:
    loss = 0.0
    for fake_out in fake_outs:
        loss += torch.mean((fake_out - 1) ** 2)
    return loss


def feature_matching_loss(
    real_fmaps: list[torch.Tensor], fake_fmaps: list[torch.Tensor]
) -> torch.Tensor:
    loss = 0.0
    for real_branch, fake_branch in zip(real_fmaps, fake_fmaps, strict=True):
        for real_f, fake_f in zip(real_branch, fake_branch, strict=True):
            loss += torch.mean(torch.abs(real_f - fake_f))
    return loss
