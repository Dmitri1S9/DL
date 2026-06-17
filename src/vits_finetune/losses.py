"""Loss functions for VITS fine-tuning.

Both losses operate on tensors returned by
``model.VitsFinetuneModel.forward_train``.
"""

from __future__ import annotations

import torch


def recon_loss(predicted_mel: torch.Tensor, target_mel: torch.Tensor) -> torch.Tensor:
    """L1 reconstruction loss between predicted and target mel-spectrograms."""
    return torch.abs(predicted_mel - target_mel).mean()


def kl_loss(
    z_p: torch.Tensor,
    posterior_log_stddev: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_log_stddev: torch.Tensor,
    z_mask: torch.Tensor,
) -> torch.Tensor:
    posterior_var = torch.exp(2 * posterior_log_stddev)
    prior_var = torch.exp(2 * prior_log_stddev)
    kl = (
        prior_log_stddev
        - posterior_log_stddev
        - 0.5
        + 0.5 * (posterior_var + (z_p - prior_mean) ** 2) / prior_var
    )
    return (kl * z_mask).sum() / z_mask.sum()


def discriminator_loss(
    d_real_outputs: list[torch.Tensor],
    d_fake_outputs: list[torch.Tensor],
) -> torch.Tensor:
    """LS-GAN discriminator loss: push real -> 1, fake -> 0."""
    loss = torch.zeros((), device=d_real_outputs[0].device)
    for d_real, d_fake in zip(d_real_outputs, d_fake_outputs, strict=True):
        loss = loss + torch.mean((1 - d_real) ** 2) + torch.mean(d_fake ** 2)
    return loss


def generator_adv_loss(d_fake_outputs: list[torch.Tensor]) -> torch.Tensor:
    """LS-GAN generator loss: push the discriminator's fake score -> 1."""
    loss = torch.zeros((), device=d_fake_outputs[0].device)
    for d_fake in d_fake_outputs:
        loss = loss + torch.mean((1 - d_fake) ** 2)
    return loss


def feature_matching_loss(
    fmaps_real: list[list[torch.Tensor]],
    fmaps_fake: list[list[torch.Tensor]],
) -> torch.Tensor:
    """L1 between real and fake discriminator feature maps (real detached)."""
    loss = torch.zeros((), device=fmaps_fake[0][0].device)
    for real_maps, fake_maps in zip(fmaps_real, fmaps_fake, strict=True):
        for real_map, fake_map in zip(real_maps, fake_maps, strict=True):
            loss = loss + torch.mean(torch.abs(real_map.detach() - fake_map))
    return loss


if __name__ == '__main__':
    predicted_mel = torch.randn(2, 80, 32)
    target_mel = torch.randn(2, 80, 32)
    print('recon_loss:', recon_loss(predicted_mel, target_mel).item())

    z_p = torch.randn(2, 192, 50)
    log_std = torch.randn(2, 192, 50) * 0.1
    prior_mean = torch.randn(2, 192, 50)
    z_mask = torch.ones(2, 1, 50)
    val = kl_loss(z_p, log_std, prior_mean, log_std, z_mask)
    print('kl_loss:', val.item())
