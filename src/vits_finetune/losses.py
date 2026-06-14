"""Loss functions for VITS fine-tuning.

Both losses operate on tensors returned by
``model.VitsFinetuneModel.forward_train``.
"""

from __future__ import annotations

import torch


recon_loss = lambda predicted_mel, target_mel : \
        torch.abs(predicted_mel - target_mel).mean()


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