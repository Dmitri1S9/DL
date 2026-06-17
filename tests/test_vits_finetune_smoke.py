"""Offline CPU smoke tests for the VITS GAN fine-tuning machinery.

These never download the real VITS model or dataset — they exercise the
discriminator, the adversarial losses, and the GAN update mechanics on tiny
synthetic tensors, so they run in seconds and prove the wiring is sound.
"""

from __future__ import annotations

import torch

from vits_finetune.discriminator import VitsDiscriminator


def test_discriminator_shapes_and_finite():
    disc = VitsDiscriminator()
    real = torch.randn(2, 1, 8192)
    fake = torch.randn(2, 1, 8192)
    d_real, d_fake, fmap_real, fmap_fake = disc(real, fake)
    # 5 period + 3 scale sub-discriminators
    assert len(d_real) == len(d_fake) == 8
    assert len(fmap_real) == len(fmap_fake) == 8
    for out in d_real + d_fake:
        assert out.shape[0] == 2
        assert torch.isfinite(out).all()
