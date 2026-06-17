"""Offline CPU smoke tests for the VITS GAN fine-tuning machinery.

These never download the real VITS model or dataset — they exercise the
discriminator, the adversarial losses, and the GAN update mechanics on tiny
synthetic tensors, so they run in seconds and prove the wiring is sound.
"""

from __future__ import annotations

import torch

from vits_finetune.discriminator import VitsDiscriminator
from vits_finetune.losses import (
    discriminator_loss,
    feature_matching_loss,
    generator_adv_loss,
)


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


def test_adversarial_losses_finite():
    disc = VitsDiscriminator()
    real = torch.randn(2, 1, 8192)
    fake = torch.randn(2, 1, 8192)
    d_real, d_fake, fmap_real, fmap_fake = disc(real, fake)

    loss_d = discriminator_loss(d_real, d_fake)
    loss_g = generator_adv_loss(d_fake)
    loss_fm = feature_matching_loss(fmap_real, fmap_fake)

    for loss in (loss_d, loss_g, loss_fm):
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss >= 0


def test_gan_step_updates_both_sides():
    torch.manual_seed(0)
    disc = VitsDiscriminator()
    # tiny stand-in generator: noise -> 8192-sample wave
    generator = torch.nn.Sequential(torch.nn.Conv1d(1, 1, 3, padding=1))
    opt_d = torch.optim.AdamW(disc.parameters(), lr=1e-3)
    opt_g = torch.optim.AdamW(generator.parameters(), lr=1e-3)

    real = torch.randn(2, 1, 8192)
    noise = torch.randn(2, 1, 8192)

    g_before = next(generator.parameters()).clone()
    d_before = next(disc.parameters()).clone()

    # discriminator step
    fake = generator(noise)
    opt_d.zero_grad()
    d_real, d_fake, _, _ = disc(real, fake.detach())
    discriminator_loss(d_real, d_fake).backward()
    opt_d.step()

    # generator step
    opt_g.zero_grad()
    _, d_fake2, fmap_real, fmap_fake = disc(real, fake)
    loss_g = generator_adv_loss(d_fake2) + feature_matching_loss(fmap_real, fmap_fake)
    loss_g.backward()
    opt_g.step()

    assert not torch.equal(d_before, next(disc.parameters()))
    assert not torch.equal(g_before, next(generator.parameters()))


def test_slice_segments_waveform_crop():
    from vits_finetune.model import _slice_segments

    wav = torch.arange(20, dtype=torch.float32).view(1, 1, 20)
    starts = torch.tensor([5])
    out = _slice_segments(wav, starts, 4)
    assert out.shape == (1, 1, 4)
    assert torch.equal(out[0, 0], torch.tensor([5.0, 6.0, 7.0, 8.0]))
