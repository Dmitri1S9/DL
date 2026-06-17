"""HiFi-GAN discriminator for VITS adversarial fine-tuning (written from scratch).

VITS is a GAN: the decoder (generator) only learns to produce realistic
waveforms when trained against a discriminator. ``transformers.VitsModel`` ships
the generator but not the discriminator, so we add the standard HiFi-GAN one:
a multi-period discriminator (MPD) + a multi-scale discriminator (MSD).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm

LRELU_SLOPE = 0.1


class DiscriminatorP(nn.Module):
    """One period sub-discriminator: reshapes the 1D wave to 2D by ``period``."""

    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3) -> None:
        super().__init__()
        self.period = period
        pad = (kernel_size - 1) // 2
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), (pad, 0))),
            weight_norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), (pad, 0))),
            weight_norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), (pad, 0))),
            weight_norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), (pad, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, (pad, 0))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, (1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap: list[torch.Tensor] = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), 'reflect')
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for layer in self.convs:
            x = F.leaky_relu(layer(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class DiscriminatorS(nn.Module):
    """One scale sub-discriminator operating directly on the 1D wave."""

    def __init__(self, use_spectral_norm: bool = False) -> None:
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap: list[torch.Tensor] = []
        for layer in self.convs:
            x = F.leaky_relu(layer(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: tuple[int, ...] = (2, 3, 5, 7, 11)) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(p) for p in periods])

    def forward(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        outs, fmaps = [], []
        for d in self.discriminators:
            out, fmap = d(x)
            outs.append(out)
            fmaps.append(fmap)
        return outs, fmaps


class MultiScaleDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        outs, fmaps = [], []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                x = self.meanpools[i - 1](x)
            out, fmap = d(x)
            outs.append(out)
            fmaps.append(fmap)
        return outs, fmaps


class VitsDiscriminator(nn.Module):
    """MPD + MSD combined; compares a real and a fake waveform segment."""

    def __init__(self, periods: tuple[int, ...] = (2, 3, 5, 7, 11)) -> None:
        super().__init__()
        self.mpd = MultiPeriodDiscriminator(periods)
        self.msd = MultiScaleDiscriminator()

    def forward(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> tuple[list, list, list, list]:
        r_mpd, fr_mpd = self.mpd(real)
        f_mpd, ff_mpd = self.mpd(fake)
        r_msd, fr_msd = self.msd(real)
        f_msd, ff_msd = self.msd(fake)
        return (
            r_mpd + r_msd,
            f_mpd + f_msd,
            fr_mpd + fr_msd,
            ff_mpd + ff_msd,
        )
