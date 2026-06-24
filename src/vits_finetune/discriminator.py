import torch
import torch.nn.functional as fu
from torch.nn import Module, Conv2d, ModuleList, Conv1d, AvgPool1d
from vits_finetune.config import DiscriminatorConfig


dtype = torch.float
device = torch.device('cpu')


def reshape_1D_to_2D(x: torch.Tensor, period: int) -> torch.Tensor:
    batch_size, c_1, time = x.shape
    if time % period != 0:
        m = period - (time % period)
        time += m
        x = fu.pad(input=x, pad=(0, m), value=0.0)
    x = x.view(batch_size, c_1, time // period, period)
    return x


class PeriodDiscriminator(Module):
    def __init__(self, config: DiscriminatorConfig, period: int):

        super().__init__()
        self.config = config
        self.period = period

        channels = config.mpd_channels
        kernel = config.mpd_kernel
        stride = config.mpd_stride
        pad = (kernel - 1) // 2

        self.convs = ModuleList(
            [
                Conv2d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=(kernel, 1),
                    stride=(stride, 1),
                    padding=(pad, 0),
                )
                for i in range(len(channels) - 2)
            ]
        )
        self.convs.append(
            Conv2d(
                in_channels=channels[-2],
                out_channels=channels[-1],
                kernel_size=(kernel, 1),
                stride=(1, 1),
                padding=(pad, 0),
            )
        )
        self.conv_post = Conv2d(channels[-1], 1, (3, 1), 1, (1, 0))

    def forward(self, x: torch.Tensor):
        x = reshape_1D_to_2D(x, self.period)
        feature_maps = []
        for conv in self.convs:
            x = conv(x)
            x = fu.leaky_relu(x)
            feature_maps.append(x)

        x = self.conv_post(x)  # 1024 -> 1
        feature_maps.append(x)
        out = torch.flatten(x, 1)
        return out, feature_maps


class MPD(Module):
    def __init__(self, config: DiscriminatorConfig):
        super().__init__()
        self.config = config
        self.periods = config.mpd_periods
        self.discriminators = ModuleList(
            [PeriodDiscriminator(config, p) for p in config.mpd_periods]
        )

    def forward(self, x):
        outs, fmaps = [], []
        for d in self.discriminators:
            out, feature_maps = d(x)
            outs.append(out)
            fmaps.append(feature_maps)
        return outs, fmaps


class ScaleDiscriminator(Module):
    def __init__(self, config: DiscriminatorConfig):
        super().__init__()
        self.conv_generator = lambda i: [
            config.msd_channels[i],
            config.msd_channels[i + 1],
            config.msd_kernels[i],
            config.msd_strides[i],
            config.msd_kernels[i] // 2,
        ]

        self.convs = ModuleList(
            [
                Conv1d(*self.conv_generator(i))
                for i in range(len(config.msd_channels) - 1)
            ]
        )

        self.conv_post = Conv1d(config.msd_channels[-1], 1, 3, 1, 1)

    def forward(self, x: torch.Tensor):
        feature_maps = []
        for conv in self.convs:
            x = conv(x)
            x = fu.leaky_relu(x, 0.1)
            feature_maps.append(x)
        x = self.conv_post(x)
        feature_maps.append(x)
        out = torch.flatten(x, 1)
        return out, feature_maps


class MSD(Module):
    def __init__(self, config: DiscriminatorConfig):
        super().__init__()
        self.scales = config.msd_scales
        self.discriminators = ModuleList(
            [ScaleDiscriminator(config) for _ in range(self.scales)]
        )
        self.pool = AvgPool1d(4, 2, padding=2)

    def forward(self, x):
        outs, fmaps = [], []
        for i in range(self.scales):
            if i > 0:
                x = self.pool(x)
            out, feature_maps = self.discriminators[i](x)
            outs.append(out)
            fmaps.append(feature_maps)
        return outs, fmaps


class Discriminator(Module):
    def __init__(self, config):
        super().__init__()
        self.mpd = MPD(config)
        self.msd = MSD(config)

    def forward(self, x):
        mpd_outs, mpd_fmaps = self.mpd(x)
        msd_outs, msd_fmaps = self.msd(x)
        return mpd_outs + msd_outs, mpd_fmaps + msd_fmaps
