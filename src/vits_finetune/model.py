"""
Trainable wrapper around a pretrained VITS model.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import VitsModel

from vits_finetune.alignment import maximum_path, neg_cross_entropy
from vits_finetune.audio import wav_to_mel_spectrogram
from vits_finetune.config import TrainingConfig
from vits_finetune.model_config import VitsModelConfig


def _sequence_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    """``(B,)`` lengths -> ``(B, max_length)`` bool mask, True where index < length."""
    return torch.arange(max_length, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def _slice_segments(x: torch.Tensor, starts: torch.Tensor, segment_size: int) -> torch.Tensor:
    """Crop a length-``segment_size`` window from each example of ``x`` (B, C, T)."""
    out = x.new_zeros((x.shape[0], x.shape[1], segment_size))
    for i in range(x.shape[0]):
        start = int(starts[i].item())
        out[i] = x[i, :, start : start + segment_size]
    return out


def _rand_slice_segments(
    x: torch.Tensor, lengths: torch.Tensor, segment_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly crop a length-``segment_size`` window from each example of ``x`` (B, C, T)."""
    max_starts = (lengths - segment_size).clamp(min=0).to(torch.float32)
    starts = (torch.rand(x.shape[0], device=x.device) * (max_starts + 1)).long()
    return _slice_segments(x, starts, segment_size), starts


class VitsFinetuneModel(nn.Module):
    """Wraps a pretrained ``VitsModel`` and adds a training-mode forward pass."""

    def __init__(
        self,
        model_config: VitsModelConfig,
        training_config: TrainingConfig | None = None,
    ) -> None:
        """Load the pretrained VITS model named in ``model_config``."""
        super().__init__()
        self.model_config = model_config
        self.training_config = training_config or TrainingConfig()
        self.vits: VitsModel = VitsModel.from_pretrained(
            model_config.pretrained_model_name,
            cache_dir=str(model_config.cache_dir),
        )

    text_mask = lambda self, batch: _sequence_mask(batch['input_lengths'],
                                            batch['input_ids'].shape[1])

    text_encoder = lambda self, text_mask, batch: self.vits.text_encoder(
            batch['input_ids'],
            text_mask.unsqueeze(-1).float(),
            attention_mask=text_mask.float(),
    )

    spec_mask = lambda self, batch: _sequence_mask(
            batch['spec_lengths'], batch['linear_spec'].shape[2]).unsqueeze(1).float()

    audio_encoder = lambda self, batch, spec_mask: self.vits.posterior_encoder( 
        batch['linear_spec'],
        spec_mask,
    )
    flow = property(lambda self: self.vits.flow)

    duration_predictor = property(lambda self: self.vits.duration_predictor)

    decoder = property(lambda self: self.vits.decoder)

    def forward_train(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        text_mask = self.text_mask(batch)  # (B, T_text)
        text_out = self.text_encoder(text_mask, batch)
        prior_mean = text_out.prior_means.transpose(1, 2)
        prior_log_stddev = text_out.prior_log_variances.transpose(1, 2)
        last_hidden_state = text_out.last_hidden_state 

        spec_mask = self.spec_mask(batch)
        z, posterior_mean, posterior_log_stddev = self.audio_encoder(batch, spec_mask)

        z_p = self.flow(z, spec_mask, reverse=False)

        # (B, T_text, T_frames)
        neg_cent = neg_cross_entropy(z_p, prior_mean, prior_log_stddev)  

        # (B, T_text, T_frames)
        mas_mask = (
            text_mask.unsqueeze(2) & spec_mask.bool()
        ).float() 
        # MAS algorithm result
        attn = maximum_path(neg_cent, mas_mask)

        prior_mean = torch.matmul(prior_mean, attn)
        prior_log_stddev = torch.matmul(prior_log_stddev, attn)
        durations = attn.sum(dim=2).unsqueeze(1)  # (B, T_text)

        duration_loss =self.duration_predictor(
            last_hidden_state.transpose(1,2),
            text_mask.unsqueeze(1).float(),
            durations=durations,
            reverse=False
        ).mean()

        segment_frames = self.training_config.segment_size // self.training_config.hop_length
        segment_frames = min(segment_frames, int(batch['spec_lengths'].min().item()))

        z_segment, starts = _rand_slice_segments(z, batch['spec_lengths'], segment_frames)
        target_mel = _slice_segments(batch['mel_spec'], starts, segment_frames)

        hop = self.training_config.hop_length
        seg_wave = segment_frames * hop
        wave = batch['waveform']
        target_waveform = wave.new_zeros((wave.shape[0], 1, seg_wave))
        for i in range(wave.shape[0]):
            s = int(starts[i].item()) * hop
            target_waveform[i, 0, :] = wave[i, 0, s : s + seg_wave]

        # (B, 1, segment_frames*hop)
        predicted_waveform = self.decoder(z_segment)
        predicted_mel = wav_to_mel_spectrogram(
            predicted_waveform.squeeze(1), self.training_config
        )

        num_frames = min(predicted_mel.shape[-1], target_mel.shape[-1])

        return {
            'target_waveform': target_waveform,
            'predicted_waveform': predicted_waveform,
            'predicted_mel': predicted_mel[..., :num_frames],
            'target_mel': target_mel[..., :num_frames],
            'posterior_mean': posterior_mean,
            'posterior_log_stddev': posterior_log_stddev,
            'prior_mean': prior_mean,
            'prior_log_stddev': prior_log_stddev,
            'z_mask': spec_mask,
            'duration_loss': duration_loss,
            'z_p': z_p,
        }