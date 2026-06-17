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
        # Keep the pretrained durations unless explicitly fine-tuning timing.
        if not self.training_config.train_duration_predictor:
            for param in self.vits.duration_predictor.parameters():
                param.requires_grad = False

    def text_mask(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return _sequence_mask(batch['input_lengths'], batch['input_ids'].shape[1])

    def text_encoder(self, text_mask: torch.Tensor, batch: dict[str, torch.Tensor]):
        return self.vits.text_encoder(
            batch['input_ids'],
            text_mask.unsqueeze(-1).float(),
            attention_mask=text_mask.float(),
        )

    def spec_mask(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return _sequence_mask(
            batch['spec_lengths'], batch['linear_spec'].shape[2]
        ).unsqueeze(1).float()

    def audio_encoder(self, batch: dict[str, torch.Tensor], spec_mask: torch.Tensor):
        return self.vits.posterior_encoder(batch['linear_spec'], spec_mask)

    @property
    def flow(self):
        return self.vits.flow

    @property
    def duration_predictor(self):
        return self.vits.duration_predictor

    @property
    def decoder(self):
        return self.vits.decoder

    def forward_train(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the VITS *training* forward pass on a padded batch from ``collate.collate_fn``.

        Steps to implement:
          1. Text encoding: input_ids -> text hidden states + prior_mean / prior_log_stddev.
          2. Posterior encoding: linear_spec -> z, posterior_mean, posterior_log_stddev.
          3. Flow: z -> z_p (prior latent space).
          4. MAS: alignment between z_p and prior (maximum_path / neg_cross_entropy).
          5. Expand per-token prior to per-frame via alignment; duration loss.
          6. Random segment crop of z (and matching target mel).
          7. Decode z segment -> waveform -> predicted mel.
          8. Return tensors for losses.py / train.py.

        Returns:
            Dict with (at least):
              - "predicted_mel", "target_mel"            -> recon_loss
              - "posterior_mean", "posterior_log_stddev",
                "prior_mean", "prior_log_stddev", "z_mask" -> kl_loss
              - "duration_loss"                          -> duration term
        """
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

        # (B, 1, segment_frames*hop)
        predicted_waveform = self.decoder(z_segment)
        predicted_mel = wav_to_mel_spectrogram(
            predicted_waveform.squeeze(1), self.training_config
        )

        num_frames = min(predicted_mel.shape[-1], target_mel.shape[-1])

        wav_segment_size = segment_frames * self.training_config.hop_length
        wav_starts = starts * self.training_config.hop_length
        target_waveform = _slice_segments(
            batch['waveform'], wav_starts, wav_segment_size
        )
        num_samples = min(predicted_waveform.shape[-1], target_waveform.shape[-1])

        return {
            'predicted_mel': predicted_mel[..., :num_frames],
            'target_mel': target_mel[..., :num_frames],
            'predicted_waveform': predicted_waveform[..., :num_samples],
            'target_waveform': target_waveform[..., :num_samples],
            'posterior_mean': posterior_mean,
            'posterior_log_stddev': posterior_log_stddev,
            'prior_mean': prior_mean,
            'prior_log_stddev': prior_log_stddev,
            'z_mask': spec_mask,
            'duration_loss': duration_loss,
            'z_p': z_p,
        }
