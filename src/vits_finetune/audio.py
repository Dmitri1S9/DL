"""
Waveform -> spectrogram feature extraction for VITS fine-tuning.
"""

from __future__ import annotations

import torch
import torchaudio

from vits_finetune.config import TrainingConfig


def wav_to_linear_spectrogram(
    wav: torch.Tensor, config: TrainingConfig
) -> torch.Tensor:
    """Compute the linear-scale magnitude STFT spectrogram for one waveform.
    Returns:
        Float tensor of shape ``(config.spectrogram_bins, num_frames)``, i.e.
        ``(n_fft // 2 + 1, T)`` — this is the posterior encoder's input.
    """
    window = torch.hann_window(config.win_length, device=wav.device)
    res = torch.stft(
        wav,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=window,
        center=False,
        return_complex=True,
    )
    return res.abs()


def wav_to_mel_spectrogram(wav: torch.Tensor, config: TrainingConfig) -> torch.Tensor:
    """Compute the log-mel spectrogram for one waveform.
    Returns:
        Float tensor of shape ``(config.n_mels, num_frames)``.
    """
    linear_spec = wav_to_linear_spectrogram(wav, config)
    f_max = config.mel_fmax if config.mel_fmax is not None else config.sampling_rate / 2
    mel_fb = torchaudio.functional.melscale_fbanks(
        n_freqs=config.n_fft // 2 + 1,
        f_min=config.mel_fmin,
        f_max=f_max,
        n_mels=config.n_mels,
        sample_rate=config.sampling_rate,
        norm='slaney',
        mel_scale='slaney',
    )  # -> (n_freqs, n_mels)
    mel_fb = mel_fb.to(linear_spec.device)
    mel_spec = torch.matmul(mel_fb.T, linear_spec)  # (n_mels, T)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
    return mel_spec
