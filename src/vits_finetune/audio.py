"""
Waveform -> spectrogram feature extraction for VITS fine-tuning.
"""

from __future__ import annotations

import soundfile as sf
import torch
import torchaudio

from vits_finetune.config import TrainingConfig


def wav_to_linear_spectrogram(wav: torch.Tensor, config: TrainingConfig) -> torch.Tensor:
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
        norm="slaney",
        mel_scale="slaney",
    )  # -> (n_freqs, n_mels)
    mel_fb = mel_fb.to(linear_spec.device)
    mel_spec = torch.matmul(mel_fb.T, linear_spec)  # (n_mels, T)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
    return mel_spec


if __name__ == "__main__":
    config = TrainingConfig()
    wav_np, sr = sf.read("audio/ljspeech_b1/LJ001-0001_b1.wav", dtype="float32")
    wav = torch.from_numpy(wav_np)
    if wav.ndim > 1:
        wav = wav.mean(dim=-1)  # -> mono, 1D (T,)
    if sr != config.sampling_rate:
        wav = torchaudio.functional.resample(wav, sr, config.sampling_rate)

    lin = wav_to_linear_spectrogram(wav, config)
    mel = wav_to_mel_spectrogram(wav, config)
    print(lin.shape)
    print(mel.shape)
    print(torch.isnan(mel).any(), torch.isinf(mel).any())
