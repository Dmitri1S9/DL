"""VITS model / pretrained-checkpoint configuration.

Separate from ``config.py``: this file describes *which* checkpoint we start
from and the architecture parameters relevant to loading and fine-tuning it.
General training hyperparameters (batch size, lr, epochs, paths, ...) live in
``config.py``.

Defaults match ``kakao-enterprise/vits-ljs``'s ``config.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# vits_finetune -> src -> project root
ROOT = Path(__file__).resolve().parents[2]


@dataclass
class VitsModelConfig:
    """Pretrained checkpoint + architecture parameters for VITS."""

    # --- pretrained checkpoint ---
    pretrained_model_name: str = 'kakao-enterprise/vits-ljs'
    cache_dir: Path = ROOT / 'models'  # HuggingFace cache (already holds vits-ljs)

    # --- architecture (must match the pretrained checkpoint's config.json for
    # `from_pretrained` to load weights cleanly) ---
    sampling_rate: int = 22050
    spectrogram_bins: int = 513  # n_fft // 2 + 1, posterior-encoder input size
    hidden_size: int = 192
    flow_size: int = 192
    num_speakers: int = 1
    speaker_embedding_size: int = 0
    use_stochastic_duration_prediction: bool = True
    noise_scale: float = 0.667
    noise_scale_duration: float = 0.8

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)

    def config_overrides(self) -> dict[str, object]:
        """Architecture fields as a dict of ``transformers.VitsConfig`` kwargs.

        Pass as ``**model_config.config_overrides()`` when constructing or
        overriding a ``VitsConfig``, e.g. for a fine-tune that changes
        ``num_speakers`` / ``speaker_embedding_size`` relative to the
        pretrained checkpoint.
        """
        return {
            'sampling_rate': self.sampling_rate,
            'spectrogram_bins': self.spectrogram_bins,
            'hidden_size': self.hidden_size,
            'flow_size': self.flow_size,
            'num_speakers': self.num_speakers,
            'speaker_embedding_size': self.speaker_embedding_size,
            'use_stochastic_duration_prediction': self.use_stochastic_duration_prediction,
            'noise_scale': self.noise_scale,
            'noise_scale_duration': self.noise_scale_duration,
        }
