"""General training configuration: hyperparameters and filesystem paths.

This holds everything needed to run the fine-tuning loop that is *not* specific
to the VITS model/checkpoint itself — that lives in ``model_config.py``. Keeping
the two separate means swapping the base checkpoint never touches batch size,
paths, or audio front-end settings, and vice versa.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# vits_finetune -> src -> project root
ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainingConfig:
    """Hyperparameters and paths for fine-tuning VITS on a custom voice."""

    # --- audio / feature extraction ---
    sampling_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float | None = None

    # --- training loop ---
    batch_size: int = 12
    learning_rate: float = 2e-4
    num_epochs: int = 5
    segment_size: int = 8192 * 2 # waveform crop length (samples) fed to the decoder
    num_workers: int = 4
    seed: int = 1234

    # --- loss weights ---
    mel_loss_weight: float = 45.0
    kl_loss_weight: float = 1.0

    # --- logging / checkpointing cadence (in steps) ---
    log_every: int = 50
    checkpoint_every: int = 1000

    # --- dataset (HuggingFace Hub, see data/push_b1_dataset.py) ---
    dataset_repo_id: str = 'Dmi1tr13/ljspeech-b1'
    train_split: str = 'train[:95%]'
    test_split: str = 'train[95%:]'

    # --- paths ---
    checkpoint_dir: Path = ROOT / 'models' / 'vits_finetune'

    def __post_init__(self) -> None:
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def spectrogram_bins(self) -> int:
        """Number of frequency bins in the linear spectrogram (``n_fft // 2 + 1``)."""
        return self.n_fft // 2 + 1




@dataclass
class DiscriminatorConfig:
    # --- fun mpd ---
    mpd_periods: tuple = (2, 7, 13, 29, 37, 73, 97, 113, 137)
    mpd_channels: tuple = (1, 32, 128, 512, 1024, 1024)
    mpd_kernel: int = 5
    mpd_stride: int = 3
    batch_size: int = TrainingConfig.batch_size
    segment_size: int = TrainingConfig.segment_size


    msd_channels: tuple = (1, 16, 64, 256, 1024, 1024)
    msd_kernels: tuple = (15, 41, 41, 41, 5)
    msd_strides: tuple = (1, 4, 4, 4, 1)
    msd_scales: int = 3

