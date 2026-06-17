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
    batch_size: int = 2
    # Fine-tuning a pretrained VITS: 2e-5 matches the proven ylacombe recipe.
    # 2e-4 diverges the stochastic duration predictor -> collapsed durations ->
    # fast, unintelligible speech at inference. The (from-scratch) discriminator
    # keeps a higher lr below, since it has to learn from nothing.
    learning_rate: float = 2e-5
    num_epochs: int = 2
    segment_size: int = 8192  # waveform crop length (samples) fed to the decoder
    num_workers: int = 4
    seed: int = 1234

    # --- loss weights ---
    mel_loss_weight: float = 45.0
    kl_loss_weight: float = 1.0
    fm_loss_weight: float = 2.0
    gen_loss_weight: float = 1.0
    disc_loss_weight: float = 1.0

    # --- adversarial optimizer ---
    disc_learning_rate: float = 2e-4

    # --- logging / checkpointing cadence (in steps) ---
    log_every: int = 50
    checkpoint_every: int = 1000

    # --- dataset (HuggingFace Hub, see data/push_b1_dataset.py) ---
    dataset_repo_id: str = 'Dmi1tr13/ljspeech-b1'
    dataset_split: str = 'train'
    # Cap on number of training clips (None = use all). On Colab we can't raise
    # vm.max_map_count, and phonemizer/espeak leaks a memory mapping per call, so
    # a smaller pass avoids the 'failed to map segment' crash. The eval test set
    # (last 500 clips) is held out regardless of this value.
    max_train_clips: int | None = None

    # --- paths ---
    checkpoint_dir: Path = ROOT / 'models' / 'vits_finetune'

    def __post_init__(self) -> None:
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def spectrogram_bins(self) -> int:
        """Number of frequency bins in the linear spectrogram (``n_fft // 2 + 1``)."""
        return self.n_fft // 2 + 1
