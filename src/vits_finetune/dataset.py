"""Dataset for VITS fine-tuning.    
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from vits_finetune.audio import wav_to_mel_spectrogram, wav_to_linear_spectrogram
from vits_finetune.config import TrainingConfig


class VitsFinetuneDataset(Dataset):
    """Maps each HF Hub dataset row to tokenized text + audio features for one clip."""

    def __init__(
        self,
        training_config: TrainingConfig,
        tokenizer: PreTrainedTokenizerBase,
        split: str | None = None,
    ) -> None:
        """
        Args:
            training_config: supplies ``dataset_repo_id`` / ``train_split``
                and the STFT/mel parameters for ``audio.py``.
            tokenizer: VITS tokenizer (``AutoTokenizer.from_pretrained(...)``)
                used to turn ``text`` into ``input_ids``.
            split: HF split string; defaults to ``training_config.train_split``.
        """
        self.tokenizer = tokenizer
        self.training_config = training_config
        split = split or training_config.train_split
        self.dataset = load_dataset(training_config.dataset_repo_id, split=split)

    def __len__(self) -> int:
        """Return the number of examples in ``self.hf_dataset``."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Build one training example.

        Returns:
            Dict with (at least):
              - ``"input_ids"``: LongTensor ``(T_text,)``
              - ``"waveform"``: FloatTensor ``(T_wav,)``
              - ``"linear_spec"``: FloatTensor ``(spectrogram_bins, T_spec)``
              - ``"mel_spec"``: FloatTensor ``(n_mels, T_spec)``
            Unpadded — ``collate.collate_fn`` pads these into a batch.
        """
        if index >= len(self.dataset):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.dataset)}")
        
        example = self.dataset[index]
        input_ids = self.tokenizer(example["text"], return_tensors="pt").input_ids.squeeze(0)
        audio_array = torch.from_numpy(example["audio"]["array"]).float() 
        linear_spectrogram = wav_to_linear_spectrogram(audio_array, self.training_config)
        mel_spectrogram = wav_to_mel_spectrogram(audio_array, self.training_config)

        return {
            "input_ids": input_ids,
            "waveform": audio_array,
            "linear_spec": linear_spectrogram,        
            "mel_spec": mel_spectrogram,
        }  
