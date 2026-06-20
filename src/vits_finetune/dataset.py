"""Dataset for VITS fine-tuning.    
"""

from __future__ import annotations

import contextlib
import hashlib
import logging

import phonemizer
from phonemizer.backend import EspeakBackend
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
        self.input_ids = self._tokenize_all(split)

    def _tokenize_all(self, split: str) -> list[torch.Tensor]:
        """Tokenize all texts once, caching the result to disk.
        """
        key = hashlib.md5(
            f'{self.training_config.dataset_repo_id}|{split}|'
            f'{self.tokenizer.name_or_path}'.encode()
        ).hexdigest()[:16]
        cache_dir = self.training_config.checkpoint_dir / 'tok_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f'{key}.pt'
        if cache_path.exists():
            return torch.load(cache_path)
        with self._reused_espeak_backend():
            ids = [
                self.tokenizer(text, return_tensors='pt').input_ids.squeeze(0)
                for text in self.dataset['text']
            ]
        torch.save(ids, cache_path)
        return ids

    @contextlib.contextmanager
    def _reused_espeak_backend(self):
        """Patch ``phonemizer.phonemize`` to reuse one EspeakBackend instance.

        The tokenizer calls ``phonemizer.phonemize(...)`` once per text, and
        that function builds a brand new EspeakBackend (which copies
        libespeak-ng.dll into a fresh temp dir) on every call. On Windows the
        DLL often can't be unloaded, so the temp dir leaks; multiplied by
        thousands of texts that fills the disk. Building the backend once and
        reusing it for the whole pass avoids that.
        """
        # EspeakBackend() with no explicit logger calls phonemizer's own
        # get_logger(), which resets this logger's level to WARNING on every
        # construction -- passing it in explicitly keeps our ERROR level.
        quiet_logger = logging.getLogger('phonemizer')
        quiet_logger.setLevel(logging.ERROR)
        backend = EspeakBackend(
            'en-us', preserve_punctuation=True, with_stress=True, logger=quiet_logger
        )
        original_phonemize = phonemizer.phonemize

        def _patched(text, **kwargs):
            return backend.phonemize([text], strip=kwargs.get('strip', False))[0]

        phonemizer.phonemize = _patched
        try:
            yield
        finally:
            phonemizer.phonemize = original_phonemize

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
        input_ids = self.input_ids[index]
        audio_array = torch.from_numpy(example["audio"]["array"]).float() 
        linear_spectrogram = wav_to_linear_spectrogram(audio_array, self.training_config)
        mel_spectrogram = wav_to_mel_spectrogram(audio_array, self.training_config)

        return {
            "input_ids": input_ids,
            "waveform": audio_array,
            "linear_spec": linear_spectrogram,        
            "mel_spec": mel_spectrogram,
        }  
