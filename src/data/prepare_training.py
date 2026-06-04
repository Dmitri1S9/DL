"""Build the LJSpeech train/val splits for fine-tuning SpeechT5.

This is the data side of the training contract. It delivers ``(train, val)``
HuggingFace Datasets with two columns:

  * ``text``  — cleaned for the SpeechT5 tokenizer (lowercase, ASCII)
  * ``audio`` — resampled to 16 kHz

The training code (``model.train``, Dima's) takes it from here: it attaches the
speaker embedding (``model.speaker``), runs the SpeechT5Processor to produce
``input_ids`` + mel ``labels``, filters by token length, collates, and trains.
This module intentionally does NONE of that model-specific work.
"""

from datasets import Audio

from core import config
from core.logger import logger
from data.download import download_ljspeech

# Characters the SpeechT5 tokenizer can't encode -> ASCII fallbacks. LJSpeech's
# 'normalized_text' already spells out numbers, so we only map the occasional
# accented character. Extend this if a vocab check surfaces more.
_CHAR_REPLACEMENTS = [
    ('à', 'a'),
    ('â', 'a'),
    ('ç', 'c'),
    ('è', 'e'),
    ('é', 'e'),
    ('ê', 'e'),
    ('í', 'i'),
    ('ï', 'i'),
    ('ñ', 'n'),
    ('ô', 'o'),
    ('ö', 'o'),
    ('û', 'u'),
    ('ü', 'u'),
]


def cleanup_text(example: dict) -> dict:
    """Lowercase + replace characters the SpeechT5 tokenizer can't handle."""
    text = example['normalized_text'].lower()
    for src, dst in _CHAR_REPLACEMENTS:
        text = text.replace(src, dst)
    example['text'] = text
    return example


def load_training_splits():
    """Return ``(train, val)`` datasets with ``text`` (cleaned) + ``audio`` (16 kHz).

    Note: the first call materializes the cleanup pass over LJSpeech (and downloads
    it, ~2.6 GB), then it is cached by the datasets library.
    """
    train_set, _ = download_ljspeech()  # fine-tune on the train split only

    train_set = train_set.cast_column('audio', Audio(sampling_rate=config.SAMPLE_RATE))
    train_set = train_set.map(cleanup_text)
    keep = {'text', 'audio'}
    train_set = train_set.remove_columns(
        [c for c in train_set.column_names if c not in keep]
    )

    split = train_set.train_test_split(test_size=config.VAL_SIZE, seed=42)
    train, val = split['train'], split['test']
    logger.success(f'Training splits ready: train={len(train)} val={len(val)}')
    return train, val


def main() -> None:
    train, val = load_training_splits()
    logger.info(f'train={len(train)} val={len(val)}')
    logger.info(f'sample text: {train[0]["text"]!r}')
    logger.info(f'sample audio: {train[0]["audio"]["sampling_rate"]} Hz')


if __name__ == '__main__':
    main()
