"""Build the evaluation test set: a manifest + reference audio.

Contract (see core.contracts): one TestItem per line in ``config.TEST_MANIFEST``,
each pointing at a reference wav under ``config.REFERENCE_DIR``. The model stage
reads the texts to synthesize; the evaluation stage reads texts + references.

Two paths:
  * real (default) — the held-out LJSpeech test split (last ``config.TEST_SIZE``
    clips): real recordings as references + their cleaned transcripts. This is what
    makes WER/CER and MCD meaningful.
  * mock (``--mock``) — fixed sentences with silent references, so the offline
    pipeline / smoke test runs without downloading LJSpeech.

The heavy imports (datasets) live inside ``_real_items`` so the mock path stays
importable on a bare checkout.
"""

import argparse

import numpy as np
import soundfile as sf

from core import config
from core.contracts import write_manifest
from core.dto import TestItem
from core.logger import logger

_MOCK_SENTENCES = [
    'The quick brown fox jumps over the lazy dog.',
    'She sells sea shells by the sea shore.',
    'Printing, in the only sense with which we are at present concerned.',
    'How much wood would a woodchuck chuck if a woodchuck could chuck wood.',
    'The birch canoe slid on the smooth planks.',
]


def _mock_items() -> list[TestItem]:
    """Fixed sentences with 1 s silent reference wavs (offline placeholder)."""
    config.REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    silence = np.zeros(config.SAMPLE_RATE, dtype=np.float32)
    items = []
    for i, text in enumerate(_MOCK_SENTENCES):
        item_id = f'test_{i:04d}'
        ref_path = config.REFERENCE_DIR / f'{item_id}.wav'
        sf.write(str(ref_path), silence, config.SAMPLE_RATE)
        items.append(TestItem(id=item_id, text=text, ref_audio=str(ref_path)))
    return items


def _real_items(limit: int | None = None) -> list[TestItem]:
    """LJSpeech held-out test split: real reference wavs + cleaned transcripts."""
    from datasets import Audio

    from data.download import download_ljspeech
    from data.prepare_training import cleanup_text

    _, test_set = download_ljspeech()
    test_set = test_set.cast_column('audio', Audio(sampling_rate=config.SAMPLE_RATE))
    if limit is not None:
        test_set = test_set.select(range(limit))

    config.REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for ex in test_set:
        item_id = ex['id']
        text = cleanup_text(ex)['text']
        ref_path = config.REFERENCE_DIR / f'{item_id}.wav'
        audio = np.asarray(ex['audio']['array'], dtype=np.float32)
        sf.write(str(ref_path), audio, config.SAMPLE_RATE)
        items.append(TestItem(id=item_id, text=text, ref_audio=str(ref_path)))
    return items


def _clear_reference_dir() -> None:
    """Drop stale reference wavs so the dir stays in sync with the new manifest."""
    if config.REFERENCE_DIR.exists():
        for wav in config.REFERENCE_DIR.glob('*.wav'):
            wav.unlink()


def prepare(mock: bool = False, limit: int | None = None) -> None:
    _clear_reference_dir()
    items = _mock_items() if mock else _real_items(limit)
    write_manifest(items, config.TEST_MANIFEST)
    if mock:
        logger.warning('prepare.py MOCK — fixed sentences + silent references.')
    logger.success(f'Wrote {len(items)} items -> {config.TEST_MANIFEST}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build the evaluation test set (manifest + reference audio).'
    )
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Offline: fixed sentences + silent references (no LJSpeech download)',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Only the first N test clips (faster partial runs)',
    )
    args = parser.parse_args()
    prepare(mock=args.mock, limit=args.limit)


if __name__ == '__main__':
    main()
