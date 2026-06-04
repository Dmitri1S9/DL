"""Build the evaluation test set: a manifest + reference audio.

Contract (see core.contracts): one TestItem per line in ``config.TEST_MANIFEST``,
each pointing at a reference wav under ``config.REFERENCE_DIR``. The model stage
reads the texts to synthesize; the evaluation stage reads texts + references.

MOCK for now: emits a few fixed sentences with silent reference wavs so the rest
of the pipeline runs offline. Replace ``_mock_items`` with the real LJSpeech test
split (the last ``config.TEST_SIZE`` clips from ``data.download``).
"""

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
    """Fixed sentences with 1 s silent reference wavs (placeholder)."""
    config.REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    silence = np.zeros(config.SAMPLE_RATE, dtype=np.float32)
    items = []
    for i, text in enumerate(_MOCK_SENTENCES):
        item_id = f'test_{i:04d}'
        ref_path = config.REFERENCE_DIR / f'{item_id}.wav'
        sf.write(str(ref_path), silence, config.SAMPLE_RATE)
        items.append(TestItem(id=item_id, text=text, ref_audio=str(ref_path)))
    return items


def prepare() -> None:
    items = _mock_items()
    write_manifest(items, config.TEST_MANIFEST)
    logger.warning('prepare.py is a MOCK — fixed sentences + silent references.')
    logger.success(f'Wrote {len(items)} items -> {config.TEST_MANIFEST}')


if __name__ == '__main__':
    prepare()
