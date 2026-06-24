"""
python -m augmentation.magic
"""

from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

from core import config
from core.contracts import write_manifest
from core.dto import TestItem
from core.logger import logger
from effects.effects import apply_b1_droid, process_text_phonemic

# Clips longer than this are skipped; shorter ones are padded with silence.
_MAX_CLIP_SECONDS = 11.0
_MAX_SAMPLES = int(_MAX_CLIP_SECONDS * config.SAMPLE_RATE)


def pad_or_skip(audio: np.ndarray) -> np.ndarray | None:
    """Pad short audio with silence. Return None if too long — caller skips."""
    if len(audio) > _MAX_SAMPLES:
        return None
    if len(audio) < _MAX_SAMPLES:
        return np.concatenate(
            [audio, np.zeros(_MAX_SAMPLES - len(audio), dtype=np.float32)]
        )
    return audio


class Magic:
    def __init__(self, n: int = 10):
        self.n = n
        self.ds = self._load(n)

    def _load(self, n: int):
        ds = load_dataset(
            'keithito/lj_speech',
            split='train',
            streaming=True,
            cache_dir=str(config.DATA_DIR),
            trust_remote_code=True,
        )
        ds = ds.cast_column('audio', Audio(sampling_rate=config.SAMPLE_RATE))
        return ds.take(n)

    def _augment_sample(
        self, text: str, audio: np.ndarray
    ) -> tuple[str, np.ndarray] | None:
        """Apply padding check + accent + droid. Returns None if sample is too long."""
        audio = pad_or_skip(audio)
        if audio is None:
            return None
        return process_text_phonemic(text), apply_b1_droid(audio)

    def _save_sample(
        self, audio: np.ndarray, text: str, sample_id: str, out_dir: Path
    ) -> TestItem:
        """Write wav to disk and return a TestItem for the manifest."""
        out_path = out_dir / f'{sample_id}.wav'
        sf.write(str(out_path), audio, config.SAMPLE_RATE)
        return TestItem(id=sample_id, text=text, ref_audio=str(out_path))

    def augment_and_save(self, out_dir: Path = config.AUDIO_DIR / 'augmented') -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f'Augmenting {self.n} samples -> {out_dir}')

        items: list[TestItem] = []
        skipped = 0

        for i, sample in enumerate(self.ds):
            original_text: str = sample['normalized_text']
            audio: np.ndarray = sample['audio']['array'].astype(np.float32)

            result = self._augment_sample(original_text, audio)
            if result is None:
                logger.warning(f'[{i}] skipped — longer than {_MAX_CLIP_SECONDS:.0f}s')
                skipped += 1
                continue

            accented_text, b1_audio = result
            item = self._save_sample(
                b1_audio, accented_text, f'sample_{i:02d}', out_dir
            )
            items.append(item)

            logger.info(f"[{i}] '{original_text[:50]}' -> '{accented_text[:50]}'")
            logger.success(f'     saved {item.id}.wav')

        manifest_path = out_dir / 'manifest.jsonl'
        write_manifest(items, manifest_path)
        logger.success(
            f'Done: {len(items)} saved, {skipped} skipped. Manifest -> {manifest_path}'
        )
        return manifest_path


def demo_b1(out_path: Path = config.AUDIO_DIR / 'b1_demo.wav') -> None:
    """Synthesize a phrase with the TTS model and apply the B1 droid effect."""
    from model.synthesize import TTSModel

    out_path.parent.mkdir(parents=True, exist_ok=True)

    text = 'Roger roger. All units, proceed with the mission. Standing by.'
    accented = process_text_phonemic(text)
    logger.info(f"Text:     '{text}'")
    logger.info(f"Accented: '{accented}'")

    logger.info('Loading TTS model...')
    model = TTSModel()
    audio = model.synthesize(accented)
    logger.success(f'Synthesized {len(audio) / config.SAMPLE_RATE:.1f}s of audio')

    b1 = apply_b1_droid(audio)
    sf.write(str(out_path), b1, config.SAMPLE_RATE)
    logger.success(f'B1 demo saved -> {out_path}')


def demo_pipeline(out_dir: Path = config.AUDIO_DIR / 'pipeline_demo') -> None:
    """Show the whole pipeline on one real clean-speech sample.

    Downloads 1 sample from LibriSpeech test-clean (individual flac, not tar.bz2)
    -> runs it through Magic -> saves wav + manifest.jsonl.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info('[1/4] Downloading 1 clean sample from LibriSpeech test-clean...')
    ds = load_dataset(
        'openslr/librispeech_asr',
        'clean',
        split='test',
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.cast_column('audio', Audio(sampling_rate=config.SAMPLE_RATE))
    sample = next(iter(ds))

    original_text: str = sample['text'].lower()
    audio: np.ndarray = sample['audio']['array'].astype(np.float32)
    sf.write(str(out_dir / 'step1_original.wav'), audio, config.SAMPLE_RATE)
    logger.success(
        f'      step1_original.wav  ({len(audio) / config.SAMPLE_RATE:.1f}s)'
    )
    logger.info(f"      text: '{original_text[:80]}'")

    logger.info('[2/4] Applying augmentation (accent + B1 droid)...')
    magic = Magic.__new__(Magic)
    result = magic._augment_sample(original_text, audio)

    if result is None:
        logger.warning(f'Sample longer than {_MAX_CLIP_SECONDS:.0f}s — skipped')
        return

    accented_text, b1_audio = result
    sf.write(str(out_dir / 'step2_b1_droid.wav'), b1_audio, config.SAMPLE_RATE)
    logger.success(
        f'      step2_b1_droid.wav  ({len(b1_audio) / config.SAMPLE_RATE:.1f}s)'
    )
    logger.info(f"      accented: '{accented_text[:80]}'")

    logger.info('[3/4] Saving the manifest (training text+audio pair)...')
    item = magic._save_sample(b1_audio, accented_text, 'demo_sample', out_dir)
    write_manifest([item], out_dir / 'manifest.jsonl')

    logger.success('=== Result ===')
    logger.success('  step1_original.wav  — the original clean speech')
    logger.success('  step2_b1_droid.wav  — what becomes the training target')
    logger.success(f"  manifest.jsonl      — text='{item.text[:60]}'")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        demo_b1()
    elif len(sys.argv) > 1 and sys.argv[1] == 'pipeline':
        demo_pipeline()
    else:
        Magic().augment_and_save()
