"""Speech synthesis: VITS (kakao-enterprise/vits-ljs) → waveform.

VITS is end-to-end: text → waveform in one model, no separate vocoder or speaker
embeddings. Trained on LJSpeech — matches our training data. Quality is noticeably
better than SpeechT5 + HiFi-GAN.

Requires espeak-ng installed system-wide (phonemizer backend for the tokenizer):
  Windows: winget install espeak-ng.espeak-ng
  Linux:   apt install espeak-ng

The mock path returns a sine wave for offline/CI use without model weights.

Contract: <out_dir>/<id>.wav, 22050 Hz mono float32 (VITS native sample rate).
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from core import config
from core.contracts import read_manifest
from core.espeak import setup_espeak
from core.logger import logger

# phonemizer (the VITS tokenizer backend) needs the espeak-ng library.
setup_espeak()

VITS_SAMPLE_RATE = 22050


class TTSModel:
    """Wraps VITS behind a single ``synthesize(text)`` call."""

    def __init__(self, checkpoint: str = config.PRETRAINED):
        import torch
        from transformers import VitsModel, AutoTokenizer

        self._torch = torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        source = (
            config.TTS_MODEL_ID if checkpoint == config.PRETRAINED else str(checkpoint)
        )
        logger.info(f'Loading VITS from {source!r} on {self.device}...')

        self.tokenizer = AutoTokenizer.from_pretrained(
            source, cache_dir=str(config.MODELS_DIR)
        )
        self.model = VitsModel.from_pretrained(
            source, cache_dir=str(config.MODELS_DIR)
        ).to(self.device)
        self.model.eval()
        logger.success('VITS model ready.')

    def synthesize(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        with self._torch.no_grad():
            output = self.model(**inputs).waveform
        return output.squeeze().cpu().numpy().astype(np.float32)


def mock_synthesize(text: str) -> np.ndarray:
    """Deterministic sine whose length scales with the text — no model needed."""
    seconds = max(0.4, 0.06 * len(text))
    n = int(seconds * VITS_SAMPLE_RATE)
    t = np.linspace(0.0, seconds, n, endpoint=False)
    return (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def generate_for_manifest(
    manifest_path: Path = config.TEST_MANIFEST,
    out_dir: Path = config.GENERATED_DIR,
    checkpoint: str = config.PRETRAINED,
    mock: bool = False,
) -> Path:
    """Synthesize every item in the manifest into ``out_dir/<id>.wav``."""
    items = read_manifest(manifest_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    synth = mock_synthesize if mock else TTSModel(checkpoint).synthesize
    sr = VITS_SAMPLE_RATE
    tag = 'mock' if mock else checkpoint
    logger.info(f'Synthesizing {len(items)} clips ({tag}) -> {out_dir}')
    for item in items:
        wav = synth(item.text)
        sf.write(str(out_dir / f'{item.id}.wav'), wav, sr)
    logger.success(f'Wrote {len(items)} wavs to {out_dir}')
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Synthesize audio for every item in the test manifest.'
    )
    parser.add_argument('--manifest', type=Path, default=config.TEST_MANIFEST)
    parser.add_argument('--out', type=Path, default=config.GENERATED_DIR)
    parser.add_argument('--checkpoint', default=config.PRETRAINED)
    parser.add_argument('--mock', action='store_true')
    args = parser.parse_args()
    generate_for_manifest(args.manifest, args.out, args.checkpoint, args.mock)


if __name__ == '__main__':
    main()
