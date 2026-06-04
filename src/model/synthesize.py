"""Speech synthesis: load a SpeechT5 checkpoint and turn text into audio.

The real path loads SpeechT5 (text -> mel) + HiFi-GAN (mel -> wav). The ``mock``
path returns a cheap sine wave so the whole pipeline (and CI) runs end-to-end
without downloading ~2 GB of weights or needing a GPU.

The ``checkpoint`` argument is the swap point of the whole project:

    config.PRETRAINED  -> the base microsoft/speecht5_tts, as-is
    <a directory>      -> a fine-tuned checkpoint produced by model.train

Generated wavs follow the contract: ``<out_dir>/<id>.wav``, 16 kHz mono float32.
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from core import config
from core.contracts import read_manifest
from core.logger import logger


class TTSModel:
    """Wraps SpeechT5 + HiFi-GAN behind a single ``synthesize(text)`` call.

    Heavy imports (torch/transformers) are done lazily inside ``__init__`` so the
    module — and the ``mock`` synthesis path — stay importable without them.
    """

    def __init__(self, checkpoint: str = config.PRETRAINED):
        import torch
        from transformers import (
            SpeechT5ForTextToSpeech,
            SpeechT5HifiGan,
            SpeechT5Processor,
        )

        from model.speaker import load_speaker_embedding

        self._torch = torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        source = (
            config.TTS_MODEL_ID if checkpoint == config.PRETRAINED else str(checkpoint)
        )
        logger.info(f'Loading SpeechT5 from {source!r} on {self.device}...')

        # Load the processor from the checkpoint dir when a fine-tuned model saved
        # one there; fall back to the base model otherwise.
        try:
            self.processor = SpeechT5Processor.from_pretrained(
                source,
                cache_dir=str(config.MODELS_DIR),
            )
        except (OSError, ValueError):
            self.processor = SpeechT5Processor.from_pretrained(
                config.TTS_MODEL_ID,
                cache_dir=str(config.MODELS_DIR),
            )

        self.model = SpeechT5ForTextToSpeech.from_pretrained(
            source, cache_dir=str(config.MODELS_DIR)
        ).to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained(
            config.VOCODER_ID, cache_dir=str(config.MODELS_DIR)
        ).to(self.device)
        self.speaker_embeddings = load_speaker_embedding().to(self.device)
        logger.success('TTS model ready.')

    def synthesize(self, text: str) -> np.ndarray:
        inputs = self.processor(text=text, return_tensors='pt').to(self.device)
        with self._torch.no_grad():
            speech = self.model.generate_speech(
                inputs['input_ids'],
                self.speaker_embeddings,
                vocoder=self.vocoder,
            )
        return speech.cpu().numpy().astype(np.float32)


def mock_synthesize(text: str) -> np.ndarray:
    """Deterministic sine whose length scales with the text — no model needed."""
    seconds = max(0.4, 0.06 * len(text))
    n = int(seconds * config.SAMPLE_RATE)
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
    tag = 'mock' if mock else checkpoint
    logger.info(f'Synthesizing {len(items)} clips ({tag}) -> {out_dir}')
    for item in items:
        wav = synth(item.text)
        sf.write(str(out_dir / f'{item.id}.wav'), wav, config.SAMPLE_RATE)
    logger.success(f'Wrote {len(items)} wavs to {out_dir}')
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Synthesize audio for every item in the test manifest.'
    )
    parser.add_argument('--manifest', type=Path, default=config.TEST_MANIFEST)
    parser.add_argument('--out', type=Path, default=config.GENERATED_DIR)
    parser.add_argument(
        '--checkpoint',
        default=config.PRETRAINED,
        help=f'"{config.PRETRAINED}" or a fine-tuned checkpoint directory',
    )
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Use a sine-wave stub instead of loading the real model',
    )
    args = parser.parse_args()
    generate_for_manifest(args.manifest, args.out, args.checkpoint, args.mock)


if __name__ == '__main__':
    main()
