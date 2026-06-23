"""Dataset-ceiling eval: WER/CER of the *real* B1 recordings themselves.

Dmitrii's point: no model trained on B1 audio can be transcribed *better* than the
B1 audio itself. Whisper was trained on human voices, so the robotic RVC droid
timbre scores a high WER no matter what — that is the ceiling. This script measures
it directly: take held-out B1 ground-truth clips and run Whisper on them (no model,
no checkpoint, no generation involved).

Reads the B1 dataset (``Dmi1tr13/ljspeech-b1``) straight from the Hub — only the
last parquet shard is downloaded, and we score the held-out tail of it. Output is a
small JSON next to the other eval logs.

Run:  PYTHONPATH=src python -m evaluation.evaluate_dataset --n 100
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from core import config
from core.logger import logger

DATASET_REPO = 'Dmi1tr13/ljspeech-b1'
# 8 shards of ~1637 clips, in LJSpeech order; the tail is the held-out region.
LAST_SHARD = 'data/train-00007-of-00008.parquet'
WHISPER_SR = 16000


def _decode_clip(audio_struct: dict) -> np.ndarray:
    """Decode a HF Audio struct ({bytes, path}) to a 16 kHz mono float32 array."""
    import torch
    import torchaudio

    wav, sr = sf.read(io.BytesIO(audio_struct['bytes']), dtype='float32')
    if wav.ndim > 1:  # stereo -> mono
        wav = wav.mean(axis=1)
    if sr != WHISPER_SR:
        tensor = torch.from_numpy(wav).unsqueeze(0)
        wav = (
            torchaudio.functional.resample(tensor, sr, WHISPER_SR)
            .squeeze(0)
            .numpy()
            .astype(np.float32)
        )
    return np.ascontiguousarray(wav, dtype=np.float32)


def evaluate_dataset(
    n: int = 100,
    shard: str = LAST_SHARD,
    whisper_size: str = 'base',
    out: Path | None = None,
) -> dict:
    """Transcribe the last ``n`` held-out B1 clips and score WER/CER against text."""
    import pyarrow.parquet as pq
    import whisper
    from huggingface_hub import hf_hub_download
    from jiwer import cer, wer

    logger.info(f'Downloading B1 shard {shard!r} (held-out tail only)...')
    path = hf_hub_download(
        DATASET_REPO, shard, repo_type='dataset', local_dir=str(config.DATA_DIR / 'b1_shards')
    )
    table = pq.read_table(path)
    total = table.num_rows
    start = max(0, total - n)
    logger.info(f'Shard has {total} clips; scoring the last {total - start} (the ceiling set).')

    audio_col = table['audio'].to_pylist()[start:]
    texts = table['text'].to_pylist()[start:]

    logger.info(f'Loading Whisper ({whisper_size}) — CPU transcription, be patient...')
    model = whisper.load_model(whisper_size)

    hypotheses: list[str] = []
    references: list[str] = []
    for i, (clip, text) in enumerate(zip(audio_col, texts)):
        wav = _decode_clip(clip)
        hyp = model.transcribe(wav, language='en', fp16=False)['text'].strip().lower()
        hypotheses.append(hyp)
        references.append(text.strip().lower())
        if (i + 1) % 10 == 0:
            logger.info(f'  transcribed {i + 1}/{len(texts)}')

    result = {
        'eval': 'b1-dataset-ceiling',
        'dataset': DATASET_REPO,
        'shard': shard,
        'n': len(references),
        'whisper': whisper_size,
        'wer': float(wer(references, hypotheses)),
        'cer': float(cer(references, hypotheses)),
    }
    logger.success(
        f"DATASET CEILING (real B1 audio) — WER {result['wer'] * 100:.1f}% / "
        f"CER {result['cer'] * 100:.1f}% over {result['n']} clips"
    )

    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding='utf-8')
        logger.info(f'Results -> {out}')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='WER/CER of the real B1 recordings (ceiling).')
    parser.add_argument('--n', type=int, default=100, help='How many held-out clips to score.')
    parser.add_argument('--shard', default=LAST_SHARD)
    parser.add_argument('--whisper', default='base')
    parser.add_argument('--out', type=Path, default=config.ROOT / 'logs' / 'eval_ceiling.json')
    args = parser.parse_args()
    evaluate_dataset(args.n, args.shard, args.whisper, args.out)


if __name__ == '__main__':
    main()
