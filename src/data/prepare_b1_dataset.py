"""Package LJSpeech-B1 (RVC output) as a HuggingFace AudioFolder dataset.

RVC writes 48kHz wavs; VITS (our TTS backbone) is 22050Hz, so we resample down
on the way out. Output layout matches HF's AudioFolder convention:
  audio/ljspeech_b1/
    LJ001-0001_b1.wav, ...
    metadata.csv          (file_name,text)

This folder can be pushed directly to the Hub:
  huggingface-cli upload <user>/ljspeech-b1 audio/ljspeech_b1 --repo-type dataset

Usage:
    PYTHONPATH=src python -m data.prepare_b1_dataset
"""

import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import soundfile as sf
import soxr
from tqdm import tqdm

from core import config
from core.logger import logger

SRC_DIR = config.AUDIO_DIR / 'rvc_output'
OUT_DIR = config.AUDIO_DIR / 'ljspeech_b1'
TARGET_SR = 22050


def resample_one(paths: tuple[Path, Path]) -> None:
    src_path, out_path = paths
    audio, sr = sf.read(str(src_path), dtype='float32')
    if sr != TARGET_SR:
        audio = soxr.resample(audio, sr, TARGET_SR)
    sf.write(str(out_path), audio, TARGET_SR, subtype='PCM_16')


def resample_all() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wavs = sorted(SRC_DIR.glob('*_b1.wav'))
    todo = [(p, OUT_DIR / p.name) for p in wavs if not (OUT_DIR / p.name).exists()]
    logger.info(f'Total: {len(wavs)} | To resample: {len(todo)} -> {TARGET_SR} Hz')

    if todo:
        with ProcessPoolExecutor() as pool:
            list(tqdm(pool.map(resample_one, todo), total=len(todo), unit='file'))

    logger.success(f'Resampled wavs -> {OUT_DIR}')


def write_metadata() -> None:
    meta_src = config.DATA_DIR / 'lj_metadata.csv'
    rows: list[tuple[str, str]] = []
    for line in meta_src.read_text('utf-8').strip().splitlines():
        parts = line.split('|')
        if len(parts) < 3:
            continue
        lj_id, _raw, normalized = parts[0], parts[1], parts[2]
        fname = f'{lj_id}_b1.wav'
        if (OUT_DIR / fname).exists():
            rows.append((fname, normalized))

    meta_path = OUT_DIR / 'metadata.csv'
    with open(meta_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file_name', 'text'])
        writer.writerows(rows)

    logger.success(f'metadata.csv -> {meta_path} ({len(rows)} rows)')


def main() -> None:
    resample_all()
    write_metadata()


if __name__ == '__main__':
    main()
