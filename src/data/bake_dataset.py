"""Overnight dataset baking: LJSpeech text -> VITS synthesis -> save wavs.

Step 1 of Variant B pipeline:
  LJSpeech normalized_text  ->  VITS (GPU)  ->  rvc_input/*.wav

After this script finishes, run B1 Docker for Step 2:
  docker run --gpus all -v .../rvc_input:/input -v .../rvc_output:/output \\
    -v .../models/rvc:/models --entrypoint python b1_droid /app/run.py

Usage:
    python -m data.bake_dataset
    python -m data.bake_dataset --out audio/rvc_input --limit 100
"""

import argparse
import os
import sys
import time
import tarfile
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from core import config
from core.logger import logger

LJSPEECH_ARCHIVE = config.DATA_DIR / "LJSpeech-1.1.tar.bz2"
LJSPEECH_META_CACHE = config.DATA_DIR / "lj_metadata.csv"


def wait_for_archive(path: Path, poll_sec: int = 30) -> None:
    logger.info(f"Waiting for {path.name} to be ready (download may still be running)...")
    while True:
        if path.exists():
            try:
                # Try opening — will fail with PermissionError if still being written
                with open(path, "rb") as f:
                    f.read(1)
                logger.success(f"{path.name} is ready.")
                return
            except PermissionError:
                pass
        size_mb = path.stat().st_size / 1024**2 if path.exists() else 0
        logger.info(f"  Still downloading... {size_mb:.0f} MB. Retrying in {poll_sec}s.")
        time.sleep(poll_sec)


def extract_metadata(archive: Path, cache: Path) -> list[tuple[str, str]]:
    """Returns list of (id, normalized_text) from LJSpeech metadata.csv."""
    if cache.exists():
        logger.info(f"Using cached metadata: {cache}")
        rows = []
        for line in cache.read_text("utf-8").strip().splitlines():
            parts = line.split("|")
            if len(parts) >= 3:
                rows.append((parts[0], parts[2]))
        return rows

    logger.info(f"Extracting metadata from {archive.name}...")
    with tarfile.open(str(archive), "r:bz2") as tf:
        f = tf.extractfile("LJSpeech-1.1/metadata.csv")
        lines = f.read().decode("utf-8").strip().splitlines()

    cache.write_text("\n".join(lines), encoding="utf-8")
    logger.success(f"Metadata cached -> {cache} ({len(lines)} lines)")

    rows = []
    for line in lines:
        parts = line.split("|")
        if len(parts) >= 3:
            rows.append((parts[0], parts[2]))
    return rows


def synthesize_all(rows: list[tuple[str, str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load VITS model once
    from model.synthesize import TTSModel, VITS_SAMPLE_RATE
    model = TTSModel()

    # Resume: skip already synthesized
    todo = [(id_, text) for id_, text in rows
            if not (out_dir / f"{id_}.wav").exists()]
    done_count = len(rows) - len(todo)
    logger.info(f"Total: {len(rows)} | Already done: {done_count} | To do: {len(todo)}")

    errors = []
    t0 = time.time()

    with open(out_dir / "synth_errors.log", "a") as err_f:
        for id_, text in tqdm(todo, unit="clip"):
            out_path = out_dir / f"{id_}.wav"
            try:
                audio = model.synthesize(text)
                sf.write(str(out_path), audio, VITS_SAMPLE_RATE)
            except Exception as e:
                err_f.write(f"{id_}: {e}\n")
                err_f.flush()
                errors.append(id_)
                tqdm.write(f"ERROR {id_}: {e}")

    elapsed = time.time() - t0
    done = len(todo) - len(errors)
    logger.success(f"Done: {done}/{len(todo)} in {elapsed/3600:.1f}h | errors: {len(errors)}")
    if errors:
        logger.warning(f"Error log: {out_dir / 'synth_errors.log'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=config.AUDIO_DIR / "rvc_input")
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    wait_for_archive(LJSPEECH_ARCHIVE)
    rows = extract_metadata(LJSPEECH_ARCHIVE, LJSPEECH_META_CACHE)

    if args.limit:
        rows = rows[: args.limit]
        logger.info(f"Limited to {args.limit} clips")

    logger.info(f"Synthesizing {len(rows)} clips -> {args.out}")
    synthesize_all(rows, args.out)


if __name__ == "__main__":
    main()
