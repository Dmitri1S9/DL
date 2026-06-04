# Project 13 — Text to Speech

Fine-tune **SpeechT5** to turn English text into speech, and evaluate it.
*192.151 Introduction to Deep Learning, 2026S.*

## What it does

Two-stage neural TTS:

```
  text  ──►  [ SpeechT5 ]  ──►  mel-spectrogram  ──►  [ HiFi-GAN ]  ──►  waveform (.wav)
             acoustic model                            vocoder
```

- **Acoustic model** — SpeechT5 (text → mel), **fine-tuned** on LJSpeech.
- **Vocoder** — HiFi-GAN (mel → waveform), pretrained, used as-is.

We **fine-tune** rather than train from scratch: SpeechT5 is a strong pretrained
checkpoint that adapts to LJSpeech in a few GPU-hours — realistic for the course's
compute budget, while still being real training (we report epochs, batch size, and
GPU-hours). Architecturally this is "text → mel + HiFi-GAN", the same class as the
assignment's Option A (FastSpeech2 + HiFi-GAN).

## Architecture (contract-first)

Pipeline stages talk through small **contracts** in `src/core`, not each other's
internals — so the team builds in parallel and the model stays swappable:

- **Test manifest** — `data/prepare.py` writes one `{id, text, ref_audio}` per held-out
  clip; the model and the evaluation both read it.
- **Checkpoint swap** — `--checkpoint pretrained` vs `--checkpoint models/finetuned` is
  the only change needed to produce the before/after comparison.

## Project structure

```
src/
├── core/          config.py · contracts.py · dto.py · logger.py   (shared infra)
├── data/          download.py · prepare.py · prepare_training.py
├── model/         synthesize.py · speaker.py · train.py
├── evaluation/    metrics.py · evaluate.py
├── effects/       effects.py        (fun voice effects — NOT in the eval path)
└── tokenization/  tokenizer.py      (char vs phoneme study — stub)
tests/             test_smoke.py     (offline end-to-end wiring test)
scripts/           colab_setup.sh · sanity_check.py
docs/              training-on-colab.md
Makefile · requirements.txt · requirements-dev.txt · pyproject.toml
```

`src` is the import root, so modules import each other as top-level packages
(`from core.logger import logger`). The Makefile sets `PYTHONPATH=src` for you; to run a
module by hand use `PYTHONPATH=src python -m <module>`.

## Setup

```bash
make install        # creates .venv and installs pinned requirements + dev tools
```

Dependencies are pinned to exact, verified versions in `requirements.txt`. After
`make install`, the Makefile automatically uses `.venv`.

## Usage

| Command | What it does |
|---|---|
| `make all` | **Offline mock** pipeline (no downloads, runs in seconds) — proves the wiring |
| `make data` | Download LJSpeech (~2.6 GB, once) |
| `make prepare` | Build the test set: manifest + 16 kHz reference audio |
| `make train` | Fine-tune SpeechT5 *(currently a MOCK — see Status)* |
| `make generate CKPT=models/finetuned` | Synthesize the test set with a checkpoint |
| `make eval LABEL=finetuned` | Score generated audio (WER/CER/MCD) |
| `make smoke` | Run the pytest smoke test |
| `make lint` / `make format` | `ruff check` / `ruff format` |
| `make clean` | Remove generated outputs |

**Real before/after run** (after training produces a checkpoint):

```bash
make prepare
make generate CKPT=pretrained         && make eval LABEL=pretrained
make generate CKPT=models/finetuned   && make eval LABEL=finetuned
```

Training the model itself runs on GPU (Colab/Kaggle) — see
[`docs/training-on-colab.md`](docs/training-on-colab.md).

## Evaluation

- **WER / CER** — an off-the-shelf ASR model (Whisper) transcribes our generated audio;
  lower = more intelligible.
- **MCD** — Mel Cepstral Distortion (via `pymcd`, DTW-aligned WORLD mel-cepstral) between
  our audio and the real recording; lower = closer to natural (~5–8 dB is typical).

Baseline (pretrained SpeechT5, sanity sample — fine-tuning numbers to follow):

| model | WER | CER | MCD |
|---|---|---|---|
| pretrained | 5.5% | 1.6% | 6.38 dB |

## Status

| Component | State |
|---|---|
| `data/download.py`, `data/prepare.py`, `data/prepare_training.py` | ✅ real |
| `model/synthesize.py` (+ checkpoint swap) | ✅ real |
| `model/train.py` (fine-tuning loop) | 🚧 **MOCK** — writes a placeholder log |
| `evaluation/metrics.py`, `evaluation/evaluate.py` | ✅ real (WER/CER/MCD) |
| `effects/`, `tokenization/` | demo / stub (optional) |

## Who does what

| Owner | Area | Files |
|---|---|---|
| **Dima** | model & fine-tuning | `src/model/` |
| **Emir** | infra, contracts, data | `src/core/`, `src/data/`, `Makefile` |
| **Ilya** | evaluation | `src/evaluation/` |

## Conventions

- **Import root** is `src` (run with `PYTHONPATH=src`).
- **Style**: single quotes, line length 88, enforced by `ruff` (`make lint` / `make format`).
- **Logging**: `loguru` via `core.logger` (no bare `print`).
- **Dataset**: LJSpeech (single female voice); last 500 clips are the held-out test set.
