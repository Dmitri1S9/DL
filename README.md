# Project 13 — Text to Speech

Fine-tune **VITS** to turn English text into speech, and evaluate it.
*192.151 Introduction to Deep Learning, 2026S.*

## What it does

Single-stage, end-to-end neural TTS:

```
  text  ──►  [            VITS            ]  ──►  waveform (.wav)
             text encoder + flow-based prior
             + stochastic duration predictor
             + posterior encoder (train only)
             + HiFi-GAN-style decoder
```

- **Model** — VITS (`kakao-enterprise/vits-ljs`), pretrained on LJSpeech. One model
  maps text directly to a raw waveform — no separate vocoder and no speaker
  embeddings to manage.

We **fine-tune** rather than train from scratch: `vits-ljs` is already a strong
LJSpeech checkpoint, so fine-tuning onto our target voice adapts it in a few
GPU-hours — realistic for the course's compute budget, while still being real
training (we report epochs, batch size, and GPU-hours). This is the assignment's
**Option C** (VITS, end-to-end text → waveform with latent-variable modeling).

> We started from the SpeechT5 + HiFi-GAN plan (Option A) and migrated to VITS
> (Option C): one model instead of two, and noticeably better audio quality
> out of the box.

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
│                  bake_dataset.py · prepare_b1_dataset.py · push_b1_dataset.py
│                  (B1 voice dataset pipeline — see "Training data" below)
├── model/         synthesize.py     (VITS inference, used by generate/eval)
├── vits_finetune/ config.py · model_config.py · dataset.py · collate.py ·
│                  audio.py · model.py · losses.py · checkpoint.py ·
│                  train.py · synthesize.py   (VITS fine-tuning — in progress)
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
| `make train` | Fine-tune VITS via `vits_finetune` *(in progress — see Status)* |
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

## Training data: the B1 voice dataset

Fine-tuning target: **[`Dmi1tr13/ljspeech-b1`](https://huggingface.co/datasets/Dmi1tr13/ljspeech-b1)**
on the HF Hub — LJSpeech text resynthesized with `vits-ljs` and voice-converted to
the *B1 Battle Droid* voice via RVC, resampled to 22050 Hz to match VITS. Public,
no token needed:

```python
from datasets import load_dataset
b1_ds = load_dataset("Dmi1tr13/ljspeech-b1", split="train")
# {"audio": {"array": ..., "sampling_rate": 22050}, "text": "..."}
```

13,100 examples; `text` is the original LJSpeech normalized transcript (the B1
conversion changes timbre, not pronunciation). Pipeline that produced it:
`data/bake_dataset.py` (VITS synthesis) → B1 RVC Docker (voice conversion) →
`data/prepare_b1_dataset.py` (resample + metadata) → `data/push_b1_dataset.py`
(push to the Hub).

## Evaluation

- **WER / CER** — an off-the-shelf ASR model (Whisper) transcribes our generated audio;
  lower = more intelligible.
- **MCD** — Mel Cepstral Distortion (via `pymcd`, DTW-aligned WORLD mel-cepstral) between
  our audio and the real recording; lower = closer to natural (~5–8 dB is typical).

Baseline (pretrained VITS, sanity sample — fine-tuning numbers to follow):

| model | WER | CER | MCD |
|---|---|---|---|
| pretrained | 5.5% | 1.6% | 6.38 dB |

## Status

| Component | State |
|---|---|
| `data/download.py`, `data/prepare.py`, `data/prepare_training.py` | ✅ real |
| `data/bake_dataset.py`, `data/prepare_b1_dataset.py`, `data/push_b1_dataset.py` | ✅ real — produced `Dmi1tr13/ljspeech-b1` |
| `model/synthesize.py` (VITS, + checkpoint swap) | ✅ real |
| `vits_finetune/` (VITS fine-tuning loop) | 🚧 in progress — config/dataset/collate/checkpoint wired up; `model.forward_train` + losses still stubs |
| `evaluation/metrics.py`, `evaluation/evaluate.py` | ✅ real (WER/CER/MCD) |
| `effects/`, `tokenization/` | demo / stub (optional) |

## Who does what

| Owner | Area | Files |
|---|---|---|
| **Dima** | model & fine-tuning | `src/model/`, `src/vits_finetune/` |
| **Emir** | infra, contracts, data | `src/core/`, `src/data/`, `Makefile` |
| **Ilya** | evaluation | `src/evaluation/` |

## Conventions

- **Import root** is `src` (run with `PYTHONPATH=src`).
- **Style**: single quotes, line length 88, enforced by `ruff` (`make lint` / `make format`).
- **Logging**: `loguru` via `core.logger` (no bare `print`).
- **Dataset**: LJSpeech (single female voice); last 500 clips are the held-out test
  set. Fine-tuning target is the B1-voiced variant, [`Dmi1tr13/ljspeech-b1`](https://huggingface.co/datasets/Dmi1tr13/ljspeech-b1)
  (see "Training data" above) — the eval test set stays on original LJSpeech audio
  so before/after comparisons are apples-to-apples on the same prompts.




## DATASET IS HERE !!!!!!
https://huggingface.co/datasets/Dmi1tr13/ljspeech-b1

