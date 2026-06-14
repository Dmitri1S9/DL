# Fine-tuning SpeechT5 on Google Colab

A short runbook for running `model.train` on a free Colab GPU. Colab gives you a
**temporary** cloud machine with a GPU: it starts empty, and everything on it is
wiped when the session ends — so we clone the code in, install deps, train, and
**save the checkpoint out** before disconnecting.

In a Colab cell, a line starting with `!` is a shell command; `%cd` changes the
notebook's working directory.

## Prerequisite

The repo must be pushed to GitHub so Colab can clone it:

```bash
git push        # from your machine, once your work is committed
```

Replace `<your-repo-url>` below with the real clone URL.

## Steps

**1. Enable the GPU** (one-time menu click):
`Runtime → Change runtime type → Hardware accelerator: GPU → Save`

**2. Clone the repo onto the Colab machine:**

```python
!git clone <your-repo-url> project
%cd project
```

**3. Install dependencies:**

```python
!bash scripts/colab_setup.sh
```

This installs `requirements.txt` and prints whether the GPU is visible to PyTorch.

**4. Load the LJSpeech-B1 dataset** (training targets — original LJSpeech audio
converted to the B1 Battle Droid voice via RVC, resampled to 22050Hz to match
VITS). Public dataset, no token needed: `Dmi1tr13/ljspeech-b1`.

```python
from datasets import load_dataset
b1_ds = load_dataset("Dmi1tr13/ljspeech-b1", split="train")
```

Each item is `{"audio": {...}, "text": "..."}` (audio already at 22050Hz) — `text`
is the original LJSpeech normalized transcript (the B1 conversion changes
timbre/voice, not pronunciation).

**5. Run training** (note the `PYTHONPATH=src` prefix — that is how the `core`,
`data`, `model` packages are found):

```python
!PYTHONPATH=src python -m model.train
```

The checkpoint is written to `models/finetuned/` (per `core.config.FINETUNED_DIR`).

**6. Save the checkpoint before the session dies** — mount Google Drive and copy it
out, otherwise it is lost when Colab disconnects:

```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r models/finetuned "/content/drive/MyDrive/tts_finetuned"
```

To use it later for inference/eval, copy it back next to the repo and point the
checkpoint at it:

```python
!PYTHONPATH=src python -m model.synthesize --checkpoint models/finetuned
```

## Notes

- **Ephemeral machine:** the VM (and `models/`, `data/`, `logs/`) is erased at the
  end of the session. Always do Step 6.
- **Time limits:** free Colab can disconnect after a while / on idle. Keep the run
  within a couple of hours (scale `core.config.MAX_STEPS` down if needed).
- **Out of memory?** Lower `BATCH_SIZE` (e.g. to 4) and raise `GRAD_ACCUM` in
  `core/config.py`.
- The training data comes from `data.prepare_training.load_training_splits()`, which
  downloads LJSpeech (~2.6 GB) on first use.
