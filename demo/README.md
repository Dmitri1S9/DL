# B1 Droid TTS — local demo

A tiny Star-Wars-themed web page: type text, hear it spoken in the fine-tuned
B1 battle-droid voice. Everything runs locally — the model synthesizes on your
machine, no network needed after the model is cached.

## Run

```bash
# one-time system dep (the VITS tokenizer needs espeak-ng):
#   macOS:   brew install espeak-ng
#   Ubuntu:  sudo apt install espeak-ng
#   Windows: winget install espeak-ng.espeak-ng
pip install flask            # one-time (or: pip install -r requirements-dev.txt)
PYTHONPATH=src python demo/app.py
```

Then open <http://127.0.0.1:5000> and hit **Transmit** (or ⌘/Ctrl + Enter).

## Which voice it uses

- If `models/vits_finetune/epoch_2.pt` exists → the **fine-tuned droid** voice.
- Otherwise it falls back to the **pretrained base** model so the demo still runs.
- Point it at any checkpoint with an env var:

  ```bash
  VITS_DEMO_CKPT=models/vits_finetune/epoch_3.pt PYTHONPATH=src python demo/app.py
  ```

> Don't have the checkpoint locally? Download `epoch_2.pt` from your Colab/Drive
> into `models/vits_finetune/` first (see `colab_train.ipynb`).

## How it works

`demo/app.py` is a ~60-line Flask server: it loads the model once at startup and
exposes `POST /synthesize` (`{"text": "..."}` → a WAV). `demo/index.html` is the
front-end that calls it and plays the audio. Synthesis reuses the project's own
`vits_finetune.synthesize` (same code path as evaluation).
