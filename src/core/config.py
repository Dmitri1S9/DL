"""Central configuration: paths, model ids, audio params, hyperparameters.

Single source of truth. Every module imports paths/ids from here instead of
hard-coding them, so changing a path or a model id happens in one place.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # core -> src -> project root

# ── Directories (created on demand by whichever module writes to them) ──────────
DATA_DIR = ROOT / 'data'
MODELS_DIR = ROOT / 'models'
AUDIO_DIR = ROOT / 'audio'

# ── Evaluation test-set layout (the data contract lives in core.contracts) ──────
TEST_MANIFEST = DATA_DIR / 'test_manifest.jsonl'  # one TestItem per line
REFERENCE_DIR = DATA_DIR / 'reference'  # ground-truth wavs from LJSpeech
GENERATED_DIR = AUDIO_DIR / 'generated'  # our synthesized wavs

# ── Audio ───────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000

# ── Models (HuggingFace ids) ────────────────────────────────────────────────────
TTS_MODEL_ID = 'kakao-enterprise/vits-ljs'   # VITS end-to-end, trained on LJSpeech
VOCODER_ID = 'microsoft/speecht5_hifigan'    # legacy SpeechT5 vocoder — not used by VITS
SPEAKER_XVECTOR_REPO = 'Matthijs/cmu-arctic-xvectors'  # legacy — not used by VITS

# ── Checkpoints ─────────────────────────────────────────────────────────────────
PRETRAINED = 'pretrained'  # sentinel: use the base model as-is
FINETUNED_DIR = MODELS_DIR / 'finetuned'  # output of model.train

# ── Dataset ─────────────────────────────────────────────────────────────────────
DATASET_ID = 'keithito/lj_speech'
TEST_SIZE = 500  # last N clips held out as the test set

# ── Training (consumed by model.train) ──────────────────────────────────────────
NUM_EPOCHS = 5
BATCH_SIZE = 8  # lower to ~4 (and raise GRAD_ACCUM) if the Colab GPU runs out of VRAM
LEARNING_RATE = 1e-5
GRAD_ACCUM = 8  # effective batch = BATCH_SIZE * GRAD_ACCUM
MAX_STEPS = 4000  # tutorial-scale budget; scale down for a quicker first run
WARMUP_STEPS = 500
EVAL_STEPS = 1000
SAVE_STEPS = 1000
LOGGING_STEPS = 25
FP16 = True  # mixed precision (GPU only)
MAX_TOKENS = 200  # drop training examples whose tokenized text exceeds this
VAL_SIZE = 200  # clips carved off the train split for the trainer's eval
XVECTOR_DIM = 512  # speaker x-vector dimensionality
