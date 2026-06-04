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
TTS_MODEL_ID = 'microsoft/speecht5_tts'  # acoustic model: text -> mel
VOCODER_ID = 'microsoft/speecht5_hifigan'  # vocoder: mel -> waveform
SPEAKER_XVECTOR_REPO = 'Matthijs/cmu-arctic-xvectors'

# ── Checkpoints ─────────────────────────────────────────────────────────────────
PRETRAINED = 'pretrained'  # sentinel: use the base model as-is
FINETUNED_DIR = MODELS_DIR / 'finetuned'  # output of model.train

# ── Dataset ─────────────────────────────────────────────────────────────────────
DATASET_ID = 'keithito/lj_speech'
TEST_SIZE = 500  # last N clips held out as the test set

# ── Training (consumed by model.train) ──────────────────────────────────────────
NUM_EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
