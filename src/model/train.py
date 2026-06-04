"""Fine-tune SpeechT5 on LJSpeech.

MOCK for now: it does NOT train. It writes a fine-tuned-checkpoint directory plus
a ``training_log.json`` with a plausible loss curve and budget, so the rest of
the pipeline (generate -> evaluate, the before/after table) can run end-to-end.

Replace ``_mock_train`` with a real HuggingFace ``Trainer`` loop (data collator,
``TrainingArguments``, speaker embeddings). See docs/background-and-model-choice.md.
"""

from pathlib import Path

from core import config
from core.dto import TrainingLog
from core.logger import logger


def _mock_train() -> TrainingLog:
    """Pretend to fine-tune; return a fake-but-structurally-real training log."""
    steps = config.NUM_EPOCHS * 100
    loss_curve = [round(2.0 * 0.97**i, 4) for i in range(0, steps, 100)]
    return TrainingLog(
        status='MOCK — no real training happened yet',
        base_model=config.TTS_MODEL_ID,
        dataset=config.DATASET_ID,
        epochs=config.NUM_EPOCHS,
        batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        steps=steps,
        gpu_hours=0.0,
        val_loss_curve=loss_curve,
    )


def train(output_dir: Path = config.FINETUNED_DIR) -> Path:
    """Produce a fine-tuned checkpoint directory (mock) and a training log."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = _mock_train()
    log_path = output_dir / 'training_log.json'
    log_path.write_text(log.model_dump_json(indent=2), encoding='utf-8')
    logger.warning('train.py is a MOCK — wrote a placeholder checkpoint + log.')
    logger.info(f'Training log -> {log_path}')
    return output_dir


if __name__ == '__main__':
    train()
