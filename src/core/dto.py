"""Shared pydantic schemas — the data shapes that flow between pipeline stages.

These are the contract: data / model / evaluation each code against these shapes
instead of each other's internals, which is what lets them be built in parallel.
pydantic gives validation on read (a drifted manifest fails loudly with a clear
error) and clean JSON serialization (``model_dump_json``).

  * TestItem    — one line of the JSONL test manifest (produced by data.prepare)
  * TrainingLog — fine-tuning budget + curve (produced by model.train)
  * EvalResult  — scores for one audio set / one before-after row (evaluation)

Naming: this module is ``dto`` rather than ``models`` to avoid colliding with the
``model`` package (the ML model).
"""

from pydantic import BaseModel, ConfigDict


class TestItem(BaseModel):
    """One evaluation example."""

    id: str  # stable id; also the wav filename stem (<id>.wav)
    text: str  # the sentence to synthesize
    ref_audio: str  # path to the ground-truth recording (used for MCD)

    model_config = ConfigDict(frozen=True)


class TrainingLog(BaseModel):
    """Fine-tuning budget + loss curve — the 'training budget' for the report."""

    status: str
    base_model: str
    dataset: str
    epochs: int
    batch_size: int
    learning_rate: float
    steps: int
    gpu_hours: float
    val_loss_curve: list[float]


class EvalResult(BaseModel):
    """Scores for one set of generated audio (one row of the before/after table)."""

    label: str
    n: int
    wer: float | None = None
    cer: float | None = None
    mcd: float | None = None
