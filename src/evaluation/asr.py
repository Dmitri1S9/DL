"""Whisper ASR helper: load the model once, transcribe to normalized text.

Every WER/CER path (``metrics``, ``evaluate_dataset``, ``eval_epochs``) needs the
same thing: run Whisper on a clip and lowercase/strip the transcript. Centralizing
it here keeps a single cached model per size (Whisper is heavy to load) and one
transcription call, instead of three near-identical copies.

``fp16=False`` is used everywhere: evaluation runs on CPU (where Whisper forces
fp32 anyway), so this matches the existing behavior and silences the fp16 warning.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=None)
def _load_model(size: str):
    """Load (and cache) a Whisper model by name. Lazy import — heavy dependency."""
    import whisper

    return whisper.load_model(size)


def transcribe(audio, size: str = 'base') -> str:
    """Transcribe one clip with Whisper, returning stripped/lowercased text.

    ``audio`` may be a wav file path or an in-memory float32 array — Whisper
    accepts both. The model for ``size`` is loaded once and reused across calls.
    """
    result = _load_model(size).transcribe(audio, language='en', fp16=False)
    return result['text'].strip().lower()
