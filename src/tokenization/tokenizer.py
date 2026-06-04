"""Tokenization study (optional bonus): character vs phoneme tokens.

The assignment's optional experiment: compare character-based tokens against
phoneme-based tokens and measure the impact on WER/CER. This is a STUB — the
interface is fixed so the experiment can be wired in later without touching the
rest of the pipeline.

Plan: ``phonemize`` (e.g. via the ``phonemizer`` package / espeak-ng) turns text
into phonemes; feed both variants through model.synthesize and compare with
evaluation.evaluate.
"""

from core.logger import logger


def to_characters(text: str) -> str:
    """Character-level tokenization is what SpeechT5 uses by default — identity."""
    return text


def to_phonemes(text: str) -> str:
    """Convert text to a phoneme string. STUB — returns text unchanged for now."""
    logger.warning('to_phonemes is a STUB — returning text unchanged.')
    return text
