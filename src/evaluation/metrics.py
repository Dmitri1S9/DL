"""Objective TTS metrics (all: lower = better).

WER / CER — Word / Character Error Rate: how well an ASR model (Whisper)
            transcribes our generated audio back to the original text.
MCD       — Mel Cepstral Distortion (via pymcd): DTW-aligned mel-cepstral
            distance between generated and reference audio.

These are the primitives; the manifest-driven runner lives in evaluation.evaluate.
"""

from core.logger import logger
from evaluation.asr import transcribe


def compute_wer_cer(
    audio_paths: list[str], reference_texts: list[str]
) -> tuple[float, float]:
    """WER and CER from a single Whisper pass over ``audio_paths``.

    Each clip is transcribed once and scored for both metrics against
    ``reference_texts`` (lower = better). Returns ``(wer, cer)``.
    """
    from jiwer import cer, wer

    hypotheses = [transcribe(path) for path in audio_paths]
    references = [text.lower() for text in reference_texts]
    word_error = float(wer(references, hypotheses))
    char_error = float(cer(references, hypotheses))
    logger.info(
        f'WER: {word_error:.4f} ({word_error * 100:.1f}%)  '
        f'CER: {char_error:.4f} ({char_error * 100:.1f}%)'
    )
    return word_error, char_error


def compute_mcd(ref_path: str, gen_path: str) -> float:
    """Mel Cepstral Distortion (dB) between two wav files, DTW-aligned.

    Uses pymcd (WORLD mel-cepstral analysis + DTW), so values land on the canonical
    MCD scale (~5-8 dB is typical for decent TTS). Lower = better. Takes file paths
    because pymcd reads the audio itself.
    """
    from pymcd.mcd import Calculate_MCD

    return float(Calculate_MCD(MCD_mode='dtw').calculate_mcd(ref_path, gen_path))
