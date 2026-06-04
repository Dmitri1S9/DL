"""Objective TTS metrics (all: lower = better).

WER / CER — Word / Character Error Rate: how well an ASR model (Whisper)
            transcribes our generated audio back to the original text.
MCD       — Mel Cepstral Distortion: how close the generated mel-spectrogram is
            to the reference recording.

These are the primitives; the manifest-driven runner lives in evaluation.evaluate.
"""

import librosa
import numpy as np

from core import config
from core.logger import logger


def _transcribe(audio_paths: list[str]) -> list[str]:
    """Transcribe each wav with Whisper (lazy import — heavy dependency)."""
    import whisper

    model = whisper.load_model('base')
    return [
        model.transcribe(path, language='en')['text'].strip().lower()
        for path in audio_paths
    ]


def compute_wer(audio_paths: list[str], reference_texts: list[str]) -> float:
    """Word Error Rate between Whisper transcripts and the reference texts."""
    from jiwer import wer

    hypotheses = _transcribe(audio_paths)
    references = [t.lower() for t in reference_texts]
    score = wer(references, hypotheses)
    logger.info(f'WER: {score:.4f} ({score * 100:.1f}%)')
    return score


def compute_cer(audio_paths: list[str], reference_texts: list[str]) -> float:
    """Character Error Rate between Whisper transcripts and the reference texts."""
    from jiwer import cer

    hypotheses = _transcribe(audio_paths)
    references = [t.lower() for t in reference_texts]
    score = cer(references, hypotheses)
    logger.info(f'CER: {score:.4f} ({score * 100:.1f}%)')
    return score


def compute_mcd(
    ref_audio: np.ndarray,
    gen_audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
) -> float:
    """Mel Cepstral Distortion between reference and generated audio (in dB)."""

    def extract_mfcc(audio: np.ndarray) -> np.ndarray:
        return librosa.feature.mfcc(y=audio.astype(float), sr=sr, n_mfcc=24)

    ref_mfcc = extract_mfcc(ref_audio)
    gen_mfcc = extract_mfcc(gen_audio)

    # Align lengths
    min_len = min(ref_mfcc.shape[1], gen_mfcc.shape[1])
    ref_mfcc = ref_mfcc[:, :min_len]
    gen_mfcc = gen_mfcc[:, :min_len]

    # MCD formula (excluding the zeroth coefficient)
    diff = ref_mfcc[1:] - gen_mfcc[1:]
    return (10 / np.log(10)) * np.sqrt(2) * np.mean(np.sqrt(np.sum(diff**2, axis=0)))
