"""Objective TTS metrics (all: lower = better).

WER / CER — Word / Character Error Rate: how well an ASR model (Whisper)
            transcribes our generated audio back to the original text.
MCD       — Mel Cepstral Distortion (via pymcd): DTW-aligned mel-cepstral
            distance between generated and reference audio.

These are the primitives; the manifest-driven runner lives in evaluation.evaluate.
"""

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


def compute_mcd(ref_path: str, gen_path: str) -> float:
    """Mel Cepstral Distortion (dB) between two wav files, DTW-aligned.

    Uses pymcd (WORLD mel-cepstral analysis + DTW), so values land on the canonical
    MCD scale (~5-8 dB is typical for decent TTS). Lower = better. Takes file paths
    because pymcd reads the audio itself.
    """
    from pymcd.mcd import Calculate_MCD

    return float(Calculate_MCD(MCD_mode='dtw').calculate_mcd(ref_path, gen_path))
