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


def compute_f0_rmse(ref_path, gen_path):
    """F0 RMSE in Hz between two wavs (pitch difference, lower = better)."""
    import librosa
    import numpy as np

    def get_f0(path):
        y, sr = librosa.load(path, sr=22050)
        f0, _, _ = librosa.pyin(y, fmin=65, fmax=400, sr=sr)
        return f0[~np.isnan(f0)]  # keep only voiced frames

    a = get_f0(ref_path)
    b = get_f0(gen_path)
    if len(a) == 0 or len(b) == 0:
        return float('nan')

    # the two pitch curves have different lengths, so align them with dtw first
    _, path = librosa.sequence.dtw(X=a.reshape(1, -1), Y=b.reshape(1, -1))
    diffs = [a[i] - b[j] for i, j in path]
    return float(np.sqrt(np.mean(np.square(diffs))))
