"""
Evaluation: WER + MCD
Run after pipeline.py has generated audio for the test set.

WER  — Word Error Rate: how well an ASR model (Whisper) understands our TTS.
MCD  — Mel Cepstral Distortion: how close the mel-spectrograms are to the original.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path


# ─── WER via Whisper ──────────────────────────────────────────────────────────


def compute_wer(audio_paths: list[str], reference_texts: list[str]) -> float:
    """
    Run Whisper over the generated audio and compute WER.
    Requires: pip install openai-whisper jiwer
    """
    import whisper
    from jiwer import wer

    model = whisper.load_model("base")
    hypotheses = []

    for path in audio_paths:
        result = model.transcribe(path, language="en")
        hypotheses.append(result["text"].strip().lower())

    references = [t.lower() for t in reference_texts]
    score = wer(references, hypotheses)

    print(f"WER: {score:.4f} ({score * 100:.1f}%)")
    return score


# ─── MCD ─────────────────────────────────────────────────────────────────────


def compute_mcd(ref_audio: np.ndarray, gen_audio: np.ndarray, sr: int = 16000) -> float:
    """
    Mel Cepstral Distortion between the original and generated audio.
    Lower = better. A value of ~5-8 is considered acceptable.
    """

    def extract_mfcc(audio):
        return librosa.feature.mfcc(y=audio.astype(float), sr=sr, n_mfcc=24)

    ref_mfcc = extract_mfcc(ref_audio)
    gen_mfcc = extract_mfcc(gen_audio)

    # Align lengths
    min_len = min(ref_mfcc.shape[1], gen_mfcc.shape[1])
    ref_mfcc = ref_mfcc[:, :min_len]
    gen_mfcc = gen_mfcc[:, :min_len]

    # MCD formula (excluding the zeroth coefficient)
    diff = ref_mfcc[1:] - gen_mfcc[1:]
    mcd = (10 / np.log(10)) * np.sqrt(2) * np.mean(np.sqrt(np.sum(diff**2, axis=0)))

    print(f"MCD: {mcd:.4f} dB")
    return mcd


def evaluate_test_set(
    generated_dir: str,
    reference_dir: str,
    texts_file: str,
) -> dict:
    """Compute WER and the average MCD over the whole test set."""
    gen_files = sorted(Path(generated_dir).glob("*.wav"))
    ref_files = sorted(Path(reference_dir).glob("*.wav"))
    texts = Path(texts_file).read_text().splitlines()

    mcd_scores = []
    for gen_path, ref_path in zip(gen_files, ref_files, strict=False):
        gen_audio, _ = sf.read(str(gen_path))
        ref_audio, _ = sf.read(str(ref_path))
        mcd_scores.append(compute_mcd(np.array(ref_audio), np.array(gen_audio)))

    wer_score = compute_wer([str(p) for p in gen_files], texts)
    avg_mcd = float(np.mean(mcd_scores))

    results = {"WER": wer_score, "MCD": avg_mcd}
    print(f"\n{'─' * 40}")
    print(f"  WER (lower=better): {wer_score:.4f}")
    print(f"  MCD (lower=better): {avg_mcd:.4f} dB")
    return results


if __name__ == "__main__":
    # Example usage
    ROOT = Path(__file__).resolve().parents[2]
    ref, _ = sf.read(str(ROOT / "data/sample_ref.wav"))
    gen, _ = sf.read(str(ROOT / "audio/demo_1_normal.wav"))
    compute_mcd(np.array(ref), np.array(gen))
