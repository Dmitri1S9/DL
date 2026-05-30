"""
Evaluation: WER + MCD
Запускать после того как pipeline.py сгенерировал аудио на тест-сете.

WER  — Word Error Rate: насколько ASR (Whisper) понимает наш TTS.
MCD  — Mel Cepstral Distortion: насколько мел-спектрограммы похожи на оригинал.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path


# ─── WER через Whisper ────────────────────────────────────────────────────────

def compute_wer(audio_paths: list[str], reference_texts: list[str]) -> float:
    """
    Прогоняет Whisper через сгенерированные аудио и считает WER.
    Нужен: pip install openai-whisper jiwer
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

    print(f"WER: {score:.4f} ({score*100:.1f}%)")
    return score


# ─── MCD ─────────────────────────────────────────────────────────────────────

def compute_mcd(ref_audio: np.ndarray, gen_audio: np.ndarray, sr: int = 16000) -> float:
    """
    Mel Cepstral Distortion между оригинальным и сгенерированным аудио.
    Ниже = лучше. Значение ~5-8 считается приемлемым.
    """
    def extract_mfcc(audio):
        return librosa.feature.mfcc(y=audio.astype(float), sr=sr, n_mfcc=24)

    ref_mfcc = extract_mfcc(ref_audio)
    gen_mfcc = extract_mfcc(gen_audio)

    # Выравниваем длину
    min_len = min(ref_mfcc.shape[1], gen_mfcc.shape[1])
    ref_mfcc = ref_mfcc[:, :min_len]
    gen_mfcc = gen_mfcc[:, :min_len]

    # MCD формула (без нулевого коэффициента)
    diff = ref_mfcc[1:] - gen_mfcc[1:]
    mcd = (10 / np.log(10)) * np.sqrt(2) * np.mean(np.sqrt(np.sum(diff**2, axis=0)))

    print(f"MCD: {mcd:.4f} dB")
    return mcd


def evaluate_test_set(
    generated_dir: str,
    reference_dir: str,
    texts_file:    str,
) -> dict:
    """Считает WER и средний MCD по всему тест-сету."""
    gen_files = sorted(Path(generated_dir).glob("*.wav"))
    ref_files = sorted(Path(reference_dir).glob("*.wav"))
    texts = Path(texts_file).read_text().splitlines()

    mcd_scores = []
    for gen_path, ref_path in zip(gen_files, ref_files):
        gen_audio, _ = sf.read(str(gen_path))
        ref_audio, _ = sf.read(str(ref_path))
        mcd_scores.append(compute_mcd(
            np.array(ref_audio), np.array(gen_audio)
        ))

    wer_score = compute_wer([str(p) for p in gen_files], texts)
    avg_mcd   = float(np.mean(mcd_scores))

    results = {"WER": wer_score, "MCD": avg_mcd}
    print(f"\n{'─'*40}")
    print(f"  WER (lower=better): {wer_score:.4f}")
    print(f"  MCD (lower=better): {avg_mcd:.4f} dB")
    return results


if __name__ == "__main__":
    # Пример использования
    ref,  _ = sf.read("data/sample_ref.wav")
    gen,  _ = sf.read("audio/demo_1_normal.wav")
    compute_mcd(np.array(ref), np.array(gen))
