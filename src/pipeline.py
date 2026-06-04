"""
TTS Pipeline — Project 13
=========================
Text → [Text Layer] → [TTS Model] → [Audio Effects] → WAV

Text Layer:   видишь каждый звук, можешь менять
TTS Model:    SpeechT5 (text→mel) + HiFi-GAN (mel→wav)
Audio Effects: дроид, эмоции, пасхалки
"""

import torch
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
import zipfile
from huggingface_hub import hf_hub_download
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

ROOT        = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16000
MODELS_DIR  = str(ROOT / "models")
AUDIO_DIR   = str(ROOT / "audio")

Path(AUDIO_DIR).mkdir(exist_ok=True)


def load_speaker_embedding(models_dir: str = MODELS_DIR) -> torch.Tensor:
    """Загружает xvector женского голоса (slt) из zip датасета."""
    zip_path = hf_hub_download(
        repo_id="Matthijs/cmu-arctic-xvectors",
        filename="spkrec-xvect.zip",
        repo_type="dataset",
        cache_dir=models_dir,
    )
    with zipfile.ZipFile(zip_path) as zf:
        slt_files = [n for n in zf.namelist() if "cmu_us_slt" in n and n.endswith(".npy")]
        with zf.open(slt_files[0]) as f:
            xvector = np.load(f)
    return torch.tensor(xvector).unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════════════
# СЛОЙ 1 — TEXT LAYER
# Здесь текст превращается в "фонемную" запись, которую ты видишь и меняешь.
# Пока это умные текстовые замены — полноценный phonemizer можно добавить позже.
# ══════════════════════════════════════════════════════════════════════════════

# Слова-триггеры для пасхалок: если в тексте есть такое слово → играем файл
EASTER_EGGS: dict[str, str] = {
    "yamete kudasai":   f"{AUDIO_DIR}/yamete.wav",
    "やめてください":    f"{AUDIO_DIR}/yamete.wav",
    "order 66":         f"{AUDIO_DIR}/order66.wav",
}

# Слова-триггеры для эмоций
EMOTION_WORDS: dict[str, str] = {
    "socialism":    "dying",
    "communism":    "dying",
    "bureaucracy":  "dying",
    "capitalism":   "excited",
    "freedom":      "excited",
    "death":        "scared",
    "horror":       "scared",
    "love":         "happy",
    "amazing":      "happy",
}

# Замены для русского акцента: английский → как произнесёт русскоговорящий
RUSSIAN_ACCENT: list[tuple[str, str]] = [
    ("the ",    "ze "),
    ("The ",    "Ze "),
    (" a ",     " "),           # артикль выбрасываем
    (" an ",    " "),
    ("th",      "z"),           # "this" → "zis"
    ("Th",      "Z"),
    ("w",       "v"),           # "we" → "ve"
    ("W",       "V"),
    ("ing ",    "ink "),        # "going" → "goink"
    ("tion",    "shon"),        # "nation" → "nashon"
    ("ould",    "ud"),          # "would" → "vud"
    ("h ",      " "),           # drop aspiration at end
    ("  ",      " "),           # убираем двойные пробелы
]


def show_phoneme_layer(text: str) -> None:
    """Показывает как текст разбивается — для отладки и понимания."""
    print(f"  Оригинал:   '{text}'")
    after_accent = apply_russian_accent(text)
    print(f"  С акцентом: '{after_accent}'")
    print(f"  Символов:   {len(after_accent.split())}")


def apply_russian_accent(text: str) -> str:
    result = text
    for eng, rus in RUSSIAN_ACCENT:
        result = result.replace(eng, rus)
    return result.strip()


def apply_stutter(text: str, stutter_first_n: int = 1) -> str:
    """Заставляет модель заикаться на первых N словах."""
    words = text.split()
    for i in range(min(stutter_first_n, len(words))):
        w = words[i]
        if len(w) > 1:
            words[i] = f"{w[0]}-{w[0]}-{w}"
    return " ".join(words)


def detect_easter_egg(text: str) -> str | None:
    text_lower = text.lower()
    for trigger, path in EASTER_EGGS.items():
        if trigger in text_lower:
            return path
    return None


def detect_emotion(text: str) -> str:
    text_lower = text.lower()
    for word, emotion in EMOTION_WORDS.items():
        if word in text_lower:
            return emotion
    return "neutral"


# ══════════════════════════════════════════════════════════════════════════════
# СЛОЙ 2 — TTS MODEL
# SpeechT5: text → mel-spectrogram  (аналог FastSpeech2)
# HiFi-GAN: mel-spectrogram → wav   (вокодер)
# ══════════════════════════════════════════════════════════════════════════════

class TTSModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[TTS] device: {self.device}")

        print("[TTS] Loading SpeechT5 (text→mel)...")
        self.processor = SpeechT5Processor.from_pretrained(
            "microsoft/speecht5_tts", cache_dir=MODELS_DIR
        )
        self.model = SpeechT5ForTextToSpeech.from_pretrained(
            "microsoft/speecht5_tts", cache_dir=MODELS_DIR
        ).to(self.device)

        print("[TTS] Loading HiFi-GAN (mel→wav)...")
        self.vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan", cache_dir=MODELS_DIR
        ).to(self.device)

        print("[TTS] Loading speaker embeddings...")
        self.speaker_embeddings = load_speaker_embedding().to(self.device)

        print("[TTS] Ready.\n")

    def synthesize(self, text: str) -> np.ndarray:
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            speech = self.model.generate_speech(
                inputs["input_ids"],
                self.speaker_embeddings,
                vocoder=self.vocoder,
            )
        return speech.cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# СЛОЙ 3 — AUDIO EFFECTS
# Постпроцессинг: программно искажаем аудио
# ══════════════════════════════════════════════════════════════════════════════

def apply_droid(audio: np.ndarray, carrier_hz: float = 80.0) -> np.ndarray:
    """Ring modulation — сигнал умножается на синусоиду → металлический дроид."""
    t = np.arange(len(audio)) / SAMPLE_RATE
    carrier = np.sin(2 * np.pi * carrier_hz * t)
    return (0.35 * audio + 0.65 * (audio * carrier)).astype(np.float32)


def apply_emotion(audio: np.ndarray, emotion: str) -> np.ndarray:
    if emotion == "dying":
        # Pitch вниз + замедление → умирает от скуки
        audio = librosa.effects.pitch_shift(audio.astype(float), sr=SAMPLE_RATE, n_steps=-5)
        audio = librosa.effects.time_stretch(audio, rate=0.55)

    elif emotion == "excited":
        # Pitch вверх + ускорение
        audio = librosa.effects.pitch_shift(audio.astype(float), sr=SAMPLE_RATE, n_steps=3)
        audio = librosa.effects.time_stretch(audio, rate=1.25)

    elif emotion == "scared":
        # Тремоло (дрожание амплитуды)
        t = np.arange(len(audio)) / SAMPLE_RATE
        tremolo = 1.0 + 0.35 * np.sin(2 * np.pi * 9 * t)
        audio = audio * tremolo

    elif emotion == "happy":
        # Лёгкий pitch up
        audio = librosa.effects.pitch_shift(audio.astype(float), sr=SAMPLE_RATE, n_steps=2)

    return audio.astype(np.float32)


def normalize(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))
    return audio / peak if peak > 0 else audio


# ══════════════════════════════════════════════════════════════════════════════
# ГЛАВНАЯ ФУНКЦИЯ
# ══════════════════════════════════════════════════════════════════════════════

def speak(
    text:           str,
    model:          TTSModel,
    russian_accent: bool = True,
    droid:          bool = True,
    stutter:        bool = False,
    output_path:    str  = f"{AUDIO_DIR}/output.wav",
) -> np.ndarray:

    print(f"\n{'─'*60}")
    print(f"Input: '{text}'")

    # Шаг 1: Пасхалки — если есть триггер, играем готовый файл
    easter_egg_path = detect_easter_egg(text)
    if easter_egg_path and Path(easter_egg_path).exists():
        print(f"  ► Easter egg: {easter_egg_path}")
        audio, sr = sf.read(easter_egg_path)
        sf.write(output_path, audio, sr)
        return np.array(audio)

    # Шаг 2: Определяем эмоцию
    emotion = detect_emotion(text)
    if emotion != "neutral":
        print(f"  ► Emotion detected: {emotion}")

    # Шаг 3: Text Layer
    processed = text
    if stutter:
        processed = apply_stutter(processed)
    if russian_accent:
        processed = apply_russian_accent(processed)
    show_phoneme_layer(text)

    # Шаг 4: TTS синтез
    audio = model.synthesize(processed)

    # Шаг 5: Аудио эффекты
    if emotion != "neutral":
        audio = apply_emotion(audio, emotion)
    if droid:
        audio = apply_droid(audio)

    audio = normalize(audio)

    sf.write(output_path, audio, SAMPLE_RATE)
    print(f"  ► Saved: {output_path}")
    return audio


# ══════════════════════════════════════════════════════════════════════════════
# ТОЧКА ВХОДА — демо тесты
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    model = TTSModel()

    demos = [
        ("Hello, I am a robot from outer space.",          "demo_1_normal.wav"),
        ("The weather is quite wonderful today.",           "demo_2_accent.wav"),
        ("I fully support socialism and its principles.",   "demo_3_dying.wav"),
        ("This is amazing! I love freedom and capitalism.", "demo_4_excited.wav"),
        ("Hello world.",                                    "demo_5_stutter.wav"),
    ]

    for i, (text, fname) in enumerate(demos):
        stutter = (i == 4)
        speak(
            text,
            model,
            russian_accent=True,
            droid=True,
            stutter=stutter,
            output_path=f"{AUDIO_DIR}/{fname}",
        )

    print("\nAll demos generated in /audio/")
