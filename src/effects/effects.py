"""Fun audio effects + a demo `speak()` — NOT part of the evaluation path.

These effects (droid voice, Russian accent, emotions, stutter, easter eggs)
deliberately distort the audio, which would wreck WER/MCD. They live here as an
optional demo only; the clean synthesis used for evaluation is in model.synthesize.
"""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from core import config
from core.logger import logger
from model.synthesize import TTSModel

SAMPLE_RATE = config.SAMPLE_RATE
AUDIO_DIR = str(config.AUDIO_DIR)

# ── Text layer ──────────────────────────────────────────────────────────────────

# Trigger words for easter eggs: if the text contains one -> play the file
EASTER_EGGS: dict[str, str] = {
    'yamete kudasai': f'{AUDIO_DIR}/yamete.wav',
    'やめてください': f'{AUDIO_DIR}/yamete.wav',
    'order 66': f'{AUDIO_DIR}/order66.wav',
}

# Trigger words for emotions
EMOTION_WORDS: dict[str, str] = {
    'socialism': 'dying',
    'communism': 'dying',
    'bureaucracy': 'dying',
    'capitalism': 'excited',
    'freedom': 'excited',
    'death': 'scared',
    'horror': 'scared',
    'love': 'happy',
    'amazing': 'happy',
}

# Substitutions for a Russian accent: English -> how a Russian speaker would say it
RUSSIAN_ACCENT: list[tuple[str, str]] = [
    ('the ', 'ze '),
    ('The ', 'Ze '),
    (' a ', ' '),  # drop the article
    (' an ', ' '),
    ('th', 'z'),  # "this" -> "zis"
    ('Th', 'Z'),
    ('w', 'v'),  # "we" -> "ve"
    ('W', 'V'),
    ('ing ', 'ink '),  # "going" -> "goink"
    ('tion', 'shon'),  # "nation" -> "nashon"
    ('ould', 'ud'),  # "would" -> "vud"
    ('h ', ' '),  # drop aspiration at end
    ('  ', ' '),  # remove double spaces
]


def apply_russian_accent(text: str) -> str:
    result = text
    for eng, rus in RUSSIAN_ACCENT:
        result = result.replace(eng, rus)
    return result.strip()


def apply_stutter(text: str, stutter_first_n: int = 1) -> str:
    """Make the model stutter on the first N words."""
    words = text.split()
    for i in range(min(stutter_first_n, len(words))):
        w = words[i]
        if len(w) > 1:
            words[i] = f'{w[0]}-{w[0]}-{w}'
    return ' '.join(words)


def show_phoneme_layer(text: str) -> None:
    """Show how the text is broken down — for debugging and understanding."""
    logger.debug(f"Original: '{text}'")
    after_accent = apply_russian_accent(text)
    logger.debug(f"Accented: '{after_accent}'")
    logger.debug(f'Words:    {len(after_accent.split())}')


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
    return 'neutral'


# ── Audio effects ─────────────────────────────────────────────────────────────


def apply_droid(audio: np.ndarray, carrier_hz: float = 80.0) -> np.ndarray:
    """Ring modulation — the signal is multiplied by a sine wave -> metallic droid."""
    t = np.arange(len(audio)) / SAMPLE_RATE
    carrier = np.sin(2 * np.pi * carrier_hz * t)
    return (0.35 * audio + 0.65 * (audio * carrier)).astype(np.float32)


def apply_emotion(audio: np.ndarray, emotion: str) -> np.ndarray:
    if emotion == 'dying':
        # Pitch down + slow down -> dying of boredom
        audio = librosa.effects.pitch_shift(
            audio.astype(float), sr=SAMPLE_RATE, n_steps=-5
        )
        audio = librosa.effects.time_stretch(audio, rate=0.55)

    elif emotion == 'excited':
        # Pitch up + speed up
        audio = librosa.effects.pitch_shift(
            audio.astype(float), sr=SAMPLE_RATE, n_steps=3
        )
        audio = librosa.effects.time_stretch(audio, rate=1.25)

    elif emotion == 'scared':
        # Tremolo (amplitude wobble)
        t = np.arange(len(audio)) / SAMPLE_RATE
        tremolo = 1.0 + 0.35 * np.sin(2 * np.pi * 9 * t)
        audio = audio * tremolo

    elif emotion == 'happy':
        # Slight pitch up
        audio = librosa.effects.pitch_shift(
            audio.astype(float), sr=SAMPLE_RATE, n_steps=2
        )

    return audio.astype(np.float32)


def normalize(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))
    return audio / peak if peak > 0 else audio


def speak(
    text: str,
    model: TTSModel,
    russian_accent: bool = True,
    droid: bool = True,
    stutter: bool = False,
    output_path: str = f'{AUDIO_DIR}/output.wav',
) -> np.ndarray:
    """Synthesize ``text`` and apply the fun effects (demo only)."""
    logger.info(f"Input: '{text}'")

    # Step 1: Easter eggs — if a trigger is present, play the ready-made file
    easter_egg_path = detect_easter_egg(text)
    if easter_egg_path and Path(easter_egg_path).exists():
        logger.info(f'Easter egg: {easter_egg_path}')
        audio, sr = sf.read(easter_egg_path)
        sf.write(output_path, audio, sr)
        return np.array(audio)

    # Step 2: Detect emotion
    emotion = detect_emotion(text)
    if emotion != 'neutral':
        logger.info(f'Emotion detected: {emotion}')

    # Step 3: Text layer
    processed = text
    if stutter:
        processed = apply_stutter(processed)
    if russian_accent:
        processed = apply_russian_accent(processed)
    show_phoneme_layer(text)

    # Step 4: TTS synthesis
    audio = model.synthesize(processed)

    # Step 5: Audio effects
    if emotion != 'neutral':
        audio = apply_emotion(audio, emotion)
    if droid:
        audio = apply_droid(audio)

    audio = normalize(audio)
    sf.write(output_path, audio, SAMPLE_RATE)
    logger.success(f'Saved: {output_path}')
    return audio


def main() -> None:
    """Generate the demo gallery with the fun effects applied."""
    model = TTSModel()
    demos = [
        ('Hello, I am a robot from outer space.', 'demo_1_normal.wav'),
        ('The weather is quite wonderful today.', 'demo_2_accent.wav'),
        ('I fully support socialism and its principles.', 'demo_3_dying.wav'),
        ('This is amazing! I love freedom and capitalism.', 'demo_4_excited.wav'),
        ('Hello world.', 'demo_5_stutter.wav'),
    ]
    for i, (text, fname) in enumerate(demos):
        speak(
            text,
            model,
            russian_accent=True,
            droid=True,
            stutter=i == 4,
            output_path=f'{AUDIO_DIR}/{fname}',
        )
    logger.success('All demos generated in /audio/')


if __name__ == '__main__':
    main()
