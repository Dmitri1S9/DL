"""Fun audio effects + a demo `speak()` — NOT part of the evaluation path.

These effects (droid voice, Russian accent, emotions, stutter, easter eggs)
deliberately distort the audio, which would wreck WER/MCD. They live here as an
optional demo only; the clean synthesis used for evaluation is in model.synthesize.
"""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from g2p_en import G2p

from core import config
from core.logger import logger
from model.synthesize import TTSModel

_g2p = G2p()

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

# ARPAbet phoneme-level substitutions for Russian accent simulation.
# None = drop phoneme entirely.
# Context-dependent rules (final devoicing, NG->NGK) live in phoneme.apply_accent.
ACCENT_MAP_PHONEME: dict[str, str | None] = {
    # ── Consonants ────────────────────────────────────────────────────────────
    'DH': 'Z',  # voiced th  "the"    -> "ze"
    'TH': 'Z',  # voiceless  "think"  -> "zink"
    'W': 'V',  # "we"       -> "ve",  "would" -> "vud"
    'HH': None,  # drop aspiration:     "have"  -> "av"
    # ── Vowels — Russian flattens English diphthongs ──────────────────────────
    'AE': 'EH',  # /æ/ "cat"  -> /ɛ/ "cet"   (no near-open front vowel in Russian)
    'OW': 'AO',  # /oʊ/ "go"  -> /o/ pure     (no glide, Russian O is pure)
    'EY': 'EH',  # /eɪ/ "say" -> /e/ pure     (no glide)
    'AW': 'AO',  # /aʊ/ "how" -> /o/          (collapse to O)
    'OY': 'AO',  # /ɔɪ/ "boy" -> /o/          (collapse)
    # ── Unstressed vowel reduction ────────────────────────────────────────────
    # Russian strongly reduces unstressed vowels -> AH (schwa) everywhere
    # (applied only to the already-unstressed AX; full AH we keep)
}

# Final devoicing: voiced consonant at word-end -> voiceless equivalent
# "bad"->"bat", "bag"->"bak", "love"->"lof", "his"->"hiss"
_DEVOICE: dict[str, str] = {'D': 'T', 'B': 'P', 'G': 'K', 'Z': 'S', 'V': 'F'}


def text_to_phonemes(text: str) -> list[str]:
    """Convert English text to ARPAbet tokens (stress digits stripped)."""
    raw = _g2p(text)
    return [p.rstrip('012') if p != ' ' else ' ' for p in raw]


def apply_accent_phonemes(phonemes: list[str]) -> list[str]:
    """Apply ACCENT_MAP_PHONEME + NG->NGK + final devoicing to a phoneme list."""
    result: list[str] = []
    i = 0
    while i < len(phonemes):
        p = phonemes[i]
        next_p = phonemes[i + 1] if i + 1 < len(phonemes) else ' '
        at_word_end = next_p == ' ' or i + 1 == len(phonemes)

        if at_word_end and p in _DEVOICE:
            result.append(_DEVOICE[p])
            i += 1
            continue

        if p in ACCENT_MAP_PHONEME:
            replacement = ACCENT_MAP_PHONEME[p]
            if replacement is not None:
                result.append(replacement)
            i += 1
            continue

        result.append(p)

        if p == 'NG' and at_word_end:
            result.append('K')

        i += 1
    return result


def phonemes_to_str(phonemes: list[str]) -> str:
    return ' '.join(p for p in phonemes if p != ' ').strip()


def process_text_phonemic(text: str) -> str:
    """Full phoneme pipeline: text -> ARPAbet -> accent -> phoneme string."""
    return phonemes_to_str(apply_accent_phonemes(text_to_phonemes(text)))


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


def apply_b1_droid(audio: np.ndarray) -> np.ndarray:
    """B1 battle droid — DSP chain based on actual Dalek/B1 ring mod technique.

    Key insight: carrier is 35-45 Hz (sub-bass buzz), NOT hundreds of Hz.
    That low carrier creates the characteristic B1 drone while keeping speech
    intelligible. Same principle as Daleks (30 Hz carrier).

    Chain:
      1. Ring mod 40 Hz          — main droid buzz
      2. Pitch shift +3 semitones — B1 is higher/more nasal than human voice
      3. Bandpass 250-6000 Hz    — telephone-thin timbre
      4. Presence boost 2.5 kHz  — nasal "tin" character
      5. Soft clip (tanh)        — digital edge
      6. Comb filter 5 ms        — metallic box resonance
    """
    from scipy.signal import butter, sosfilt

    x = audio.astype(np.float64)

    # 1. Ring modulation at 40 Hz
    t = np.arange(len(x)) / SAMPLE_RATE
    carrier = np.sin(2 * np.pi * 40.0 * t)
    y = x * carrier

    # 2. Pitch shift +3 semitones
    y = librosa.effects.pitch_shift(
        y.astype(np.float32), sr=SAMPLE_RATE, n_steps=3
    ).astype(np.float64)

    # 3. Bandpass 250-6000 Hz
    sos_band = butter(4, [250, 6000], btype='band', fs=SAMPLE_RATE, output='sos')
    y = sosfilt(sos_band, y)

    # 4. Presence boost at 2.5 kHz (peaking EQ via narrow bandpass blend)
    sos_presence = butter(2, [2000, 3500], btype='band', fs=SAMPLE_RATE, output='sos')
    y = y + 0.4 * sosfilt(sos_presence, y)

    # 5. Soft clipping
    y = np.tanh(y * 2.5)

    # 6. Comb filter — short delay (5 ms) + feedback for metallic resonance
    delay_samples = int(0.005 * SAMPLE_RATE)
    comb = y.copy()
    for i in range(delay_samples, len(comb)):
        comb[i] += 0.4 * comb[i - delay_samples]
    y = comb

    y /= np.max(np.abs(y)) + 1e-9
    return y.astype(np.float32)


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
