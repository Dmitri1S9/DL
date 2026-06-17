import glob
import os
import platform
from pathlib import Path

# phonemizer (the VITS tokenizer backend) needs to locate the espeak shared
# library, and doesn't always find it on its own — point it there explicitly.
_ESPEAK_LIB = None
if platform.system() == 'Windows':
    # winget install espeak-ng.espeak-ng
    _dll = Path(r'C:\Program Files\eSpeak NG\libespeak-ng.dll')
    _ESPEAK_LIB = str(_dll) if _dll.exists() else None
elif platform.system() == 'Linux':
    # On Colab/Linux, phonemizer often can't auto-find libespeak-ng after
    # `apt install espeak-ng`; setting the library explicitly is what works.
    for _lib in [
        '/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1',
        *glob.glob('/usr/lib/**/libespeak-ng.so*', recursive=True),
        *glob.glob('/usr/local/lib/**/libespeak-ng.so*', recursive=True),
    ]:
        if os.path.exists(_lib):
            _ESPEAK_LIB = _lib
            break

if _ESPEAK_LIB is not None:
    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        EspeakWrapper.set_library(_ESPEAK_LIB)
    except Exception:  # phonemizer missing or API change — fall back to its defaults
        pass
