import glob
import os
import platform
from pathlib import Path

# phonemizer (the VITS tokenizer backend) needs to locate the espeak shared
# library, and doesn't always find it on its own.
if platform.system() == 'Windows':
    # winget install espeak-ng.espeak-ng
    _ESPEAK_DLL = Path(r'C:\Program Files\eSpeak NG\libespeak-ng.dll')
    if _ESPEAK_DLL.exists():
        from phonemizer.backend.espeak.espeak import EspeakBackend

        EspeakBackend.set_library(str(_ESPEAK_DLL))
elif platform.system() == 'Linux' and 'PHONEMIZER_ESPEAK_LIBRARY' not in os.environ:
    # On Colab/Linux, phonemizer often can't auto-find libespeak-ng after
    # `apt install espeak-ng` -> point it at the shared library explicitly.
    _candidates = [
        '/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1',
        *glob.glob('/usr/lib/**/libespeak-ng.so*', recursive=True),
        *glob.glob('/usr/local/lib/**/libespeak-ng.so*', recursive=True),
    ]
    for _lib in _candidates:
        if os.path.exists(_lib):
            os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = _lib
            break
