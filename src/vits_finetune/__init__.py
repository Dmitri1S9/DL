import glob
import os
import platform
from pathlib import Path

# phonemizer (the VITS tokenizer backend) needs to locate the espeak shared
# library, and doesn't always find it on its own — point it there explicitly.
_system = platform.system()
if _system == 'Windows':
    # winget install espeak-ng.espeak-ng
    _dll = Path(r'C:\Program Files\eSpeak NG\libespeak-ng.dll')
    _candidates = [str(_dll)]
elif _system == 'Darwin':
    # brew install espeak-ng  (Apple Silicon vs Intel paths)
    _candidates = [
        '/opt/homebrew/lib/libespeak-ng.dylib',
        '/usr/local/lib/libespeak-ng.dylib',
        *glob.glob('/opt/homebrew/Cellar/espeak-ng/**/libespeak-ng.dylib', recursive=True),
    ]
else:  # Linux (Colab): apt install espeak-ng
    _candidates = [
        '/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1',
        *glob.glob('/usr/lib/**/libespeak-ng.so*', recursive=True),
        *glob.glob('/usr/local/lib/**/libespeak-ng.so*', recursive=True),
    ]

_ESPEAK_LIB = next((lib for lib in _candidates if os.path.exists(lib)), None)
if _ESPEAK_LIB is not None:
    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        EspeakWrapper.set_library(_ESPEAK_LIB)
    except Exception:  # phonemizer missing or API change — fall back to its defaults
        pass
