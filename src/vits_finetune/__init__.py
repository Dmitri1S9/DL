import platform
from pathlib import Path

# On Windows, phonemizer can't find espeak-ng on its own; point it at the DLL
# installed by `winget install espeak-ng.espeak-ng`. On Linux (Colab), the
# system espeak-ng (apt install espeak-ng) is found automatically.
if platform.system() == 'Windows':
    _ESPEAK_DLL = Path(r'C:\Program Files\eSpeak NG\libespeak-ng.dll')
    if _ESPEAK_DLL.exists():
        from phonemizer.backend.espeak.espeak import EspeakBackend

        EspeakBackend.set_library(str(_ESPEAK_DLL))
