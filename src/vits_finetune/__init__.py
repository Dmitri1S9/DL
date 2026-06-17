import glob
import os
import platform


def _set_espeak_library(lib_path: str, data_dir: str | None = None) -> bool:
    """Point phonemizer at an espeak-ng shared library (+ optional data dir)."""
    if not os.path.exists(lib_path):
        return False
    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        if data_dir is not None:
            # phonemizer 3.3 has no set_data_path; espeak reads this env var.
            os.environ.setdefault('ESPEAK_DATA_PATH', data_dir)
        EspeakWrapper.set_library(lib_path)
        return True
    except Exception:  # phonemizer missing or API change — leave its defaults
        return False


# phonemizer (the VITS tokenizer backend) needs to locate the espeak shared
# library. Prefer the pip-shipped `espeakng-loader` — it bundles a matching-arch
# espeak-ng plus its data, which sidesteps the usual macOS arch mismatch and the
# Colab "espeak not installed" headaches. Fall back to a system install.
_done = False
try:
    import espeakng_loader

    _done = _set_espeak_library(
        espeakng_loader.get_library_path(),
        os.path.dirname(espeakng_loader.get_data_path()),
    )
except Exception:
    _done = False

if not _done:
    _system = platform.system()
    if _system == 'Windows':
        _candidates = [r'C:\Program Files\eSpeak NG\libespeak-ng.dll']
    elif _system == 'Darwin':
        _candidates = [
            '/opt/homebrew/lib/libespeak-ng.dylib',
            '/usr/local/lib/libespeak-ng.dylib',
        ]
    else:  # Linux (Colab): apt install espeak-ng
        _candidates = [
            '/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1',
            *glob.glob('/usr/lib/**/libespeak-ng.so*', recursive=True),
            *glob.glob('/usr/local/lib/**/libespeak-ng.so*', recursive=True),
        ]
    for _lib in _candidates:
        if _set_espeak_library(_lib):
            break
