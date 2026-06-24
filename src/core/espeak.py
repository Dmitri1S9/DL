"""Locate the espeak-ng shared library and point phonemizer at it.

phonemizer (the backend behind the VITS tokenizer) needs the espeak-ng *library*,
not just the CLI. On Linux the system package (``apt install espeak-ng``) is found
automatically, but on Windows and macOS the library lives in a non-standard place
that phonemizer can't discover on its own, so we locate it and register the path.

This consolidates what used to be three separate per-platform bootstraps
(``vits_finetune.__init__``, ``model.synthesize``, ``evaluation.eval_epochs``).

``setup_espeak`` is idempotent and safe to call on import: if the library is
already configured (``PHONEMIZER_ESPEAK_LIBRARY`` set) or none of the known paths
exist, it does nothing.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

# Known install locations of the espeak-ng shared library, per platform.
_LIBRARY_CANDIDATES: dict[str, tuple[str, ...]] = {
    'Windows': (
        # installed by `winget install espeak-ng.espeak-ng`
        r'C:\Program Files\eSpeak NG\libespeak-ng.dll',
    ),
    'Darwin': (
        '/opt/homebrew/lib/libespeak-ng.dylib',  # macOS arm64 (brew)
        '/usr/local/lib/libespeak-ng.dylib',  # macOS intel (brew)
    ),
    'Linux': (
        '/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1',  # debian/ubuntu
    ),
}


def setup_espeak() -> str | None:
    """Point phonemizer at the espeak-ng library if it isn't already configured.

    Returns the library path that was registered, or ``None`` if nothing was
    needed -- either the environment already pointed at a library, or no known
    library was found (e.g. it is already on the default search path, as on Linux).
    """
    existing = os.environ.get('PHONEMIZER_ESPEAK_LIBRARY')
    if existing:
        return existing

    for candidate in _LIBRARY_CANDIDATES.get(platform.system(), ()):
        if not Path(candidate).exists():
            continue
        # Set both the env var (read by phonemizer at backend construction) and
        # the library directly, so it works regardless of which path is taken.
        os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = candidate
        try:
            from phonemizer.backend.espeak.espeak import EspeakBackend

            EspeakBackend.set_library(candidate)
        except Exception:
            pass
        return candidate

    return None
