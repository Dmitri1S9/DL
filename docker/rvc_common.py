"""Shared helpers for the RVC B1 inference scripts (run.py / infer.py / diag.py).

These all run *inside* the RVC Docker container (RVC source mounted at /rvc), so
the heavy imports (torch, infer.modules.vc) only resolve there. Keep this module
import-light so it can be imported at the very top of each script — before they
prune the container's own paths out of sys.path.
"""

import os

# RVC vc_single inference parameters, shared by every B1 conversion call.
INDEX_RATE = 0.75
FILTER_RADIUS = 3
RESAMPLE_SR = 0
RMS_MIX_RATE = 0.25
PROTECT = 0.33


def patch_torch_load() -> None:
    """Force ``weights_only=False`` for ``torch.load``.

    PyTorch 2.6+ flipped the ``weights_only`` default to True, which breaks the
    fairseq/RVC checkpoints. Call once after torch is importable.
    """
    import torch

    original_load = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return original_load(*args, **kwargs)

    torch.load = _load


def convert_b1(vc, wav_path: str, index_path: str = '', f0_method: str = 'pm'):
    """Run one B1 voice conversion with the shared RVC parameters.

    ``index_path`` is used only if it points at an existing file. Returns the
    ``(sample_rate, waveform)`` pair from ``vc.vc_single``.
    """
    index = index_path if index_path and os.path.exists(index_path) else ''
    _, result = vc.vc_single(
        sid=0,
        input_audio_path=wav_path,
        f0_up_key=0,
        f0_file=None,
        f0_method=f0_method,
        file_index=index,
        file_index2=index,
        index_rate=INDEX_RATE,
        filter_radius=FILTER_RADIUS,
        resample_sr=RESAMPLE_SR,
        rms_mix_rate=RMS_MIX_RATE,
        protect=PROTECT,
    )
    return result
