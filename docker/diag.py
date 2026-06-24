"""RVC B1 container diagnostics: device, model-load time, single-file inference.

Runs inside the RVC Docker container (RVC source mounted at /rvc). Reports where
the model ends up (GPU/CPU), how long it takes to load, and the real-time factor
for converting one clip — used to debug why inference is slow in the container.
"""

import glob
import os
import sys
import time

# Imported before the sys.path pruning below removes this script's own dir.
from rvc_common import convert_b1, patch_torch_load

sys.path = [p for p in sys.path if p not in ('', '/app')]
sys.path.insert(0, '/rvc')
os.chdir('/rvc')

import torch  # noqa: E402  (must follow the sys.path setup for the RVC source)

patch_torch_load()

print('=== DIAGNOSTICS ===')
print(f'torch version:        {torch.__version__}')
print(f'cuda available:       {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:                  {torch.cuda.get_device_name(0)}')
    print(
        f'VRAM:                 {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB'
    )
else:
    print('GPU:                  none — running on CPU')

# RVC reads these exact lowercase env var names — do not capitalize.
os.environ['weight_root'] = '/models'  # noqa: SIM112
os.environ['index_root'] = '/models'  # noqa: SIM112
os.environ['outside_index_root'] = '/models'  # noqa: SIM112

from configs.config import Config  # noqa: E402  (RVC module, needs /rvc on sys.path)
from infer.modules.vc.modules import VC  # noqa: E402  (RVC module)

print('\n=== MODEL LOAD ===')
t0 = time.time()
config = Config()
print(f'config.device:        {config.device}')
print(f'config.is_half:       {config.is_half}')
vc = VC(config)
vc.get_vc('b1.pth')
print(f'Load time:            {time.time() - t0:.1f}s')

# Check which device the weights ended up on.
params = list(vc.net_g.parameters())
print(f'Model parameters on:  {params[0].device}')

print('\n=== SINGLE-FILE INFERENCE ===')
wavs = glob.glob('/input/*.wav')
if not wavs:
    print('No wav in /input/')
else:
    wav = wavs[0]
    index_path = '/models/b1.index' if os.path.exists('/models/b1.index') else ''

    t_start = time.time()
    result = convert_b1(vc, wav, index_path)
    elapsed = time.time() - t_start

    dur = len(result[1]) / result[0] if result and result[1] is not None else 0
    print(f'Audio length:         {dur:.2f}s')
    print(f'Inference time:       {elapsed:.2f}s')
    print(f'Real-time factor:     {elapsed / dur:.1f}x' if dur > 0 else 'n/a')

    print('\n=== BOTTLENECK ===')
    if not torch.cuda.is_available():
        print('→ GPU unavailable. Everything runs on CPU.')
        print('  Cause: PyTorch is built for CUDA 12.1, but the driver inside the')
        print('  container reports as 12090 (< 12.1.0 by internal numbering).')
        print('  A torch build with CUDA 12.8 is needed inside the container.')
    else:
        print(
            '→ GPU present. Bottleneck is Hubert feature extraction (CPU-bound if f0=pm).'
        )
