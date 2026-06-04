import sys, os, time
sys.path = [p for p in sys.path if p not in ('', '/app')]
sys.path.insert(0, '/rvc')
os.chdir('/rvc')

import torch

# Patch torch.load
_orig = torch.load
def _patched(*a, **kw): kw.setdefault('weights_only', False); return _orig(*a, **kw)
torch.load = _patched

print("=== ДИАГНОСТИКА ===")
print(f"torch version:        {torch.__version__}")
print(f"cuda available:       {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:                  {torch.cuda.get_device_name(0)}")
    print(f"VRAM:                 {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")
else:
    print("GPU:                  НЕТ — работает на CPU")

os.environ["weight_root"] = "/models"
os.environ["index_root"]  = "/models"
os.environ["outside_index_root"] = "/models"

from configs.config import Config
from infer.modules.vc.modules import VC

print("\n=== ЗАГРУЗКА МОДЕЛИ ===")
t0 = time.time()
config = Config()
print(f"config.device:        {config.device}")
print(f"config.is_half:       {config.is_half}")
vc = VC(config)
vc.get_vc("b1.pth")
print(f"Время загрузки:       {time.time()-t0:.1f}s")

# Проверить на каком устройстве сидят веса
net = vc.net_g
params = list(net.parameters())
print(f"Параметры модели на:  {params[0].device}")

print("\n=== ИНФЕРЕНС ОДНОГО ФАЙЛА ===")
import glob
wavs = glob.glob("/input/*.wav")
if not wavs:
    print("Нет wav в /input/")
else:
    wav = wavs[0]
    t1 = time.time()
    index_path = "/models/b1.index" if os.path.exists("/models/b1.index") else ""
    toc = {}

    import time as T
    T0 = T.time()
    _, result = vc.vc_single(0, wav, 0, None, "pm", index_path, index_path,
                              0.75, 3, 0, 0.25, 0.33)
    total = T.time() - T0

    import soundfile as sf
    dur = len(result[1]) / result[0] if result and result[1] is not None else 0
    print(f"Аудио длина:          {dur:.2f}s")
    print(f"Время инференса:      {total:.2f}s")
    print(f"Реалтайм-фактор:      {total/dur:.1f}x" if dur > 0 else "n/a")

    # Разбивка по шагам (грубо через логи)
    print("\n=== УЗКОЕ МЕСТО ===")
    if not torch.cuda.is_available():
        print("→ GPU недоступен. Весь расчёт идёт на CPU.")
        print("  Причина: PyTorch собран под CUDA 12.1, но драйвер внутри контейнера")
        print("  отображается как версия 12090 (< 12.1.0 по внутренней нумерации).")
        print("  Нужен torch с CUDA 12.8 внутри контейнера.")
    else:
        print("→ GPU есть. Узкое место — Hubert feature extraction (CPU bound если f0=pm)")
