"""RVC B1 Battle Droid batch inference.

Usage:
    python run.py                        # обработать все wav в /input
    python run.py --shard 0 2            # первая половина из двух машин
    python run.py --shard 1 2            # вторая половина
"""

import sys, os, argparse, time, traceback

# Parse OUR args first and strip them from sys.argv
# so RVC's own argparse doesn't choke on --watch / --shard
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--shard", nargs=2, type=int, metavar=("IDX", "TOTAL"))
_parser.add_argument("--watch", action="store_true")
_our_args, _remaining = _parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining  # RVC only sees the rest

sys.path = [p for p in sys.path if p not in ('', '/app')]
sys.path.insert(0, '/rvc')
os.chdir('/rvc')

import torch

# PyTorch 2.6+ changed weights_only default to True — breaks fairseq checkpoints
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

INPUT_DIR  = "/input"
OUTPUT_DIR = "/output"
MODELS_DIR = "/models"
PTH_PATH   = f"{MODELS_DIR}/b1.pth"
INDEX_PATH = f"{MODELS_DIR}/b1.index"
ERROR_LOG  = f"{OUTPUT_DIR}/errors.log"


def setup_model():
    from configs.config import Config
    from infer.modules.vc.modules import VC

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.environ["weight_root"] = MODELS_DIR
    os.environ["index_root"]  = MODELS_DIR
    os.environ["outside_index_root"] = MODELS_DIR

    config = Config()
    vc = VC(config)
    vc.get_vc("b1.pth")

    # Move model to GPU if available
    if device == "cuda:0":
        vc.net_g = vc.net_g.cuda()
        if hasattr(vc, 'hubert_model') and vc.hubert_model is not None:
            vc.hubert_model = vc.hubert_model.cuda()

    params = list(vc.net_g.parameters())
    print(f"Model on: {params[0].device}")
    return vc


def process_file(vc, wav_path: str, out_path: str) -> None:
    index = INDEX_PATH if os.path.exists(INDEX_PATH) else ""
    _, result = vc.vc_single(
        sid=0,
        input_audio_path=wav_path,
        f0_up_key=0,
        f0_file=None,
        f0_method="pm",
        file_index=index,
        file_index2=index,
        index_rate=0.75,
        filter_radius=3,
        resample_sr=0,
        rms_mix_rate=0.25,
        protect=0.33,
    )
    import soundfile as sf
    sf.write(out_path, result[1], result[0])


def main():
    args = _our_args  # already parsed at module level, stripped from sys.argv

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading model once...")
    vc = setup_model()

    idle_rounds = 0
    while True:
        wavs = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".wav")])
        if args.shard:
            idx, total = args.shard
            wavs = [f for i, f in enumerate(wavs) if i % total == idx]

        todo = [f for f in wavs if not os.path.exists(
            os.path.join(OUTPUT_DIR, os.path.splitext(f)[0] + "_b1.wav")
        )]

        if not todo:
            if not args.watch:
                print("All files processed.")
                return
            idle_rounds += 1
            if idle_rounds >= 4:  # 4 × 15s = 60s без новых файлов
                print("No new files for 60s — done.")
                return
            print(f"No new files, waiting 15s... ({idle_rounds}/4)")
            time.sleep(15)
            continue

        idle_rounds = 0
        skipped = len(wavs) - len(todo)
        print(f"Found {len(todo)} new files (skipping {skipped} done)")

        try:
            from tqdm import tqdm
        except ImportError:
            os.system("pip install -q tqdm")
            from tqdm import tqdm

        errors = []
        t_start = time.time()

        with open(ERROR_LOG, "a") as err_file:
            for fname in tqdm(todo, unit="file"):
                wav_path = os.path.join(INPUT_DIR, fname)
                stem = os.path.splitext(fname)[0]
                out_path = os.path.join(OUTPUT_DIR, f"{stem}_b1.wav")
                try:
                    process_file(vc, wav_path, out_path)
                except Exception as e:
                    msg = f"{fname}: {e}\n{traceback.format_exc()}\n"
                    err_file.write(msg)
                    err_file.flush()
                    errors.append(fname)
                    tqdm.write(f"ERROR: {fname} — {e}")

        total_time = time.time() - t_start
        done = len(todo) - len(errors)
        print(f"\nDone: {done}/{len(todo)} in {total_time:.0f}s | avg {total_time/max(done,1):.1f}s/file")
        if errors:
            print(f"Errors: {len(errors)} — see {ERROR_LOG}")


if __name__ == "__main__":
    main()
