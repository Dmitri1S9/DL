"""RVC B1 Battle Droid inference — runs inside Docker container.

Uses RVC-Project source cloned at /rvc.
Input:  /input/*.wav
Output: /output/*.wav  (same filenames, B1 voice applied)
Model:  /models/b1.pth + /models/b1.index
"""

import sys
import os
sys.path.insert(0, "/rvc")

from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

INPUT_DIR  = Path("/input")
OUTPUT_DIR = Path("/output")
MODELS_DIR = Path("/models")
PTH_PATH   = MODELS_DIR / "b1.pth"
INDEX_PATH = MODELS_DIR / "b1.index"

HF_REPO = "Homiebear/B1BattleDroid"


def download_model() -> None:
    import zipfile
    print(f"Downloading B1 model from {HF_REPO}...")
    files = list(list_repo_files(HF_REPO))
    print(f"  Found: {files}")

    zip_files = [f for f in files if f.endswith(".zip")]
    pth_files = [f for f in files if f.endswith(".pth")]
    index_files = [f for f in files if f.endswith(".index")]

    # Try direct .pth first, otherwise extract from zip
    if pth_files:
        src = hf_hub_download(HF_REPO, pth_files[0], local_dir=str(MODELS_DIR))
        os.rename(src, PTH_PATH)
        print(f"  Model -> {PTH_PATH}")
    elif zip_files:
        zip_src = hf_hub_download(HF_REPO, zip_files[0], local_dir=str(MODELS_DIR))
        print(f"  Extracting {zip_files[0]}...")
        with zipfile.ZipFile(zip_src) as zf:
            print(f"  Zip contents: {zf.namelist()}")
            extracted_pth = [n for n in zf.namelist() if n.endswith(".pth")]
            extracted_idx = [n for n in zf.namelist() if n.endswith(".index")]
            if not extracted_pth:
                print("ERROR: no .pth inside zip", file=sys.stderr); sys.exit(1)
            with zf.open(extracted_pth[0]) as f:
                PTH_PATH.write_bytes(f.read())
            print(f"  Model -> {PTH_PATH}")
            if extracted_idx:
                with zf.open(extracted_idx[0]) as f:
                    INDEX_PATH.write_bytes(f.read())
                print(f"  Index -> {INDEX_PATH}")
    else:
        print("ERROR: no .pth or .zip in repo", file=sys.stderr); sys.exit(1)

    if index_files:
        src = hf_hub_download(HF_REPO, index_files[0], local_dir=str(MODELS_DIR))
        os.rename(src, INDEX_PATH)
        print(f"  Index -> {INDEX_PATH}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not PTH_PATH.exists():
        download_model()

    from infer.modules.vc.modules import VC
    from configs.config import Config

    config = Config()
    vc = VC(config)
    vc.get_vc(str(PTH_PATH))

    wavs = sorted(INPUT_DIR.glob("*.wav"))
    if not wavs:
        print("No WAV files in /input/"); return

    print(f"Processing {len(wavs)} files on {config.device}...")
    for wav in wavs:
        out = OUTPUT_DIR / wav.name
        _, wav_opt = vc.vc_single(
            sid=0,
            input_audio_path=str(wav),
            f0_up_key=0,
            f0_file=None,
            f0_method="rmvpe",
            file_index=str(INDEX_PATH) if INDEX_PATH.exists() else "",
            index_rate=0.75,
            filter_radius=3,
            resample_sr=0,
            rms_mix_rate=0.25,
            protect=0.33,
        )
        import soundfile as sf
        sf.write(str(out), wav_opt[1], wav_opt[0])
        print(f"  {wav.name} -> {out.name}")

    print(f"Done. {len(wavs)} files -> /output/")


if __name__ == "__main__":
    main()
