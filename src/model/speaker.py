"""Load the speaker x-vector that conditions SpeechT5 on a particular voice."""

import zipfile

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from core import config


def load_speaker_embedding() -> torch.Tensor:
    """Load the xvector of a female voice (slt), close to LJSpeech, from the zip."""
    zip_path = hf_hub_download(
        repo_id=config.SPEAKER_XVECTOR_REPO,
        filename='spkrec-xvect.zip',
        repo_type='dataset',
        cache_dir=str(config.MODELS_DIR),
    )
    with zipfile.ZipFile(zip_path) as zf:
        slt_files = [
            n for n in zf.namelist() if 'cmu_us_slt' in n and n.endswith('.npy')
        ]
        with zf.open(slt_files[0]) as f:
            xvector = np.load(f)
    return torch.tensor(xvector).unsqueeze(0)
