"""Minimal test — check that the model produces sound at all."""

import sys
import torch
import numpy as np
import zipfile
import soundfile as sf
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = str(ROOT / 'models')

# This script lives in tests/, so put src/ on the path to import the shared logger.
sys.path.insert(0, str(ROOT / 'src'))
from core.logger import logger  # noqa: E402  (import follows the sys.path bootstrap)


def load_speaker_embedding(models_dir: str) -> torch.Tensor:
    """Load the xvector of a female voice (slt) from the dataset zip."""
    zip_path = hf_hub_download(
        repo_id='Matthijs/cmu-arctic-xvectors',
        filename='spkrec-xvect.zip',
        repo_type='dataset',
        cache_dir=models_dir,
    )
    with zipfile.ZipFile(zip_path) as zf:
        # Take the first slt voice file (female, similar to LJSpeech)
        slt_files = [
            n for n in zf.namelist() if 'cmu_us_slt' in n and n.endswith('.npy')
        ]
        with zf.open(slt_files[0]) as f:
            xvector = np.load(f)
    logger.debug(f'Speaker embedding shape: {xvector.shape}  (from {slt_files[0]})')
    return torch.tensor(xvector).unsqueeze(0)


logger.info('Loading processor...')
processor = SpeechT5Processor.from_pretrained(
    'microsoft/speecht5_tts', cache_dir=MODELS_DIR
)

logger.info('Loading model...')
model = SpeechT5ForTextToSpeech.from_pretrained(
    'microsoft/speecht5_tts', cache_dir=MODELS_DIR
)

logger.info('Loading vocoder...')
vocoder = SpeechT5HifiGan.from_pretrained(
    'microsoft/speecht5_hifigan', cache_dir=MODELS_DIR
)

logger.info('Loading speaker embeddings...')
speaker_embeddings = load_speaker_embedding(MODELS_DIR)

text = 'Hello world.'
logger.info(f"Synthesizing: '{text}'")
inputs = processor(text=text, return_tensors='pt')
logger.debug(f'Input ids shape: {inputs["input_ids"].shape}')

with torch.no_grad():
    speech = model.generate_speech(
        inputs['input_ids'], speaker_embeddings, vocoder=vocoder
    )

audio = speech.numpy()
logger.debug(f'Output shape: {audio.shape}')
logger.debug(f'Max amplitude: {np.max(np.abs(audio)):.6f}')
logger.debug(f'Mean amplitude: {np.mean(np.abs(audio)):.6f}')

if np.max(np.abs(audio)) < 1e-6:
    logger.warning('Audio is silent!')
else:
    (ROOT / 'audio').mkdir(exist_ok=True)
    sf.write(str(ROOT / 'audio/test_raw.wav'), audio, 16000)
    logger.success('Saved to audio/test_raw.wav')
