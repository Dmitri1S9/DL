"""Минимальный тест — проверяем что модель вообще генерирует звук."""
import torch
import numpy as np
import zipfile
import soundfile as sf
from huggingface_hub import hf_hub_download
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

MODELS_DIR = "models"


def load_speaker_embedding(models_dir: str) -> torch.Tensor:
    """Загружает xvector женского голоса (slt) из zip датасета."""
    zip_path = hf_hub_download(
        repo_id="Matthijs/cmu-arctic-xvectors",
        filename="spkrec-xvect.zip",
        repo_type="dataset",
        cache_dir=models_dir,
    )
    with zipfile.ZipFile(zip_path) as zf:
        # Берём первый файл голоса slt (женский, похож на LJSpeech)
        slt_files = [n for n in zf.namelist() if "cmu_us_slt" in n and n.endswith(".npy")]
        with zf.open(slt_files[0]) as f:
            xvector = np.load(f)
    print(f"Speaker embedding shape: {xvector.shape}  (from {slt_files[0]})")
    return torch.tensor(xvector).unsqueeze(0)


print("Loading processor...")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts", cache_dir=MODELS_DIR)

print("Loading model...")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts", cache_dir=MODELS_DIR)

print("Loading vocoder...")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan", cache_dir=MODELS_DIR)

print("Loading speaker embeddings...")
speaker_embeddings = load_speaker_embedding(MODELS_DIR)

text = "Hello world."
print(f"\nSynthesizing: '{text}'")
inputs = processor(text=text, return_tensors="pt")
print(f"Input ids shape: {inputs['input_ids'].shape}")

with torch.no_grad():
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

audio = speech.numpy()
print(f"\nOutput shape: {audio.shape}")
print(f"Max amplitude: {np.max(np.abs(audio)):.6f}")
print(f"Mean amplitude: {np.mean(np.abs(audio)):.6f}")

if np.max(np.abs(audio)) < 1e-6:
    print("WARNING: Audio is silent!")
else:
    sf.write("audio/test_raw.wav", audio, 16000)
    print("Saved to audio/test_raw.wav")
