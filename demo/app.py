"""Local web demo: type text -> hear the B1 droid voice.

Run from the project root:

    pip install flask                       # one-time (or: pip install -r requirements-dev.txt)
    PYTHONPATH=src python demo/app.py
    # then open http://127.0.0.1:5000

Loads the fine-tuned checkpoint if present (``models/vits_finetune/epoch_2.pt``),
otherwise falls back to the pretrained base model so the demo still runs. Override
the checkpoint with the VITS_DEMO_CKPT environment variable.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import soundfile as sf
import torch
from flask import Flask, jsonify, request, send_file, send_from_directory
from transformers import AutoTokenizer

from vits_finetune.model_config import VitsModelConfig
from vits_finetune.synthesize import load_model, synthesize

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / 'models' / 'vits_finetune' / 'epoch_2.pt'

app = Flask(__name__, static_folder=str(Path(__file__).parent))

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_model_config = VitsModelConfig()
_tokenizer = AutoTokenizer.from_pretrained(
    _model_config.pretrained_model_name, cache_dir=str(_model_config.cache_dir)
)

_ckpt_env = os.environ.get('VITS_DEMO_CKPT')
_ckpt = Path(_ckpt_env) if _ckpt_env else DEFAULT_CKPT
if _ckpt.exists():
    _model = load_model(_ckpt, _model_config, _device)
    _label = _ckpt.name
else:
    _model = load_model(None, _model_config, _device)
    _label = 'pretrained base (no fine-tuned checkpoint found)'

print(f'Demo ready on {_device} — voice: {_label}')


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/info')
def info():
    return jsonify(checkpoint=_label, device=str(_device))


@app.route('/synthesize', methods=['POST'])
def synth():
    text = (request.get_json(silent=True) or {}).get('text', '').strip()
    if not text:
        return jsonify(error='empty text'), 400
    waveform = synthesize(text, _model, _tokenizer, _device)
    buffer = io.BytesIO()
    sf.write(buffer, waveform.numpy(), _model_config.sampling_rate, format='WAV')
    buffer.seek(0)
    return send_file(buffer, mimetype='audio/wav', download_name='droid.wav')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
