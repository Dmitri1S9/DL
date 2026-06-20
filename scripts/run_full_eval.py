"""Evaluate the fine-tuned VITS model on held-out droid clips.

Computes WER and CER (intelligibility) and MCD and F0 RMSE (how close to the
droid voice) for the base model and the fine-tuned model, then saves a small
table to docs/eval_results.md.
"""

import io
import json
import ssl
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from transformers import AutoTokenizer

from evaluation.metrics import compute_cer, compute_f0_rmse, compute_mcd, compute_wer
from vits_finetune.model import VitsFinetuneModel
from vits_finetune.model_config import VitsModelConfig

# uni network breaks https cert checks, skip them so the downloads work
ssl._create_default_https_context = ssl._create_unverified_context

N = 20
DATASET = 'Dmi1tr13/ljspeech-b1'
WORK = Path('/tmp/full_eval')
# label -> checkpoint file on HF (None means the pretrained base model)
MODELS = {'base': None, 'finetuned': 'epoch_5_G.pt'}


def get_clips(n):
    # take the last 5% of the dataset (not used in training) as the test set
    ds = load_dataset(DATASET, split='train[95%:]').cast_column('audio', Audio(decode=False))
    (WORK / 'ref').mkdir(parents=True, exist_ok=True)
    clips = []
    for i in range(min(n, len(ds))):
        ex = ds[i]
        text = (ex.get('text') or '').strip()
        if not text:
            continue
        audio, sr = sf.read(io.BytesIO(ex['audio']['bytes']))
        ref = WORK / 'ref' / f'ref_{i}.wav'
        sf.write(str(ref), audio, sr)
        clips.append((text, str(ref)))
    return clips


def load_model(ckpt_file, cfg):
    model = VitsFinetuneModel(cfg)
    if ckpt_file:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download('Dmi1tr13/vits-b1-droid', ckpt_file, local_dir=str(WORK / 'ckpt'))
        state = torch.load(path, map_location='cpu', weights_only=False)['model']
        model.load_state_dict(state, strict=False)
    return model.eval()


def score(label, ckpt_file, clips, tok, cfg):
    model = load_model(ckpt_file, cfg)
    out_dir = WORK / 'gen' / label
    out_dir.mkdir(parents=True, exist_ok=True)
    gens, texts, mcds, f0s = [], [], [], []
    for i, (text, ref) in enumerate(clips):
        ids = tok(text, return_tensors='pt').input_ids
        with torch.no_grad():
            wav = model.vits(input_ids=ids).waveform.squeeze(0).cpu().numpy()
        gen = out_dir / f'gen_{i}.wav'
        sf.write(str(gen), wav, 22050)
        gens.append(str(gen))
        texts.append(text)
        # MCD and F0 can fail on a bad clip, just skip those
        try:
            mcds.append(compute_mcd(ref, str(gen)))
        except Exception:
            pass
        try:
            f0s.append(compute_f0_rmse(ref, str(gen)))
        except Exception:
            pass
    return {
        'label': label,
        'n': len(gens),
        'wer': round(compute_wer(gens, texts) * 100, 2),
        'cer': round(compute_cer(gens, texts) * 100, 2),
        'mcd_db': round(float(np.nanmean(mcds)), 2) if mcds else None,
        'f0_rmse_hz': round(float(np.nanmean(f0s)), 2) if f0s else None,
    }


def main():
    cfg = VitsModelConfig()
    tok = AutoTokenizer.from_pretrained(cfg.pretrained_model_name, cache_dir=str(cfg.cache_dir))

    clips = get_clips(N)
    print(f'scoring {len(clips)} held-out clips...')

    rows = []
    for label, ckpt_file in MODELS.items():
        rows.append(score(label, ckpt_file, clips, tok, cfg))
        print(rows[-1])

    out = Path('docs/eval_results')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix('.json').write_text(json.dumps({'n_clips': len(clips), 'results': rows}, indent=2))

    lines = [
        '# Objective evaluation — VITS B1-droid fine-tune',
        '',
        f'Held-out set: last-5% of `{DATASET}` ({len(clips)} clips, unseen in training). '
        'WER/CER vs prompt text (Whisper); MCD & F0 RMSE vs the B1 droid reference audio. '
        'All metrics: lower = better.',
        '',
        '| model | n | WER % | CER % | MCD (dB) | F0 RMSE (Hz) |',
        '|---|---|---|---|---|---|',
    ]
    for r in rows:
        lines.append(
            f'| {r["label"]} | {r["n"]} | {r["wer"]} | {r["cer"]} | {r["mcd_db"]} | {r["f0_rmse_hz"]} |'
        )
    out.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print('saved docs/eval_results.md')


if __name__ == '__main__':
    main()
