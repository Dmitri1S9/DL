"""Per-epoch eval: WER / CER / MCD after every fine-tuning epoch.

Dmitrii asked for "metrics after each epoch". The epoch checkpoints are public on
the Hub (``Dmi1tr13/vits-b1-droid`` as ``epoch_<N>_G.pt``), so this downloads each
one, synthesizes a fixed set of held-out B1 texts with it, and scores:

  - WER / CER  — Whisper transcribes the *generated* audio vs the text
                 (lower = more intelligible to ASR).
  - MCD        — pymcd DTW mel-cepstral distance between the generated clip and the
                 *real* B1 recording of the same text (lower = closer to the B1
                 timbre — this is the actual goal of the fine-tune).

Read alongside the dataset ceiling (``evaluation.evaluate_dataset``): the ceiling is
how well Whisper reads the *real* B1 audio; these epochs are how well it reads what
the *model* produces. The gap between them is the model's own intelligibility loss,
not an ASR artifact.

Run:  PYTHONPATH=src python -m evaluation.eval_epochs --n 20
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from core import config
from core.espeak import setup_espeak
from core.logger import logger

MODEL_REPO = 'Dmi1tr13/vits-b1-droid'
DATASET_SHARD = config.DATA_DIR / 'b1_shards' / 'data' / 'train-00007-of-00008.parquet'
DATASET_REPO = 'Dmi1tr13/ljspeech-b1'
LAST_SHARD = 'data/train-00007-of-00008.parquet'
EPOCHS = (0, 1, 2, 3, 4, 5)
SR = 22050


def _load_heldout(n: int) -> tuple[list[str], list[np.ndarray]]:
    """Return (texts, real_b1_audio) for the last ``n`` held-out clips of the shard."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = DATASET_SHARD
    if not path.exists():
        path = Path(
            hf_hub_download(
                DATASET_REPO,
                LAST_SHARD,
                repo_type='dataset',
                local_dir=str(config.DATA_DIR / 'b1_shards'),
            )
        )
    table = pq.read_table(path)
    start = max(0, table.num_rows - n)
    texts = [t.strip() for t in table['text'].to_pylist()[start:]]
    audio = [
        sf.read(io.BytesIO(a['bytes']), dtype='float32')[0]
        for a in table['audio'].to_pylist()[start:]
    ]
    return texts, audio


def _build_model(checkpoint: Path):
    """Load a fine-tuned VITS generator checkpoint ready for inference (CPU)."""
    import torch
    from transformers import AutoTokenizer

    from vits_finetune.checkpoint import load_checkpoint
    from vits_finetune.model import VitsFinetuneModel
    from vits_finetune.model_config import VitsModelConfig

    mc = VitsModelConfig()
    tokenizer = AutoTokenizer.from_pretrained(
        mc.pretrained_model_name, cache_dir=str(mc.cache_dir)
    )
    model = VitsFinetuneModel(mc)
    load_checkpoint(checkpoint, model, map_location='cpu')
    model.eval()
    return model, tokenizer, torch


def evaluate_epochs(n: int = 20, epochs=EPOCHS, out: Path | None = None) -> list[dict]:
    """Synthesize + score the held-out set with every epoch checkpoint."""
    setup_espeak()
    from huggingface_hub import hf_hub_download
    from jiwer import cer, wer

    from evaluation.asr import transcribe
    from evaluation.metrics import compute_mcd

    texts, real_audio = _load_heldout(n)
    logger.info(f'Held-out set: {len(texts)} clips.')

    work = config.ROOT / 'logs' / 'epoch_audio'
    ref_dir = work / 'reference'
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_paths = []
    for i, wav in enumerate(real_audio):
        rp = ref_dir / f'{i:03d}.wav'
        sf.write(str(rp), wav, SR)
        ref_paths.append(str(rp))

    logger.info('Loading Whisper (base)...')

    results: list[dict] = []
    for ep in epochs:
        ckpt = Path(
            hf_hub_download(
                MODEL_REPO,
                f'epoch_{ep}_G.pt',
                local_dir=str(config.MODELS_DIR / 'vits_finetune'),
            )
        )
        logger.info(f'=== epoch {ep}: loading {ckpt.name} ===')
        model, tokenizer, torch = _build_model(ckpt)

        gen_dir = work / f'epoch_{ep}'
        gen_dir.mkdir(parents=True, exist_ok=True)
        hyps, refs, mcds = [], [], []
        for i, text in enumerate(texts):
            inp = tokenizer(text, return_tensors='pt')
            torch.manual_seed(
                1234
            )  # pin the stochastic duration/noise for reproducibility
            with torch.no_grad():
                wav = (
                    model.vits(
                        input_ids=inp.input_ids, attention_mask=inp.attention_mask
                    )
                    .waveform.squeeze(0)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            gp = gen_dir / f'{i:03d}.wav'
            sf.write(str(gp), wav, SR)
            hyps.append(transcribe(str(gp)))
            refs.append(text.lower())
            try:
                mcds.append(compute_mcd(ref_paths[i], str(gp)))
            except Exception as exc:
                logger.warning(f'MCD failed clip {i}: {exc}')
                mcds.append(float('nan'))

        valid = [m for m in mcds if not np.isnan(m)]
        row = {
            'epoch': ep,
            'n': len(refs),
            'wer': float(wer(refs, hyps)),
            'cer': float(cer(refs, hyps)),
            'mcd': float(np.mean(valid)) if valid else None,
        }
        results.append(row)
        logger.success(
            f'epoch {ep}: WER {row["wer"] * 100:.1f}%  CER {row["cer"] * 100:.1f}%  '
            f'MCD {row["mcd"]:.3f} dB'
        )

    # summary table
    logger.info('Per-epoch (lower = better on all three):')
    logger.info(f'{"epoch":>6}{"n":>5}{"WER":>10}{"CER":>10}{"MCD(dB)":>10}')
    for r in results:
        logger.info(
            f'{r["epoch"]:>6}{r["n"]:>5}{r["wer"] * 100:>9.1f}%{r["cer"] * 100:>9.1f}%{r["mcd"]:>10.3f}'
        )

    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding='utf-8')
        logger.info(f'Results -> {out}')
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Per-epoch WER/CER/MCD for the B1 fine-tune.'
    )
    parser.add_argument(
        '--n', type=int, default=20, help='Held-out clips to synthesize per epoch.'
    )
    parser.add_argument('--epochs', type=int, nargs='+', default=list(EPOCHS))
    parser.add_argument(
        '--out', type=Path, default=config.ROOT / 'logs' / 'eval_epochs.json'
    )
    args = parser.parse_args()
    evaluate_epochs(args.n, tuple(args.epochs), args.out)


if __name__ == '__main__':
    main()
