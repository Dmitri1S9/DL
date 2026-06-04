"""Evaluate generated audio against the test manifest: WER / CER / MCD.

Consumes the contract (core.dto / core.contracts): a manifest (texts + reference
wavs) and a directory of generated wavs named ``<id>.wav``. Produces an
``EvalResult``; ``compare`` prints the pretrained-vs-fine-tuned before/after table.

``compute_asr=False`` (the --mock flag) skips the heavy Whisper step and leaves
WER/CER as None, so the wiring can be exercised offline.
"""

import argparse
from pathlib import Path

import numpy as np

from core import config
from core.contracts import read_manifest
from core.dto import EvalResult
from core.logger import logger
from evaluation.metrics import compute_cer, compute_mcd, compute_wer


def _mean_mcd(scores: list[float]) -> float | None:
    """Average MCD over the clips that scored (NaN = a per-clip failure)."""
    valid = [s for s in scores if not np.isnan(s)]
    return float(np.mean(valid)) if valid else None


def evaluate(
    generated_dir: Path = config.GENERATED_DIR,
    manifest_path: Path = config.TEST_MANIFEST,
    label: str = 'model',
    compute_asr: bool = True,
) -> EvalResult:
    """Score one set of generated audio against the manifest."""
    items = read_manifest(manifest_path)
    generated_dir = Path(generated_dir)

    gen_paths: list[str] = []
    texts: list[str] = []
    mcd_scores: list[float] = []
    for item in items:
        gen = generated_dir / f'{item.id}.wav'
        if not gen.exists():
            logger.warning(f'Missing generated wav for {item.id}, skipping')
            continue
        gen_paths.append(str(gen))
        texts.append(item.text)
        try:
            mcd_scores.append(compute_mcd(item.ref_audio, str(gen)))
        except Exception as exc:
            logger.warning(f'MCD failed for {item.id}: {exc}')
            mcd_scores.append(float('nan'))

    result = EvalResult(
        label=label,
        n=len(gen_paths),
        wer=compute_wer(gen_paths, texts) if compute_asr else None,
        cer=compute_cer(gen_paths, texts) if compute_asr else None,
        mcd=_mean_mcd(mcd_scores),
    )
    logger.success(f'[{label}] scored {result.n} clips: {result.model_dump()}')
    return result


def compare(results: list[EvalResult]) -> None:
    """Print a before/after table (e.g. pretrained vs fine-tuned)."""
    logger.info('Comparison (lower = better):')
    logger.info(f'{"model":<16}{"n":>4}{"WER":>10}{"CER":>10}{"MCD (dB)":>12}')

    def fmt(value: float | None) -> str:
        return f'{value:.4f}' if value is not None else '—'

    for r in results:
        logger.info(
            f'{r.label:<16}{r.n:>4}{fmt(r.wer):>10}{fmt(r.cer):>10}{fmt(r.mcd):>12}'
        )


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate generated audio.')
    parser.add_argument('--generated', type=Path, default=config.GENERATED_DIR)
    parser.add_argument('--manifest', type=Path, default=config.TEST_MANIFEST)
    parser.add_argument('--label', default='model')
    parser.add_argument('--out', type=Path, help='Optional path to write results JSON')
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Skip the heavy Whisper ASR step (MCD only)',
    )
    args = parser.parse_args()
    result = evaluate(
        args.generated, args.manifest, args.label, compute_asr=not args.mock
    )
    compare([result])
    if args.out:
        Path(args.out).write_text(result.model_dump_json(indent=2), encoding='utf-8')
        logger.info(f'Results -> {args.out}')


if __name__ == '__main__':
    main()
