"""Inference: text -> waveform, for sanity-checking a fine-tuned checkpoint.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torchaudio
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vits_finetune.checkpoint import load_checkpoint
from vits_finetune.model import VitsFinetuneModel
from vits_finetune.model_config import VitsModelConfig

logger = logging.getLogger(__name__)


def load_model(
    checkpoint_path: str | Path | None,
    model_config: VitsModelConfig,
    device: torch.device,
) -> VitsFinetuneModel:
    """Load a (optionally fine-tuned) ``VitsFinetuneModel`` ready for inference.
    Returns:
        The model, ready for ``synthesize``.
    """
    model = VitsFinetuneModel(model_config)
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, map_location=device)
    model = model.to(device)
    model.eval()
    return model


def synthesize(
    text: str,
    model: VitsFinetuneModel,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> torch.Tensor:
    """Run text-to-speech inference for a single string.
    Returns:
        1-D float tensor of audio samples at ``model_config.sampling_rate``.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.vits(input_ids=inputs.input_ids)
    waveform = output.waveform.squeeze(0).cpu()    # (1, T) -> (T,)
    return waveform


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

    parser = argparse.ArgumentParser(
        description='Synthesize speech from text using a (fine-tuned) VITS checkpoint.'
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=None,
        help='Fine-tuned checkpoint .pt file (omit to use the pretrained base model).',
    )
    parser.add_argument('--text', required=True, help='Text to synthesize.')
    parser.add_argument('--out', type=Path, default=Path('synthesized.wav'), help='Output .wav path.')
    parser.add_argument(
        '--device', default=None, help='"cuda" or "cpu" (default: auto-detect).'
    )
    args = parser.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model_config = VitsModelConfig()

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.pretrained_model_name, cache_dir=str(model_config.cache_dir)
    )
    model = load_model(args.checkpoint, model_config, device)

    waveform = synthesize(args.text, model, tokenizer, device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    sf.write(str(args.out), waveform.numpy(), model_config.sampling_rate)
    logger.info(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
