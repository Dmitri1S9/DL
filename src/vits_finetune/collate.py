"""Batch collation for ``VitsFinetuneDataset``."""

from __future__ import annotations

import torch


def _pad_input_ids(batch):
    return (
        torch.nn.utils.rnn.pad_sequence(
            [i['input_ids'] for i in batch], batch_first=True, padding_value=0
        ),
        torch.tensor([len(i['input_ids']) for i in batch], dtype=torch.long),
    )


def _pad_waveforms(batch):
    return (
        torch.nn.utils.rnn.pad_sequence(
            [i['waveform'] for i in batch], batch_first=True, padding_value=0.0
        ).unsqueeze(1),
        torch.tensor([len(i['waveform']) for i in batch], dtype=torch.long),
    )


def _pad_linear_specs(batch):
    return (
        torch.nn.utils.rnn.pad_sequence(
            [i['linear_spec'].transpose(0, 1) for i in batch],
            batch_first=True,
            padding_value=0.0,
        ).transpose(1, 2),
        torch.tensor([i['linear_spec'].shape[1] for i in batch], dtype=torch.long),
    )


def _pad_mel_specs(batch):
    return (
        torch.nn.utils.rnn.pad_sequence(
            [i['mel_spec'].transpose(0, 1) for i in batch],
            batch_first=True,
            padding_value=0.0,
        ).transpose(1, 2),
        torch.tensor([i['mel_spec'].shape[1] for i in batch], dtype=torch.long),
    )


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad a list of ``VitsFinetuneDataset.__getitem__`` outputs into one batch.

    Args:
        batch: list of dicts, each with ``"input_ids"``, ``"waveform"``,
            ``"linear_spec"``, ``"mel_spec"`` (see ``dataset.VitsFinetuneDataset``).

    Returns:
        Dict with padded ``"input_ids"``, ``"linear_spec"``, ``"mel_spec"``,
        ``"waveform"`` plus ``"input_lengths"``, ``"spec_lengths"``,
        ``"waveform_lengths"`` (each ``(B,)`` ``LongTensor``s), ready for
        ``model.VitsFinetuneModel.forward_train``.
    """
    padded_ids, input_lengths = _pad_input_ids(batch)
    padded_wav, wav_lengths = _pad_waveforms(batch)
    padded_linear_spec, spec_lengths = _pad_linear_specs(batch)
    padded_mel_spec, _ = _pad_mel_specs(batch)

    return {
        'input_ids': padded_ids,
        'linear_spec': padded_linear_spec,
        'mel_spec': padded_mel_spec,
        'waveform': padded_wav,
        'input_lengths': input_lengths,
        'spec_lengths': spec_lengths,
        'waveform_lengths': wav_lengths,
    }
