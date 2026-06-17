"""Batch collation for ``VitsFinetuneDataset``.
"""

from __future__ import annotations

import torch

padding_input_ids = lambda batch: (torch.nn.utils.rnn.pad_sequence(
      [i["input_ids"] for i in batch], batch_first=True, padding_value=0
    ),
    torch.tensor([len(i["input_ids"]) for i in batch], dtype=torch.long)
)

padding_waveforms = lambda batch: (torch.nn.utils.rnn.pad_sequence(
      [i["waveform"] for i in batch], batch_first=True, padding_value=0.0
    ).unsqueeze(1),
    torch.tensor([len(i["waveform"]) for i in batch], dtype=torch.long)
)

padding_linear_specs = lambda batch: (torch.nn.utils.rnn.pad_sequence(
      [i["linear_spec"].transpose(0, 1) for i in batch], batch_first=True, padding_value=0.0
    ).transpose(1, 2),
    torch.tensor([i["linear_spec"].shape[1] for i in batch], dtype=torch.long)
)

padding_mel_specs = lambda batch: (torch.nn.utils.rnn.pad_sequence(
      [i["mel_spec"].transpose(0, 1) for i in batch], batch_first=True, padding_value=0.0
    ).transpose(1, 2),
    torch.tensor([i["mel_spec"].shape[1] for i in batch], dtype=torch.long)
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
    padded_ids, input_lengths = padding_input_ids(batch)
    padded_wav, wav_lengths = padding_waveforms(batch)
    padded_linear_spec, spec_lengths = padding_linear_specs(batch)
    padded_mel_spec, mel_lengths = padding_mel_specs(batch)

    # to check that it works
    # hop = 256
    # expected_spec = (wav_lengths - config.n_fft) // hop + 1
    # assert torch.all(spec_lengths == expected_spec), \
    #     f"frame mismatch: spec={spec_lengths} expected={expected_spec}"
    
    return {
        "input_ids": padded_ids,
        "linear_spec": padded_linear_spec,
        "mel_spec": padded_mel_spec,
        "waveform": padded_wav,
        "input_lengths": input_lengths,
        "spec_lengths": spec_lengths,
        "waveform_lengths": wav_lengths,
    }


# if __name__ == "__main__":
#     from transformers import AutoTokenizer

#     from vits_finetune.config import TrainingConfig
#     from vits_finetune.dataset import VitsFinetuneDataset
#     from vits_finetune.model_config import VitsModelConfig

#     config = TrainingConfig()
#     model_config = VitsModelConfig()
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_config.pretrained_model_name, cache_dir=str(model_config.cache_dir)
#     )
#     ds = VitsFinetuneDataset(config, tokenizer)

#     b = collate_fn([ds[0], ds[1], ds[2]])
#     for k, v in b.items():
#         print(k, tuple(v.shape), v.dtype)
 