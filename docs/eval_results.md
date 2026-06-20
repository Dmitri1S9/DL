# Objective evaluation — VITS B1-droid fine-tune

Held-out set: last-5% of `Dmi1tr13/ljspeech-b1` (20 clips, unseen in training). WER/CER vs prompt text (Whisper); MCD & F0 RMSE vs the B1 droid reference audio. All metrics: lower = better.

| model | n | WER % | CER % | MCD (dB) | F0 RMSE (Hz) |
|---|---|---|---|---|---|
| base | 20 | 9.09 | 1.79 | 10.48 | 30.75 |
| finetuned | 20 | 29.22 | 13.93 | 6.19 | 32.36 |

**Reading it.** Fine-tuning **moved the voice toward the B1 droid target**: MCD drops from
**10.48 → 6.19 dB** (the fine-tuned output is spectrally much closer to the droid reference than
the pretrained voice is). The cost is **intelligibility**: WER rises **9.1% → 29.2%** — the model
trades some clarity for the new timbre. Pitch (F0 RMSE) is roughly unchanged. This is the central
trade-off of a timbre-only fine-tune, and the WER cost is the main thing left to reduce (see the
front-end-freeze fix discussed in the report).
