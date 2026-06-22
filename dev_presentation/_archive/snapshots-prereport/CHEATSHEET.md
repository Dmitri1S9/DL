# Cheat Sheet — carry this while presenting

## Elevator pitch (memorize)
> "We built end-to-end text-to-speech by **fine-tuning VITS** onto a **custom
> voice we created and published**. Hugging Face ships VITS **inference-only**, so
> we **re-implemented the whole training stack** — alignment search, five losses,
> the HiFi-GAN discriminators, the GAN loop. We evaluate with **Whisper WER/CER**
> (intelligibility) and **MCD** (spectral fidelity)."

## The numbers (know cold)
| | |
|---|---|
| Course / project | 192.151 Intro to DL, 2026S · Project 13 · **Option C (VITS)** |
| Base model | `kakao-enterprise/vits-ljs` (single-speaker, 22050 Hz) |
| Dataset | LJSpeech **13,100** clips → B1 voice `Dmi1tr13/ljspeech-b1` |
| Latent / mels | `z` = 192-dim · 80 mel bands · n_fft 1024 · hop 256 |
| Training budget | batch **16** · lr **2e-4** · segment **16384** · AdamW · ~2 epochs |
| Loss weights | mel **45×** · KL 1× · + duration + adversarial + feature-match |
| Discriminator | MPD periods (2,7,13,29,37,73,97,113,137) + MSD 3 scales |
| **Baseline (pretrained)** | **WER 5.5% · CER 1.6% · MCD 6.38 dB** |
| Tokenizer | **phonemes** (espeak via VitsTokenizer) — NOT characters |
| Training | **trained 3 epochs on L4** (Colab Pro), **epoch 2 best**; discriminator was decisive |
| Team | Dima (model) · Emir (infra/data) · Ilya (eval) · 32 commits |

## The 5 losses
mel-L1 (45×) · KL (prior‖posterior) · duration · adversarial (LSGAN) · feature-matching

## One training step
text-enc→prior · posterior-enc(real audio)→z · flow z→z_p · **MAS** align ·
duration loss · slice segment→decode→waveform · D step (detach fake) · G step (5 losses)

## Must-say lines (rubric ticks)
- "Option C — VITS, end-to-end text→waveform."
- "We report the **training budget**: batch 16, lr 2e-4, ~N epochs."
- "**Failure modes**: mispronunciation, prosody, GAN artifacts."
- "**WER/CER** via Whisper (intelligibility) + **MCD** (spectral) — lower better."
- "Before/after is a one-flag checkpoint swap on identical prompts."

## Three traps → instant answers
1. **"Synthetic data is circular?"** → demonstrates *voice adaptation*; real
   recordings drop into the same pipeline.
2. **"Whisper scoring is circular?"** → independent frozen ASR; constant error
   floor across before/after.
3. **"Did it actually train?"** → machinery built + wired; multi-hour GPU run is
   the last step; mock log ≠ real results, baseline is the real number.

## Demo order (pre-loaded!)
pretrained `test_0000.wav` → B1 `droid_test.wav` → effect `demo_4_excited.wav`
- **Be precise:** dataset droid voice = **RVC** (a *learned* B1 voice model,
  `Homiebear/B1BattleDroid`, `index_rate 0.75 / protect 0.33`). The `demo_*.wav`
  clips = **DSP effects** (ring-mod robot filter) — post-processing, *not* the model.
- **"Why doesn't it sound like a droid?"** Verified by synthesizing our checkpoint:
  fine-tune shifted brightness 965→1364 Hz (toward target), but pitch stays human,
  no ring-mod. RVC transfers *timbre*, not the robot effect → brighter voice, not a
  hard droid. The adaptation worked; the target just isn't a ring-mod robot.

## Known limitations (if asked — answer honestly)
- mel target truncated ~3 frames (STFT `center=False`); `min()` handles it.
- duration loss has no config weight (added at 1).
- trained (3 ep, L4, epoch 2 best); formal fine-tuned WER/CER/MCD table still TODO; mock log ≠ results.
- char-vs-phoneme ablation = stub (model already uses phonemes).

## If cut to 30 seconds
See the elevator pitch above. Lead with "one model, end-to-end" + "we wrote VITS
training from scratch."
