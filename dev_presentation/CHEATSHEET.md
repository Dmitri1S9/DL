# Cheat Sheet — Project 13: Text to Speech (one-page carry card)

**Course:** 192.151 Introduction to Deep Learning · 2026S · Project 13 · **Option C (VITS)** · TU Wien
**Team:** Dima (model & training) · Emir (infra, data, engineering) · Ilya (evaluation)
**Repo:** github.com/Dmitri1S9/DL · 🤗 dataset `Dmi1tr13/ljspeech-b1` · checkpoints `Dmi1tr13/vits-b1-droid`

## The one-liner
We fine-tuned a pretrained VITS model onto a **custom B1 droid voice** we built with RVC. The timbre
moved to the droid (MCD ↓), at a measured cost to intelligibility (WER ↑).

## The numbers
- **Base model:** `kakao-enterprise/vits-ljs` (single speaker, 22050 Hz)
- **Dataset:** LJSpeech (13,100 clips) → RVC → B1 droid → `Dmi1tr13/ljspeech-b1`. Held-out = **last 5%**.
- **Trained:** local **RTX 5060 (8 GB)**, **5 epochs**, **≈ 9.7 GPU-h** (+~3× in failed runs)
- **Config:** `lr 1e-4` · batch **1 + grad-accum** · AMP fp16 · `segment 8192` · `grad-clip 10` · `disc-warmup 1500`
- **5 losses:** recon (L1 mel, **×45**) · KL · duration · adversarial (LSGAN) · feature-matching
- **Discriminator (from scratch):** MPD prime periods `(2, 7, 13, 29, 37, 73, 97, 113, 137)` · MSD 3 scales

## Results (20 held-out clips, lower = better)
| model | WER | CER | MCD (dB) | F0 RMSE (Hz) |
|---|---|---|---|---|
| base (pretrained) | 9.1% | 1.8% | 10.48 | 30.8 |
| **fine-tuned (droid)** | **29.2%** | **13.9%** | **6.19** | 32.4 |

**Read it:** MCD **10.5 → 6.2 dB** = learned the droid timbre (fine-tuning worked). WER **9 → 29%** =
traded some clarity. Classic timbre-vs-intelligibility trade-off.

## The debugging story (the best part)
Nothing converged at first. Three bugs:
1. **Duration loss ~100× too large** (averaged wrong) → `.sum()/mask.sum()` → dur **300 → 1.7**
2. **Grad-clip too tight** killed the generator update → loosen it → recon broke its plateau
3. **Fresh discriminator corrupted the decoder** → **warm up D 1500 steps** + lower LR
→ recon **0.85 → 0.40**, correct rhythm. (Lesson: a falling loss ≠ good speech; only inference WER caught it.)

## Failure modes (honest)
- Intelligibility cost (WER up); some **short prompts collapse** (durations too short, e.g. "Roger roger…" → 0.8 s)
- Timbre ≠ hard robot: RVC moves **timbre**, not the ring-mod buzz (separate DSP effect)
- GAN artifacts early; over-fits past the best epoch
- Russian accent = **future work** (input pre-processing)
- **Future fix for WER:** freeze the text/timing front-end (keeps durations stable)

## Demo (slide 16) — same sentence, before/after
"The mission is complete. All systems are nominal. Returning to base."
1. **Base VITS** (pretrained) → 2. **Fine-tuned droid** (same line). Slide 17 = the collapse clip.

## Q&A routing
Model/training → **Dima** · Evaluation → **Ilya** · Data/engineering → **Emir**. Repeat the question first.

## Three things to remember
- MCD down = it learned the voice; WER up = the cost. That's the headline.
- We **re-implemented VITS training from scratch** (HF ships inference-only): forward pass, MAS, 5 losses, discriminator, GAN loop.
- A converging loss does **not** prove good speech — that's why we evaluate the output (WER/CER/MCD/F0).
