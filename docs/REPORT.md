# Project 13 — Teaching VITS to Speak Like a Droid

**Final report · 192.151 Introduction to Deep Learning, 2026S**

**Team:** Dmitrii Tsygankov (model & fine-tuning) · Emir (infra, data, evaluation) · Ilya (evaluation)

---

## 1. What we built

A **text-to-speech** system that learns to speak in a Star Wars **B1 Battle Droid** voice,
via **voice-conversion distillation**:

```
LJSpeech text ──► [VITS synth ─► RVC B1]  ──► droid dataset ──► fine-tune VITS ──► droid speech from text
                   teacher (used once)        (HF Hub)          student model
```

- **Base model:** pretrained **VITS** (`kakao-enterprise/vits-ljs`), end-to-end text → waveform.
- **Teacher:** an **RVC B1** voice-converter, used **once** to build a droid-voiced dataset.
- **Student:** VITS fine-tuned on that dataset so it produces the droid voice **natively** from text.

**Why distillation, not "TTS → RVC at inference":** baking the voice into the trained model makes the
droid the model's *own* voice — a real *train → measure → report* experiment, which is what the course
grades (process & analysis, not SOTA quality).

## 2. Data

- **Source — LJSpeech:** ~13,100 clips, single voice. **Last 500 clips held out** as the test set.
- **Target — [`Dmi1tr13/ljspeech-b1`](https://huggingface.co/datasets/Dmi1tr13/ljspeech-b1):** LJSpeech text
  resynthesized with VITS, voice-converted to B1 via RVC, 22050 Hz. Same transcripts, droid timbre.

## 3. Training (what we actually ran)

Fine-tuned VITS on **Google Colab (T4)** from the clean `finetune` branch — clone → install → train →
generate → evaluate, no patch cells. Settings mirror the known-good baseline: **batch 2, LR 2e-4**, on a
**4,000-clip subset** (test set still held out), `num_workers=4`.

- Loss fell fast in the first ~50 steps then **plateaued from ~step 100**: `recon_loss` ≈ 0.4,
  `kl_loss` ≈ 68 (never converged), `duration_loss` oscillating ~150–400. total_loss ~250–450.
- Reached **2,000 steps** (the baseline's setting), saving `step_1000.pt` and `step_2000.pt`.
- Infra note: the run **hung after each checkpoint save** (espeak memory-map leak accumulating across the
  DataLoader workers); we reached 2,000 steps by resuming in a fresh process. See §5.

## 4. Results & honest analysis

Evaluated **objectively with Whisper** (ASR → WER on the held-out prompt; low = intelligible,
~100% = garbage). This removes the need for subjective listening.

| model | WER | Whisper transcription |
|---|---|---|
| pretrained VITS | **10–20 %** | "roger roger, all units, proceed with the mission, standing by" ✅ intelligible |
| fine-tuned, 1000 steps | **90 %** | "boderaer or i'll get it. with magic, hey, it goes!" ❌ |
| fine-tuned, 2000 steps | **120 %** | "sorry, i owe ya everfield if thy he monday get spien i" ❌ |

**Conclusion (negative, but rigorous):** the droid fine-tune does **not** produce intelligible speech with
this setup — at *either* 1000 or 2000 steps. The fine-tuned model's output is noise (WER ≥ 90 %), while the
same code on the pretrained checkpoint is fully intelligible (so the inference/eval code is correct — it's
the fine-tuned weights that break generation).

**Why (analysis):** the training loss plateaued almost immediately (recon ~0.4, KL never dropping from ~68).
VITS trains on one path (reconstruct audio from the *posterior* of real audio) but *generates* on another
(prior + stochastic duration predictor + flow + decoder). Low recon loss with a flat KL means the prior /
generation path never learned to match the decoder → at inference it feeds the decoder out-of-distribution
latents → noise. The most likely root cause: the training objective here is **reconstruction + KL +
duration only, with no adversarial (discriminator) loss** — and real VITS relies on adversarial training to
make the decoder produce good waveforms. Without it, more steps don't help (confirmed: 2000 ≈ 1000).

**Next steps:** add the HiFi-GAN-style discriminator + adversarial loss to `vits_finetune` (or adopt an
established VITS fine-tuning recipe); and verify what the repo's `droid_test.wav` was actually produced by
(possibly the RVC post-process path, not this fine-tuned model).

## 5. Engineering challenges (where the real work was)

1. **Reproducible env on Colab:** the VITS tokenizer needs the *system* package `espeak-ng`; the pinned
   `torch==2.11.0+cu128` (a local GPU build) isn't on Colab → use Colab's torch instead.
2. **Test-set leakage (caught & fixed):** the B1 training flow loaded the *entire* dataset, including the
   500 held-out test texts → fixed by always holding out the last 500 in `vits_finetune/dataset.py`.
3. **espeak memory-map leak:** phonemizer `dlopen`s the espeak C library on every call and never releases
   it → hits `vm.max_map_count` (~65k) and the run **hangs/crashes** after ~1000 steps. Colab forbids
   raising the limit, so we trained on a subset + split the load across workers, and reached 2000 steps by
   resuming in a fresh process. *Diagnostic lesson: a crash that got **worse** with fewer processes pointed
   to a per-process resource leak, not a concurrency bug.*
4. **Low loss ≠ good audio (the key lesson):** a falling/low training loss does **not** guarantee good
   generation when training and inference use different paths. The objective Whisper WER exposed this — the
   model that "trained fine" generates noise.

## 6. Contributions & evolution (git, 26 commits)

| Member | Commits | Area |
|---|---|---|
| Dima (Tsygankov Dmitrii) | 14 | model, RVC conversion, VITS migration, B1 dataset, fine-tuning loop |
| Emir (Wiped-Out) | 12 | infra, restructure/tooling, data + train/test split, evaluation (MCD/WER), Colab runbook, README |
| Ilya | 0 | (no commits in history — verify before the contributions slide) |

**Timeline:** 30 May — Dima seeds the TTS + droid-effects prototype → 4 Jun — Emir builds the engineering
foundation (structure, contracts, eval, Colab, README) → 4 Jun — Dima's RVC + SpeechT5→VITS migration →
13–14 Jun — Dima builds the B1 dataset + the real fine-tune loop → 16–17 Jun — Emir runs the fine-tune
end-to-end on Colab, debugs the environment, measures WER, and produces this report + presentation.

## 7. How to reproduce

1. `colab_train.ipynb` (branch `finetune`): clone → install → train → listen → Whisper WER.
2. Training command: `vits_finetune.train --batch-size 2 --num-epochs 1 --max-train-clips 4000 --num-workers 4`
   (resume from a `step_*.pt` to continue past the espeak hang).
3. Fixes live in `src/vits_finetune/{config,dataset,train}.py` (held-out test set, `max_train_clips`, quiet logs).
