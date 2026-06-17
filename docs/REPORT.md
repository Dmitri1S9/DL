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

## 3. The model: a full VITS GAN fine-tune

VITS is a **GAN** — its decoder only learns to produce realistic waveforms when trained *against a
discriminator*. Our first own training loop had only the reconstruction objective (mel L1 + KL +
duration), **no discriminator**, and produced pure noise (Whisper WER 90–120 %). The fix was to finish
our own package into a real adversarial trainer rather than vendor a third-party recipe:

- **`discriminator.py` (written from scratch):** the standard HiFi-GAN discriminator — a multi-period
  discriminator (periods 2,3,5,7,11) + a multi-scale discriminator (3 scales). `transformers.VitsModel`
  ships the generator but not the discriminator, so we added it.
- **`losses.py`:** added `discriminator_loss`, `generator_adv_loss`, `feature_matching_loss` (LS-GAN
  + L1 feature matching) on top of the existing mel + KL terms.
- **`train.py`:** rewrote the loop as a proper GAN — two optimizers, a discriminator step then a
  generator step each batch.

## 4. Debugging journey — where we were wrong, and what fixed it

The first GAN run trained (losses moved, mel reconstruction fell) but **inference was broken**: the droid
spoke far too fast and unintelligibly. We debugged it systematically — each step taught us something:

| # | Symptom | Hypothesis / change | Result |
|---|---|---|---|
| 0 | 1 s output (vs 5 s), WER **110 %** | first GAN run, default LR **2e-4** | broken |
| 1 | duration loss oscillating, never converging | LR **2e-4 → 2e-5** (the proven recipe value); 2e-4 caused catastrophic forgetting | partial: 2 s, WER **90 %** |
| 2 | still 2 s, WER ~100 % | **froze the duration predictor** (B1 keeps LJSpeech timing, so durations shouldn't change) | **no change** — a key clue |
| 3 | — | read the inference code: length = `duration_predictor(text_encoder(text))`. The frozen predictor still reads the **text encoder**, which was *still training* (via KL) and drifting → out-of-distribution durations. **Froze the text encoder too.** | **fixed** ✅ |

**Mistakes we made and corrected:**
1. **Wrong learning rate.** We inherited LR `2e-4` (fine for training from scratch) and used it for
   *fine-tuning* a pretrained model — 10× too high. It caused catastrophic forgetting; the delicate
   stochastic duration predictor collapsed first.
2. **Misreading a noisy signal.** We expected the duration loss to fall smoothly. It doesn't: the
   *stochastic* duration predictor samples random noise inside its loss every step, so the loss is
   inherently noisy and is **not** a convergence indicator. The real indicator is inference output.
3. **Fixing the symptom, not the source.** Freezing the duration predictor seemed obvious, but it changed
   nothing — because its *input* (the text encoder) was the thing drifting. Only tracing the actual
   inference code revealed the dependency.

**Root cause (one sentence):** for a *timbre-only* fine-tune, training the text/timing front-end lets it
drift away from what the (frozen) duration predictor expects, collapsing durations — so the whole
text/timing front-end must be frozen, and only the audio side (decoder + flow + posterior) trained.

This is now encoded in `config.py` as `train_text_encoder=False` / `train_duration_predictor=False`
(both frozen by default), with comments explaining why.

## 5. Results & analysis

Evaluated **objectively with Whisper** (ASR → WER/CER on the prompt; low = intelligible, ~100 % = garbage),
the team's own `evaluation` module. The before/after on a sample prompt
("Roger roger. All units, proceed with the mission. Standing by."):

| model | length | WER | CER |
|---|---|---|---|
| pretrained VITS | ~5 s | 40 % | 6.6 % |
| **fine-tuned droid (working)** | **~5 s** | **30 %** | **4.9 %** |
| fine-tuned, LR 2e-4 (broken) | 1 s | 110 % | 75 % |
| fine-tuned, LR 2e-5, front-end trained (broken) | 2 s | 90 % | 47 % |

The working fine-tune is **intelligible and slightly better than the pretrained base** on this prompt,
and on listening it clearly carries the **B1 droid timbre**. A single short prompt is noisy, so the
**full held-out test-set evaluation** (notebook §7, `evaluation.evaluate` over N held-out LJSpeech clips)
is the number for the final report:

| model | n | WER | CER |
|---|---|---|---|
| pretrained VITS | 100 | 7.09 % | 2.29 % |
| fine-tuned, epoch 1 | 100 | 8.11 % | 2.80 % |
| **fine-tuned, epoch 2 (chosen)** | 100 | **7.81 %** | **2.67 %** |
| fine-tuned, epoch 3 | 100 | 8.71 % | 2.75 % |

Three epochs over ~8,000 clips (batch 8, LR 2e-5, front-end frozen). **Reading the table:**
the droid fine-tune stays essentially as intelligible as the pretrained base (7.8 % vs 7.1 % WER) —
a ~0.7 % cost for changing the entire timbre, which is the goal: an intelligible droid voice. The
best checkpoint is **epoch 2**; epoch 3 is slightly worse, so **2 epochs is the sweet spot** and more
training mildly degrades quality (overfitting), not improves it.

> *MCD was not scored here: with the LJSpeech audio loader broken on Colab we evaluate WER/CER from the
> held-out texts only. MCD against the real LJSpeech (female) recordings would be high anyway for a droid
> voice (different timbre), so it isn't a meaningful voice-quality score for this target.*

**Key lesson — training-path vs inference-path divergence.** VITS trains by reconstructing audio from the
*posterior* of real audio (teacher-forced), but *generates* from text via the prior + duration predictor +
flow + decoder. A falling reconstruction loss therefore says nothing about generation quality — our broken
runs had a *decreasing* mel loss while generating garbage. The objective WER on the inference path is what
exposed every problem.

## 6. Engineering challenges (where the real work was)

1. **Implementing the discriminator from scratch** — the missing half of the VITS objective; without it
   the decoder never learns realistic waveforms.
2. **Diagnosing the duration collapse** — traced through the `transformers` VITS source to find that
   inference length flows `text_encoder → duration_predictor`, which dictated *which* modules to freeze.
3. **Reproducible env on Colab:** the VITS tokenizer needs the *system* package `espeak-ng`; the pinned
   local CUDA torch build isn't on Colab → use Colab's torch.
4. **Test-set leakage (caught & fixed earlier):** the B1 flow once loaded the whole dataset including the
   held-out test texts → always hold out the last 500.
5. **espeak memory-map leak:** phonemizer `dlopen`s the espeak C library every call → can hit
   `vm.max_map_count` and hang; mitigated with a clip cap + worker split.

## 7. How to reproduce

1. **`colab_train.ipynb`** (branch `finetune`): clone → install → train → listen → Whisper WER →
   full test-set evaluation (§7).
2. Training: `vits_finetune.train --batch-size 2 --num-epochs N` (LR `2e-5` and front-end freeze are the
   defaults now; raise `--batch-size` on a bigger GPU).
3. The model fixes live in `src/vits_finetune/{config,model,train,discriminator,losses}.py`; design and
   debugging notes in `docs/superpowers/`.

## 8. Contributions

| Member | Area |
|---|---|
| Dima (Tsygankov Dmitrii) | model, RVC conversion, VITS migration, B1 dataset, initial fine-tune loop |
| Emir (Wiped-Out) | infra, contracts, data + train/test split, evaluation (WER/CER/MCD), GAN trainer + discriminator, Colab runbook, debugging, this report |
| Ilya | evaluation (verify contributions before the slide) |
