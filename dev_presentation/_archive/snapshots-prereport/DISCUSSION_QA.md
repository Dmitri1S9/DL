# Discussion / Q&A — prepared answers (5–10 min)

Anticipated questions for Project 13 (VITS TTS). Each has a **30-second answer**
and a deeper follow-up. Know the first line cold.

---

### Model choice

**Q: Why VITS over FastSpeech2 or Tacotron2?**
One end-to-end model instead of two — no separate vocoder to train and keep in
sync, and no intermediate mel that gets predicted then re-synthesized (where
two-stage systems lose quality). VITS's latent + adversarial modeling also gives
more natural prosody. *Trade-off:* training is more complex (flow + GAN + MAS).

**Q: Why fine-tune instead of training from scratch?**
From-scratch VITS is days of multi-GPU training. Fine-tuning a strong LJSpeech
checkpoint reaches a new target voice in a few GPU-hours — feasible on free
Colab, and still real training we can report (epochs/batch/GPU-hours).

**Q: What changes between the pretrained and fine-tuned model — all weights?**
We fine-tune the generator end-to-end (text encoder, posterior, flow, duration,
decoder) plus train a fresh discriminator. You *can* freeze the front-end and
only adapt the decoder for a pure timbre change — there was an experiment in that
direction; the current loop fine-tunes the full generator.

---

### Data

**Q: Why a synthetic RVC dataset instead of real recordings?**
We needed ~24h of *paired* (text, target-voice audio). Recording that is
infeasible; RVC voice-converts the existing 13,100 LJSpeech clips into the target
voice while preserving pronunciation, giving instant paired data.

**Q: Isn't training on VITS-then-RVC audio just learning your own artifacts?**
Partly — it's self-distillation-like, and a fair critique. It's fine for our
goal (*demonstrating voice adaptation*). For production you'd fine-tune on real
target-voice recordings; the pipeline is identical, only the audio source changes.

**Q: Why resample to 22050 Hz?** That's `vits-ljs`'s native rate; the posterior
encoder and decoder are built for it. RVC outputs 48 kHz, so we downsample.

**Q: Held-out test set — is it the droid voice too?**
No, deliberately. Eval stays on the **original LJSpeech audio** so before/after
is apples-to-apples on identical prompts. (Last 500 LJSpeech clips.)

---

### Method / architecture

**Q: What does MAS actually do?**
Finds the best *monotonic* alignment between text tokens and audio frames with no
alignment labels — a Viterbi-like DP over per-(token,frame) log-likelihoods.
Monotonic = speech never jumps back to an earlier token. We implemented it from
scratch as a vectorized forward pass + traceback under `no_grad`.

**Q: Why segment-slice the audio for the decoder?**
The decoder (waveform generation) is the expensive part. Like HiFi-GAN, we train
it on a fixed-length random crop (`segment_size`) rather than the full utterance
— large memory/speed win, no quality loss because the decoder is convolutional.

**Q: Why is the reconstruction loss weighted 45×?**
It's the dominant supervision — getting the mel spectrum right. The other terms
(KL, duration, adversarial, feature-matching) are regularizers / realism terms.
45 is the standard VITS weighting.

**Q: MPD vs. MSD — why both?**
MPD reshapes audio by *periods* to catch periodic/harmonic structure (pitch);
MSD looks at multiple *time scales*. They catch different artifact classes;
together they cover more than either alone. Both from HiFi-GAN.

**Q: What is feature-matching loss for?**
L1 between the discriminator's intermediate activations on real vs. generated
audio. It stabilizes GAN training and improves quality — the generator matches
the *features* the discriminator computes, not just its final real/fake verdict.

**Q: LSGAN — why least-squares instead of vanilla GAN loss?**
Least-squares GAN gives smoother gradients and is more stable than the original
cross-entropy GAN — standard in HiFi-GAN/VITS.

---

### Evaluation

**Q: Why WER *and* MCD — isn't one enough?**
They measure different things. WER/CER = *intelligibility* (can an ASR read it
back?). MCD = *spectral fidelity* (does it match the reference acoustically?). A
sample can score well on one and badly on the other.

**Q: Isn't using Whisper to score circular / unfair?**
Whisper is an independent off-the-shelf ASR — it's an *intelligibility proxy*,
exactly as the brief specifies, not part of our training. Its own errors are a
small constant floor across both before and after, so the *comparison* stays
valid.

**Q: Your baseline WER is 5.5% — is that good?**
Yes for pretrained `vits-ljs` on a sanity sample — it's already a strong model.
The point of fine-tuning isn't to beat that on LJSpeech, it's to move the
*voice* while keeping intelligibility comparable.

---

### Status / engineering

**Q: Did you actually finish training?**
The full fine-tuning loop is implemented and wired — forward pass, MAS, five
losses, both discriminators, the GAN loop. The remaining step is the multi-hour
GPU run that produces the final fine-tuned metrics and the B1 demo. *(Update live
if it's done.)*

**Q: How long does training take / GPU-hours?**
Defaults: batch 16, lr 2e-4, 2 epochs on a Colab GPU, scalable via
`--num-epochs`. Exact GPU-hours come from the run; fine-tuning is hours, not days.

**Q: What was the hardest part?**
Re-implementing VITS *training* on HF's inference-only `VitsModel`: MAS, the
segment-slice decoder path, KL between the posterior and the MAS-aligned prior,
and getting a stable GAN (MPD+MSD + feature matching).

**Q: How did three people work without stepping on each other?**
Contract-first: stages communicate through tiny contracts (a `{id,text,ref_audio}`
manifest, an `EvalResult` DTO) in `core/`, never each other's internals. Model,
data, and eval were independently buildable and testable — and the A→C model swap
barely touched downstream code.

---

### Curveballs

**Q: Could this do multi-speaker / voice cloning?**
VITS supports speaker embeddings; `vits-ljs` is single-speaker
(`num_speakers=1`). Multi-speaker would need a speaker-conditioned checkpoint
(e.g. VCTK) and embeddings — architecturally supported, out of scope here.

**Q: Character vs. phoneme tokens?**
The model **already uses phonemes** — `vits-ljs`'s `VitsTokenizer` runs `espeak-ng`
grapheme→phoneme (`phonemize=True`), so training and inference are phoneme-based.
Mispronunciations are espeak g2p errors on rare words/names, not a missing-phoneme
problem. The repo's `tokenization/tokenizer.py` is a separate **stub** for a
*char-vs-phoneme ablation* (future work) against that phoneme default.

**Q: What would you do with more time?**
Run the full fine-tune and report numbers; add the phoneme tokenizer ablation;
try freezing the front-end for a cleaner timbre-only transfer; and a small MOS
(human listening) study to complement WER/MCD.

---

# Extended bank — by difficulty

## Tier 1 — warm-ups (everyone should nail these)

**Q: In one sentence, what is VITS?** A single neural network that turns text
straight into a speech waveform, end-to-end, with no separate vocoder.

**Q: What dataset did you use?** LJSpeech (one English speaker, 13,100 clips) as
the base; for fine-tuning, a voice-converted "B1 droid" version we built and
published on the HF Hub.

**Q: What does "fine-tuning" mean here?** Continue training from the pretrained
`vits-ljs` weights instead of random init, so we adapt to a new voice in
GPU-hours rather than days.

**Q: What metrics, and which direction is good?** WER, CER, MCD — all
**lower = better**.

**Q: Is it single- or multi-speaker?** Single-speaker (`num_speakers=1`).

## Tier 2 — architecture depth

**Q: Walk me through one training step.** Text encoder → prior; posterior encoder
(real audio) → latent `z`; flow `z→z_p`; MAS aligns text to frames; duration
loss; slice a segment of `z`, decode to waveform; then five-term generator loss
and a discriminator step. (See `TECHNICAL_APPENDIX.md §E`.)

**Q: What's the latent variable `z`, intuitively?** A frame-level acoustic code
of the speech (192-dim). The posterior infers it from real audio in training; at
inference we sample it from the text-conditioned prior.

**Q: Why both a prior and a posterior encoder?** It's a conditional VAE: the
posterior gives a high-quality `z` from real audio (training only); the text
prior learns to produce a matching `z` from text alone (KL term), so inference
works without audio.

**Q: What's the normalizing flow for?** It makes the text prior expressive enough
to match the rich posterior. It's invertible, so we train forward (`z→z_p`) and
sample in reverse at inference.

**Q: Why a *stochastic* duration predictor?** Speech is one-to-many — same text,
many rhythms. A deterministic predictor averages them to flat prosody; the
stochastic one samples durations for natural variation.

**Q: How are durations supervised with no labels?** MAS produces a hard
alignment; the number of frames per token *is* the duration target.

**Q: Why MAS over attention?** Attention can skip/repeat/babble; MAS guarantees a
monotonic, complete alignment by construction → no alignment collapse.

**Q: Is MAS learned?** No — it's a dynamic-programming search run under
`no_grad`. Gradients flow through the *aligned prior* it produces, not the argmax.

**Q: What does the decoder architecture look like?** HiFi-GAN-style: stacked
transposed convolutions upsampling `z` to the raw waveform.

## Tier 3 — the GAN / discriminator

**Q: Why two discriminators (MPD + MSD)?** MPD catches periodic/pitch structure
(reshaping audio by prime periods); MSD catches multi-time-scale structure
(pooled copies). Complementary failure coverage.

**Q: Why prime periods in MPD?** Co-prime periods minimize overlap so the
branches see genuinely different views of the harmonic structure.

**Q: What stabilizes the GAN?** Feature-matching loss (L1 on discriminator
activations), least-squares (LSGAN) loss, and the heavily-weighted mel
reconstruction anchoring the generator.

**Q: Why `.detach()` the fake audio in the discriminator step?** So the
discriminator's loss doesn't backprop into the generator — they're trained
alternately, not jointly.

**Q: What if the discriminator overpowers the generator?** Classic GAN collapse;
mitigations are the mel anchor (45×), feature matching, and tuning LR. We use a
shared `lr=2e-4` for both with AdamW.

## Tier 4 — hard / trap questions

**Q (trap): Your fine-tuning data is model-generated — aren't you just distilling
VITS into itself?** Honest answer: partially, yes — it's self-distillation-like
for the *content*, with RVC changing the timbre. It's valid for demonstrating
**voice adaptation**, which is our goal. For a production voice you'd fine-tune on
real target-voice recordings; the pipeline is unchanged, only the audio source.

**Q (trap): Whisper scoring your audio — circular?** No. Whisper is an
independent, frozen, off-the-shelf ASR, exactly the brief's "intelligibility
proxy." It never touches training. Its own error is a roughly constant floor
across before/after, so the *comparison* is valid even if the absolute number
isn't perfect.

**Q (trap): Did anything actually train, or is this all just code?**
It trained. We fine-tuned for **3 epochs on an NVIDIA L4** (Colab Pro); the
**second epoch came out best**, and we have that checkpoint (~1 GB) plus a
reproducible notebook and a local HTML demo. What's still outstanding is the
formal fine-tuned **WER/CER/MCD table** — the pretrained baseline (WER 5.5%) is
the only *tabulated* number so far. (Note: the committed `training_log.json` is a
stale placeholder, not these results.)

**Q: MCD of 6.38 dB — good or bad?** Mid-range; ~5–8 dB is typical for decent
TTS, 0 = identical to reference. It's a sanity baseline, not a SOTA claim.

**Q: WER 5.5% but CER 1.6% — why the gap?** WER penalizes a whole word for any
error; CER counts characters, so small slips (one phoneme) hurt CER far less.
The gap means most "errors" are minor.

**Q: Could MCD be misleading?** Yes — DTW alignment can mask timing errors, and
MCD ignores phase/perceptual masking. That's why we pair it with WER (content)
and a listening demo (perception).

**Q: Why not report MOS (human listening)?** MOS needs a panel of listeners and
is out of scope for the compute/time budget; WER+MCD are the objective proxies
the brief asks for. We'd add MOS with more time.

**Q: How do you know fine-tuning won't just degrade intelligibility?** We hold the
eval set on original LJSpeech prompts and compare WER before/after; if WER spikes,
the LR was too high or it over-fit the timbre — the experiment would catch it.

**Q: What's your effective receptive field / why segment 16384?** 16384 samples
≈ 0.74 s ≈ 64 mel frames — long enough to capture phoneme-scale structure for the
decoder/discriminator, short enough to be cheap. Standard VITS magnitude.

**Q: Memory / OOM strategy?** Lower `--batch-size` (the Colab runbook notes 4),
the segment slice already caps decoder cost, and gradient flows only through the
sliced segment.

## Tier 5 — "what about…" extensions

**Q: Multi-speaker / voice cloning?** VITS supports speaker embeddings; our
checkpoint is single-speaker. Multi-speaker needs a speaker-conditioned base
(e.g. VCTK) + embeddings — architecturally supported, out of scope.

**Q: Other languages?** Needs a tokenizer/checkpoint for that language and matching
data; the training stack is language-agnostic.

**Q: Streaming / real-time?** VITS is non-autoregressive, so inference is fast and
parallel, but the current code synthesizes whole utterances; true streaming needs
chunked decoding.

**Q: Phoneme vs. character tokens — expected effect?** We already run **phonemes**
(espeak via `VitsTokenizer`). A char-vs-phoneme *ablation* would quantify the gap;
it's stubbed in `tokenization/` as future work.

---

# Questions to ask the examiners back (shows depth)

- "Would you prioritize a phoneme front-end or real target-voice recordings as
  the next step?"
- "For this voice-adaptation setting, do you weight intelligibility (WER) or
  spectral fidelity (MCD) more?"
- "Is a small MOS study worth it here, or are objective metrics sufficient for
  the scope?"

---

# Tier 6 — for an examiner who actually reads the repo

**Q: Is the "droid voice" produced by the model, or is it just an audio effect?**
Two different things — be precise. The **training dataset** voice comes from **RVC
(Retrieval-based Voice Conversion)** — a *trained* B1 voice model — so fine-tuning
VITS to reproduce it is a genuine voice-adaptation task. Separately, the repo has
**DSP voice effects** (`effects.py`) — a ring-modulator "robot" filter chain — which
produced the fun `demo_*.wav` clips. **Those effects are post-processing, not the
model**, and they're explicitly excluded from evaluation.

**Q: Why RVC for the dataset instead of just applying that robot DSP effect to
everything?**
A DSP effect is a fixed filter you could slap on at inference anyway — it wouldn't
test whether the *model* learned anything. RVC gives a genuine, consistent learned
timbre, so fine-tuning VITS to reproduce it is a real adaptation task with a
measurable before/after.

**Q: How exactly did the RVC step work?**
A Docker container (CUDA 12.8) running RVC-Project with a B1 model
(`Homiebear/B1BattleDroid`, a `.pth` + FAISS `.index`). Key params: `index_rate
0.75` (75% target-voice characteristics), `protect 0.33` (preserve consonant
clarity), `f0_method` pm/rmvpe, no pitch shift. It changes timbre, not content.

**Q: What were the hardest *engineering* pain points (not modelling)?**
The RVC environment — pinning `torch==2.11.0+cu128`, removing `fairseq` from RVC's
requirements and reinstalling it against `omegaconf==2.0.6`, and a CUDA 12.8-vs-12.9
driver mismatch we diagnosed with a profiling script. Real deployment grit.

**Q: Are there known bugs or limitations in the training code? (honesty test)**
Yes, two minor ones we're aware of: (1) the predicted-vs-target **mel lengths differ
by ~3 frames** (STFT with `center=False` on a 16384-sample segment yields 61 frames
vs the 64-frame target) — we truncate to the shorter with a `min()`, so it trains
fine but drops a few frames of the reconstruction target; the clean fix is
`center=True` or padding the segment. (2) The **duration loss has no explicit weight**
in the config (it's added at weight 1 alongside the weighted mel/KL terms). Neither
breaks training; both are easy to tighten.

**Q: Phonemes or characters?** **Phonemes** — `vits-ljs` tokenizes via `espeak-ng`
(`phonemize=True`), so the whole pipeline is phoneme-based. The `tokenization/`
module is a **stub** for a future char-vs-phoneme *ablation* against that default.

---

# The three things to never get wrong

1. **WER/CER/MCD are all lower-is-better.**
2. **HF VITS is inference-only — we wrote training from scratch.** (Your headline.)
3. **The mock `training_log.json` is not real results.** The real number is the
   pretrained baseline. And the **droid dataset voice = RVC** (a learned voice),
   while the `demo_*.wav` clips = **DSP effects** (not the model).
