# Speaker Notes — Project 13: Text to Speech

**Target: 15–20 min talk + 5–10 min discussion.** Aim ~45–60 s per content slide.
Times below are *cumulative*. Split across 3 speakers as marked.

> **Deck is now 17 slides** (trimmed from 20): the **components** table moved onto
> the *What is VITS?* slide, the **discriminator + GAN loop** share one slide, and
> the **team** slide was folded into Takeaways. The per-section numbers below are
> the *original* beats — for exact slide order use `SCRIPT.md` or the presenter
> view (press **S** in `presentation.html`), which are in sync with the 17-slide deck.

> Tip: rehearse the **bold one-liner** at the top of each note — if you're short
> on time, say just that and move on.

---

### 1 · Title *(0:00–0:20)* — *[any]*
**"Project 13, Text to Speech — we fine-tuned VITS to do end-to-end neural
speech synthesis."** Introduce the three of you. Move on fast.

### 2 · The task *(0:20–1:40)* — *[Speaker 1: Emir]*
**"The brief: build text→audio and evaluate it; pick one of three pipelines."**
Walk the table. A and B are *two-stage* (acoustic model + separate vocoder);
**C, VITS, is one end-to-end model.** Note the hard requirements: report
epochs/batch/GPU-hours, show a gallery + failure modes, evaluate with ASR→WER/CER
plus a spectral metric (MCD). Land: **"We picked Option C."**

### 3 · Our story in four decisions *(1:40–3:10)* — *[Emir]*
**"Four decisions shaped the project."** This is the roadmap for the whole talk.
(1) We *pivoted* from the two-model SpeechT5 plan to VITS. (2) We *fine-tune* a
pretrained checkpoint instead of training from scratch. (3) We built our *own
dataset* — a droid voice. (4) We *wrote the training loop from scratch* because
HF only gives inference. Tell the audience the rest of the talk expands these.

### 4 · What is VITS? *(3:10–4:40)* — *[Speaker 2: Dima]*
**"VITS is a single network from text straight to a raw waveform."** Contrast
with A/B: no separate vocoder, no intermediate mel that you predict then
re-synthesize — that's where two-stage systems lose quality. Name the four
ingredients (CVAE, flow, GAN, stochastic duration) but keep it one sentence each
— the next slides unpack them.

### 5 · Components: train vs. inference *(4:40–6:20)* — *[Dima]*
**"The key subtlety: the posterior encoder only exists at training time."**
Walk the table. In training the posterior sees the *real* audio and produces a
great latent `z`; the text-side prior is trained to *match* it via the KL term.
At inference there's no audio, so we sample `z` from the (flow-enriched) prior
and decode. This train/infer asymmetry is the heart of the CVAE.

### 6 · Alignment / MAS *(6:20–7:50)* — *[Dima]*
**"Text and audio are different lengths with no alignment labels — MAS solves
that."** Explain monotonic: speech doesn't jump back to an earlier letter.
MAS is a Viterbi-like DP that finds the best monotonic token→frame map. Stress:
**we implemented it from scratch** — vectorized forward pass + traceback, under
`no_grad` because it's a search, not a learned layer.

### 7 · Data story / droid voice *(7:50–9:20)* — *[Emir]*
**"We needed a target voice with 24 hours of paired data — so we built one."**
LJSpeech = 13,100 clips, one speaker. We resynthesized all of it and
voice-converted to a B1 Battle Droid via RVC. Why a droid? **So you can *hear*
the fine-tune work** — a subtle voice makes the demo ambiguous. Key point: RVC
changes *timbre not pronunciation*, so the original transcripts stay valid → an
instant paired dataset, which we **published on the HF Hub**.

### 8 · The core challenge *(9:20–10:40)* — *[Dima]*
**"Hugging Face ships VITS as inference-only — no training code at all."** This
is the slide to slow down on. No training forward pass, no MAS, no losses, no
discriminator. We rebuilt the whole training stack on top of HF's modules. List
the five files. Say plainly: **"this was the main engineering work."**

### 9 · forward_train *(10:40–12:00)* — *[Dima]*
**"Here's one training step."** Walk the seven points but briskly. Emphasize the
**segment slice** (point 6): the decoder is expensive, so like HiFi-GAN we train
it on a short random crop of the audio, not the whole utterance — standard VITS
trick, big speed/memory win.

### 10 · The loss — five terms *(12:00–13:10)* — *[Dima]*
**"The generator is trained on five terms at once."** Reconstruction (mel L1) is
weighted 45× — spectrum accuracy matters most. KL is the VAE term tying prior to
posterior. Duration trains the rhythm. Adversarial + feature-matching add realism
and *stabilize* the GAN. Don't read the table verbatim — point at the 45 and
explain why it dominates.

### 11 · Discriminator *(13:10–14:10)* — *[Dima]*
**"HiFi-GAN-style — two discriminators that look at audio differently."** MPD
folds the 1-D signal by prime periods to catch pitch/harmonic structure; MSD
looks at multiple time scales. Both also output feature maps, which feed the
feature-matching loss. Built from scratch.

### 12 · GAN training loop *(14:10–15:10)* — *[Dima]*
**"Standard adversarial alternation: discriminator step, then generator step."*
Two AdamW optimizers. D trains on real vs. *detached* fake (so G's graph isn't
touched). Checkpoints are resumable. Mention the budget defaults
(batch 16, lr 2e-4, segment 16384) — that's the assignment's "training budget"
ask. One line on Colab: free GPU, save checkpoints to Drive because the VM is
wiped. *(Hand back to Emir/Ilya.)*

### 13 · Evaluation *(15:10–16:30)* — *[Speaker 3: Ilya]*
**"Two metrics that measure different things."** WER/CER via Whisper =
*intelligibility*. MCD via pymcd = *spectral closeness to the target*. They're
complementary — a voice can be intelligible but timbrally off, or vice versa.
Emphasize the clean experimental design: **before/after is literally one
`--checkpoint` flag**, same prompts both times. Show the baseline row.

### 14 · Engineering *(16:30–17:40)* — *[Emir]*
**"Contract-first is why three people could build this in parallel."** Stages
talk through tiny contracts (the manifest, the EvalResult), never each other's
internals. Payoff: the A→C model swap barely touched anything downstream, and we
test the whole wiring offline in seconds via `make all` + a smoke test. Pinned
deps, ruff, loguru round it out.

### 15 · Status *(17:40–18:40)* — *[Emir]*
**"Be honest: everything is built and wired; the full GPU run is the remaining
step."** Don't oversell. The fine-tuning machinery — forward_train, MAS, five
losses, MPD+MSD, the GAN loop — is implemented (latest commit "DISCRIMINANT is
ready"). What's pending is the multi-hour training run that produces the
fine-tuned numbers and the B1 demo. *(If you've since run it, replace this with
the real table.)*

### 16 · Failure modes *(18:40–19:20)* — *[Ilya]*
**"What goes wrong with neural TTS."** Mispronunciation (espeak g2p errors — the
model already uses a phoneme tokenizer), prosody, GAN artifacts, fine-tuning risks
(LR too high, GAN instability, over-fitting timbre), and the honest one: RVC
transfers *timbre* not the ring-mod robot effect, so the fine-tuned voice is
brighter/thinner, not a hard droid (verified by synthesizing our checkpoint).
The brief explicitly asks for this — don't skip it.

### 17 · Demo *(19:20–20:00)* — *[any]*
**Play audio.** Pretrained VITS on a test prompt → then the B1 droid voice →
then a fun effect. Keep it to ~3 short clips. *(Pre-load the wavs; don't fumble
with the terminal live.)*

### 18 · Team *(20:00–20:20)* — *[any]* — optional, cut if over time.
Who did what. One sentence.

### 19 · Takeaways *(20:20–20:50)* — *[Emir]*
**"Five things to remember."** Read the five bullets as your closing summary —
this is the slide the audience will retain.

### 20 · Thank you / Q&A *(20:50+)*
Open the floor. See `DISCUSSION_QA.md` for prepared answers.

---

## Timing safety valves
- **Running long?** Cut slide 18 (team), compress 9–11 (forward_train / loss /
  discriminator) into "we implemented the training pass, five losses, and two
  HiFi-GAN discriminators."
- **Running short?** Expand the MAS slide (6) and the discriminator slide (11) —
  both have natural depth, and examiners love the from-scratch parts.
- **Hard floor:** slides 2 (task), 4 (what is VITS), 8 (the challenge),
  13 (eval), 15 (status). If you only had 7 minutes, those five tell the story.
