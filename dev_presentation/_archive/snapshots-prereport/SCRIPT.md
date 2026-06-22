# Word-for-word Script — Project 13: Text to Speech

> Verbatim 3-person script for the ~18-minute talk. Read your **[NAME]** lines.
> Hand-off lines are marked `→`. Timing is cumulative. This is also embedded as
> **presenter notes** in `presentation.html` (press **S** for speaker view).
>
> Speakers: **EMIR** (infra/data) · **DIMA** (model) · **ILYA** (evaluation).
> Pace ~130 words/min. Pause at the `//` marks. Don't rush the bold lines.

---

## Slide 1 — Title  ·  EMIR  ·  *(0:00–0:20)*

**[EMIR]** "Hi everyone. We're Dima, Emir, and Ilya, and this is **Project 13 —
Text to Speech**. // Our system turns written text into spoken audio, and we
built it by **fine-tuning a model called VITS**. Let me start with what the
assignment actually asked for."

`→ stays with Emir`

---

## Slide 2 — The task  ·  EMIR  ·  *(0:20–1:40)*

**[EMIR]** "The task was simple to state: **build a system that maps text to
audio, and evaluate it.** // We had to choose one of three pipelines.

Options A and B — FastSpeech2 and Tacotron2 — are **two-stage**: one model
predicts a spectrogram, and a second model, a vocoder, turns that into sound.

Option C is **VITS**: a **single, end-to-end model** that goes straight from text
to a waveform. //

The brief also required us to report our **training budget** — epochs, batch
size, GPU-hours — to show a few audio samples and discuss failure modes, and to
**evaluate** with an ASR model for word and character error rate, plus a spectral
metric. //

**We chose Option C — VITS.** And here's why."

`→ stays with Emir`

---

## Slide 3 — Our story in four decisions  ·  EMIR  ·  *(1:40–3:10)*

**[EMIR]** "Our project really came down to four decisions. //

**First** — we actually started on the two-model SpeechT5 plan, and we *pivoted*
to VITS. One end-to-end model instead of two, and noticeably better audio out of
the box. //

**Second** — we **fine-tune** a pretrained checkpoint instead of training from
scratch. That gets us a new voice in a few GPU-hours, and it's still real
training we can measure. //

**Third** — we built our **own dataset**: a Star Wars 'B1 Battle Droid' voice. I'll
explain why that was a smart move. //

**Fourth, and the big one** — Hugging Face only ships VITS for *inference*. So we
**wrote the entire training loop ourselves**. // Dima will walk you through the
model and that training code. Dima?"

`→ HANDS TO DIMA`

---

## Slide 4 — What is VITS? (+ components)  ·  DIMA  ·  *(3:10–6:00)*

**[DIMA]** "Thanks. So — what is VITS? // It's **one network that goes from text
straight to a raw waveform**. There's no separate vocoder, and no intermediate
spectrogram that you predict and then convert. That seam is exactly where
two-stage systems lose quality, and VITS removes it. //

Under the hood it combines four ideas: a **conditional variational autoencoder**,
which gives it a latent representation of the speech; a **normalizing flow** that
makes that representation flexible; **adversarial training**, like a GAN, for
realistic audio; and a **stochastic duration predictor** for natural rhythm. //
Now here are the pieces. // The key subtlety is this column here — **the
posterior encoder only exists during training.** //

During training, the posterior encoder gets to look at the *real* audio and
produce a high-quality latent code. The text side — the prior — is then trained
to *match* that, and that matching is the KL term in our loss. //

At inference there is no audio. So we sample the latent from the text-conditioned
prior instead, and decode that into sound. // That train-versus-inference
asymmetry is the heart of how VITS works — and it creates one hard problem."

`→ stays with Dima`

---

## Slide 5 — The alignment problem / MAS  ·  DIMA  ·  *(6:20–7:50)*

**[DIMA]** "The problem is **alignment**. // We have a handful of text tokens and
hundreds of audio frames, and **no labels** telling us which frames belong to
which letter. //

VITS solves this with **Monotonic Alignment Search**, or MAS. It's a
dynamic-programming search — like Viterbi — that finds the best **monotonic** map
from text to frames. Monotonic just means speech moves left to right; it never
jumps back to an earlier letter. //

We implemented MAS **from scratch**: a vectorized forward pass that accumulates
the best score, then a backward traceback. // The nice thing is it *guarantees* a
valid alignment by construction, so unlike attention it can't skip or babble."

`→ HANDS BACK TO EMIR`

---

## Slide 6 — The data story  ·  EMIR  ·  *(7:50–9:20)*

**[EMIR]** "So we needed data to fine-tune on. Our base is **LJSpeech** — one
English speaker, about 13,000 clips, 24 hours. //

For a target voice, we needed 24 hours of *paired* text and audio in that voice —
which you can't just record. So **we built it.** We re-synthesized all of
LJSpeech and voice-converted it into a **B1 Battle Droid** voice using a technique
called RVC. //

Why a droid? // **So you can actually *hear* the fine-tuning work.** With a subtle
voice, a demo is ambiguous. With a Star Wars droid, it's obvious. // And crucially,
the voice conversion changes the *timbre*, not the words — so the original
transcripts stay valid. That gave us an instant paired dataset, which we
**published on the Hugging Face Hub.** // Back to Dima for the part we're most
proud of."

`→ HANDS TO DIMA`

---

## Slide 7 — The core challenge  ·  DIMA  ·  *(9:20–10:40)*

**[DIMA]** "Here's the core engineering challenge. // **Hugging Face ships VITS as
inference-only.** There is no training code at all — no training forward pass, no
alignment search, no losses, no discriminator. //

So to fine-tune it, we **rebuilt the entire training stack ourselves**, on top of
Hugging Face's building blocks. // That's five pieces of code: the training
forward pass, the alignment search I just described, five loss functions, the
discriminator, and the GAN training loop. // This was the main work of the
project, so let me take you through it."

`→ stays with Dima`

---

## Slide 8 — forward_train  ·  DIMA  ·  *(10:40–12:00)*

**[DIMA]** "This is one training step. // We encode the text into a prior. The
posterior encoder turns the real audio into a latent code. The flow transforms
that code. MAS aligns text to frames, and from that we get a duration loss. //

Then there's one important trick — **step six**. The decoder, which generates the
actual waveform, is expensive. So instead of decoding the whole clip, we take a
short **random slice** — about three-quarters of a second — decode just that, and
compare it against the matching slice of real audio. // That's straight from
HiFi-GAN, and because the decoder is convolutional, a short crop trains it
perfectly well while saving a huge amount of memory."

`→ stays with Dima`

---

## Slide 9 — The five losses  ·  DIMA  ·  *(12:00–13:10)*

**[DIMA]** "All of that feeds **five loss terms**. //

The **reconstruction** loss compares the generated spectrum to the real one — and
we weight it **45 times** higher than the rest, because getting the spectrum right
matters most. // The **KL** term ties the text prior to the audio posterior —
that's the VAE part. **Duration** trains the rhythm. And the **adversarial** and
**feature-matching** terms come from the GAN — they add realism and, just as
importantly, they keep training stable."

`→ stays with Dima`

---

## Slide 10 — Discriminator + the GAN loop  ·  DIMA  ·  *(13:10–15:10)*

**[DIMA]** "Those last two losses need a **discriminator** — and we built it from
scratch, HiFi-GAN style. // Two of them, looking at the audio differently: the
**Multi-Period Discriminator** folds the audio by different periods to catch pitch
and harmonic structure, and the **Multi-Scale Discriminator** looks at three time
scales. // Together they catch artifacts either one alone would miss. //

The loop is a standard GAN alternation: each step the **discriminator** trains
first, on real versus detached fake; then the **generator** — the VITS model —
trains on all five losses to fool it. // Batch 16, learning rate 2e-4, on a Colab
GPU, resumable checkpoints. //

And this part mattered: our early runs **without** the discriminator were
unusable — adding it, plus gradient clipping, is what made the audio clean. //
Ilya will tell you how we measure it."

`→ HANDS TO ILYA`

---

## Slide 11 — Evaluation  ·  ILYA  ·  *(15:10–16:30)*

**[ILYA]** "Thanks. We evaluate with **two metrics that measure different
things**, and for both, lower is better. //

First, **word and character error rate** — *intelligibility*. We run an
off-the-shelf ASR model, Whisper, on our generated audio and check how well it can
transcribe it back to the original text. That answers: can a listener understand
it? //

Second, **MCD — Mel Cepstral Distortion** — *spectral fidelity*. It measures how
acoustically close our audio is to the real recording. That answers: does it sound
like the target? //

And our experimental design is clean: switching from the pretrained to the
fine-tuned model is a **single flag**, on the same prompts, so the before-and-after
is truly apples-to-apples. // Our pretrained baseline is **5.5% word error rate**
and an MCD of **6.4 decibels** — a solid starting point."

`→ HANDS TO EMIR`

---

## Slide 12 — Engineering  ·  EMIR  ·  *(16:30–17:30)*

**[EMIR]** "A quick word on how three people built this without stepping on each
other. // We used a **contract-first** design: the data, the model, and the
evaluation talk to each other only through small, fixed interfaces — not each
other's internals. //

That had two payoffs. We could build **in parallel**, and when we made that big
pivot from SpeechT5 to VITS, it **barely touched** the rest of the code. // We
also have an offline mock pipeline and a smoke test, so we can check the whole
thing wires together in seconds, without a GPU."

`→ stays with Emir`

---

## Slide 13 — Status  ·  EMIR  ·  *(17:30–18:20)*

**[EMIR]** "Where do we stand — honestly. // The **entire fine-tuning system is
built and wired**: the forward pass, the alignment search, all five losses, both
discriminators, the training loop. The data is published, evaluation works. //

The one thing still ahead of us is the **multi-hour GPU run** that produces the
final fine-tuned numbers and the polished droid demo. // So: the machinery is
done; the long training run is the last step."

`→ HANDS TO ILYA`

---

## Slide 14 — Failure modes  ·  ILYA  ·  *(18:20–19:00)*

**[ILYA]** "Briefly, the failure modes we watch for. // **Mispronunciation** of
rare words and names — we use character tokens, not phonemes, which is the obvious
next improvement. **Prosody** — rhythm can be flat. **Artifacts** — GANs can
produce a metallic buzz, especially early in training. // And fine-tuning itself
has risks: too high a learning rate can wreck the pretrained weights."

`→ stays with Ilya`

---

## Slide 15 — Demo  ·  ILYA  ·  *(19:00–19:50)*

**[ILYA]** "Let me play some audio. //

First, the **base VITS voice** — the pretrained starting point. // *(play
`test_0000.wav`)* //

Next, the **target voice** our dataset teaches — the B1 droid, made with RVC. //
*(play `droid_test.wav`)* //

Our **fine-tuned model** learns to map text straight to that droid voice — that's
the run we're completing. // *(IF the fine-tune is done: "and here's the fine-tuned
output" — play it as the climax.)* //

And as a bonus, some DSP robot effects. *(optional: `demo_4_excited.wav`)*"

> **Honesty guard:** the droid **target** is RVC; the **bonus** clips are
> post-processing effects, *not* the model. Don't claim the model produced the
> droid voice unless the fine-tune has actually run.

`→ HANDS TO EMIR`

---

## Slide 16 — Takeaways  ·  EMIR  ·  *(19:50–20:40)*

**[EMIR]** "To wrap up — five things. // VITS is **one end-to-end model**. We
**fine-tuned** it onto a **custom voice we built and published**, training three
epochs on an L4. We **re-implemented VITS training from scratch** on an
inference-only model — and the discriminator was what made the audio usable. Our
**contract-first** design let us build in parallel — Dima on the model and
fine-tuning, me on infra and the data pipeline, Ilya on evaluation. And we
evaluate with **error rate** for intelligibility and **MCD** for fidelity. //
Thank you — we're happy to take questions."

`→ ALL — open the floor`

---

## Slide 17 — Q&A  ·  ALL  ·  *(20:40+)*

Route by ownership: **model** questions → Dima, **evaluation** → Ilya, **data and
engineering** → Emir. Repeat each question before answering. Prepared answers are
in `DISCUSSION_QA.md`.

---

### Timing summary (17 slides)
| Speaker | Slides | Approx |
|---|---|---|
| Emir | 1, 2, 3, 6, 12, 13, 16 | ~7 min |
| Dima | 4, 5, 7, 8, 9, 10, 11 | ~8 min |
| Ilya | 14, 15 (+ demo) | ~3 min |

**Total talk ≈ 17–18 min**, comfortable inside the 15–20 min window plus the
5–10 min discussion. (The team slide was folded into takeaways; the components
table lives on the *What is VITS?* slide; the discriminator and GAN loop share
one slide.)

### If you're running long (cut these first)
- Compress slide 11 to: "five losses, and two HiFi-GAN discriminators we built
  from scratch — which were essential for usable audio."
- Trim the data-story slide (6) to the one-line "we built and published a droid
  voice with RVC."
