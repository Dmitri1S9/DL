# Project 13 — Speaking Script

**Target: ~19 min.** Casual, student tone. Say it, don't read it.
Roles: **Emir** = intro + architecture + model internals + wrap-up · **Dima** = ideas, data, training, losses · **Ilya** = evaluation, demo, limitations.

Handoffs (say the next person's name so it's smooth): Emir → **Dima** → Emir → **Dima** → **Ilya** → Emir.

> ⏱️ **Reality check:** good run ≈ **18 min**, fits 20 with buffer. All the variance is Dima (slower English). Two things blow the budget — guard them:
> 1. **Loss slides (17–23) = ONE montage**, ~30s total, two sentences for all seven. Do *not* explain each.
> 2. **Slide 15 (3 bugs)** — say the short version on the slide, don't re-derive. Title / Thank-you ~15s each.

---

## EMIR — opening (slides 0–2) · ~2 min

**Slide 0 · Title**
> Hi — we're Dima, Emir and Ilya. We took a model that already talks perfectly fine… and taught it to talk like a Star Wars battle droid. On purpose. *(beat)* This is Project 13. Let me tell you why.

**Slide 1 · The goal**
> A pretrained TTS model already produces clean speech, so "make good speech" isn't a project — it already does that. The actual goal is to **change the voice**. We fine-tune it into a B1 battle droid. And we picked a droid on purpose: a subtle voice makes a demo ambiguous, but a droid? You'll hear it instantly — you don't even need the metrics to know it worked. *(beat)* Although we did the metrics anyway, because it's a deep learning course.

**Slide 2 · Project architecture**
> Quick word on how three people work on one model without killing each other. I set the repo up **contract-first**: data, model and evaluation are separate packages that only talk through small fixed interfaces — a manifest and one result object — never each other's internals. So the model stays swappable and we could all build in parallel. One Makefile runs the whole thing, same commands locally and on Colab, and there's an offline mock + smoke test so the pipeline wires up in seconds with no GPU. With that skeleton, Dima takes the model.

---

## DIMA — the idea & the data (slides 3–7) · ~2.5 min
*(short sentences, simple words)*

**Slide 3 · Three ways to change a TTS voice**
> Thanks. There are three ways to change a voice. **Pre-process** the input — good for an accent, but limited. **Post-process** the audio — I tried it, it sounds fake. Or **bake** the voice into the model by fine-tuning. That is the real way, so that is what I did.

**Slide 4 · Where to start: the ideas**
> I had four ideas: droid voice, accent, emotions, stuttering. Emotions and stuttering were too shallow, so I dropped them. I kept the **droid**, and left the accent as future work.

**Slide 5 · Our choice: bake in a droid voice**
> So: bake a droid voice into the model. One consistent voice, you can hear it without metrics, and it lets me play with the training itself. That's the fun part.

**Slide 6 · Building the dataset — RVC**
> But I need data — a lot of paired text and audio in the droid voice. You can't record 24 hours of a droid. *(beat)* I checked. So I built it: I took all of LJSpeech and ran it through **RVC**, voice conversion. It changes the timbre to the droid but keeps the words, so the transcripts still match. 13,100 clips, for free. Published on Hugging Face — QR's on the slide.

**Slide 7 · The trade-off I chose**
> One honest trade-off. The data comes from RVC, so the model learns from RVC's output — it can't beat that teacher. But in return, the model makes the droid voice **itself**, end to end. No second model in production. A bit less ceiling, full autonomy. And Ilya actually measured where that ceiling is — you'll see. Back to Emir for the model.

---

## EMIR — the model (slides 8–11) · ~3 min

**Slide 8 · Why VITS**
> Why VITS. Three reasons. It's **end-to-end** — text straight to waveform, one model, no separate vocoder to keep in sync. It gives **phoneme-level access**, which we wanted for the accent idea. And there's a strong pretrained checkpoint to start from, `vits-ljs`. Under the hood it's a VAE with a normalizing flow, a GAN decoder, and a stochastic duration predictor.

**Slide 9 · The generator: why it isn't enough**
> Here's the generator. In training it sees both the text and the real audio, and rebuilds the waveform, comparing on the mel-spectrogram. The problem: train on that reconstruction loss alone, and the audio comes out blurry and muffled. L1 just can't capture fine detail. That's the gap a discriminator fills. One thing to stress — the real audio is **only** used in training; at inference there's no audio, it's purely text to speech.

**Slide 10 · Discriminator: built from scratch**
> And here's the catch: Hugging Face ships VITS inference-only — no discriminator. So we wrote one from scratch, HiFi-GAN style. A multi-period discriminator that folds the waveform by prime periods to catch pitch and harmonics, and a multi-scale one over three time scales. They also expose their feature maps, which gives a feature-matching loss. Without this, the decoder never learns realistic audio.

**Slide 11 · Lining up text & audio: MAS**
> Last model piece: alignment. We have maybe ten text tokens but hundreds of audio frames, and nothing says which frame is which token. **Monotonic Alignment Search** finds the best monotonic mapping — speech moves left to right, can't skip or repeat, so it can't babble. Bonus: it also tells us how long each token should last, which is the target for the duration loss. Dima — how'd training go?

---

## DIMA — training (slides 12–23) · ~4.5 min
*(short sentences; slides 17–23 are a fast montage)*

**Slide 12 · How training works**
> One training step: encode text, encode the real audio, align them with MAS, decode a short segment to a waveform, compute the losses. Then a standard GAN: discriminator on real versus fake, then generator on five losses. Reconstruction is weighted 45. The messy plumbing — two optimizers, detach, clipping, mixed precision — I put in decorators, so the step itself reads clean.

**Slide 13 · Abstraction**
> Quick thing I'm proud of. The GAN loop is messy. I moved all the wiring into **decorators**, and left only the real step logic in the methods. Read the method — it's linear. Need the plumbing — read the decorator. Same idea as `torch.nn.Module` hiding backward. The complexity doesn't vanish, it just goes where it belongs.

**Slide 14 · Where & how we trained**
> Compute. Colab first — it dropped my session after 15 minutes. Very generous. *(beat)* So I moved to my local RTX 5060, 8 gigs. To fit 8 gigs: batch size 1 with gradient accumulation, fp16, and zero dataloader workers because Windows kept crashing.

**Slide 15 · Problems & fixes: nothing converged**
> At first — nothing worked. Three bugs. *(point at table)* Duration loss: a hundred times too big — I fixed the averaging, three hundred down to two. Grad clipping: too tight, it killed the generator — I loosened it. And a fresh discriminator was wrecking the pretrained decoder — so I warmed it up first. After that: real training.

**Slide 16 · Training setup & cost**
> The numbers: 5 epochs, about 1.9 hours each, ~9.7 GPU-hours for the clean run — plus maybe three times that in failed runs, plus baking the whole dataset with RVC. Config's on the slide: learning rate 1e-4, segment 8192, grad-clip 10, disc warm-up 1500.

**Slides 17–23 · Loss curves** ⏩ *montage — click fast*
> And here are all my training losses. Short version: everything's stable. *(click)* Reconstruction drops then flattens — the voice is learned early. *(click, click)* Duration and KL — stable. *(click ×4)* The three GAN losses and the total — stable, nothing exploded, the discriminator I built does its job. *(land on last)* But a nice loss curve does **not** prove the speech is actually understandable. That's Ilya's job.

---

## ILYA — evaluation & demo (slides 24–28) · ~4 min

**Slide 24 · Evaluation: what we measure & why**
> Thanks. We use three numbers, all lower-is-better. **WER and CER** from Whisper — intelligibility, can a listener understand it. **MCD** — spectral closeness to the droid reference, the key one: did it learn the timbre. And the **ceiling** — and this is the important bit — we ran Whisper on the *real* B1 recordings, and even those score 5.75% WER. So no model trained on this data can beat 5.75. The B1 twist: we measure MCD against the **droid** reference, not the original female voice — otherwise we'd just be measuring how *un*-droid-like we are.

**Slide 25 · How we evaluate**
> Quickly, how: we hold out the last 5% the model never saw, before-and-after is one checkpoint swap on the same prompts, and the ASR is Whisper — frozen, off-the-shelf — so there's no circularity, it never touched our training.

**Slide 26 · Results: vs the ceiling**
> The payoff. **MCD drops from 10.5 to 6.0** — hard proof the fine-tuning worked, the voice moved to the droid timbre. And intelligibility lands close to the ceiling: our best checkpoint, epoch 2, is **14.4% WER** versus the **5.75%** of the real recordings. It can't beat its teacher — but it gets close. So: we gained the new voice, at a small, *measured* cost.

**Slide 27 · Demo**
> Okay — the part you actually came for. Same sentence, before… *(play base)* …and after. *(play fine-tuned)* Same words. Different voice entirely. *(beat)* …Roger, roger.

**Slide 28 · Failure modes & limitations**
> Honest limitations. Intelligibility sits above the ceiling — understandable, but less clean than the base. It's not a *hard* sci-fi robot — RVC moves a speaker's timbre, not the ring-mod buzz; that'd be a separate effect. GAN artifacts early, and it over-fits if you train past the best epoch — which is why we don't take the last checkpoint. And the Russian accent is still future work.

---

## EMIR — wrap-up (slides 29–30) · ~1 min

**Slide 29 · Takeaways & future work**
> To wrap up: we built and published our own droid dataset with RVC, fine-tuned VITS onto it, and re-implemented the whole GAN training stack from scratch — discriminator, losses, the loop — because the library ships inference-only. The result is a clear, measured trade-off: timbre moved to the droid, at a small intelligibility cost, bounded by the data's own ceiling. Next step to close that gap: freeze the text and timing front-end during fine-tuning. Plus the accent.

**Slide 30 · Thank you**
> That's us — Dima on the model, Ilya on evaluation, me on data and infra. Thanks for listening — questions? *(route by ownership: model → Dima, eval → Ilya, data/infra → Emir)*

---

### Q&A quick-fire (route to owner, repeat the question first)
- **"Why is WER higher than the base?"** → trade-off; the ceiling is 5.75% and we're near it; freezing the front-end is the fix. *(Ilya)*
- **"Can you reproduce the numbers?"** → yes, `make eval`; full held-out 5% (655 clips), Whisper frozen. *(Ilya)*
- **"Is it really a droid or just a different person?"** → RVC transfers a speaker timbre; the hard mechanical buzz is a separate DSP step, out of scope. *(Dima)*
- **"Why VITS over Tacotron/SpeechT5?"** → end-to-end, one model, phoneme access, strong checkpoint. *(Emir)*
- **"How did three people not collide?"** → contract-first, mocks + smoke test. *(Emir)*
