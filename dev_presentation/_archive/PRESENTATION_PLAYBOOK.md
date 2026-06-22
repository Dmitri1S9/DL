# Presentation Playbook — how to actually deliver this

> Everything about *giving* the talk: timing, who-says-what, rehearsal, the demo,
> handling questions, examiner psychology, and what gets you marks. Pair with
> `SPEAKER_NOTES.md` (per-slide script) and `DISCUSSION_QA.md` (answers).

---

## 1. The format, decoded

- **15–20 min talk + 5–10 min discussion.** Plan the talk for **18 min** — it
  leaves slack and you won't get cut off. Over-running looks unprepared.
- **20 slides** → ~50 s/slide average. Some (title, team, demo) are 15–20 s;
  the technical ones (8–12) are 60–90 s. That balances out to ~18 min.
- **Three speakers** — examiners expect everyone to talk roughly equally. The
  split below gives each person ~6 min + their own Q&A territory.

## 2. Speaker split (rehearse these boundaries)

| Speaker | Owns | Slides | ~min |
|---|---|---|---|
| **Emir** (infra/data) | task, story, data, engineering, status, takeaways | 2, 3, 7, 14, 15, 19 | ~6 |
| **Dima** (model) | VITS, components, MAS, the challenge, forward_train, losses, discriminator, GAN loop | 4, 5, 6, 8, 9, 10, 11, 12 | ~8 |
| **Ilya** (eval) | evaluation, failure modes, (co-drive demo) | 13, 16, 17 | ~4 |

Title (1), team (18), thank-you (20) — whoever. **Hand-offs out loud:**
"…and Dima will take the model." A clean verbal baton looks rehearsed.

> Dima carries the technical core — that's by design, it's the meat. If you want
> a more even split, give Dima's slides 5 (components) and 6 (MAS) to whoever is
> most comfortable with the VAE/alignment story.

## 3. The narrative arc (so it's a story, not a file dump)

1. **Hook** — "TTS usually needs two models; we used one, end-to-end: VITS." (2–4)
2. **Twist** — "and we built our own voice dataset so you can *hear* it work." (7)
3. **The hard part** — "Hugging Face only ships VITS for inference, so we wrote
   the entire training loop ourselves." (8–12) ← *this is your differentiator*
4. **Proof** — "here's how we measure it, and here's the baseline." (13)
5. **Honesty** — "everything's built; the GPU run is the last step." (15)
6. **Payoff** — play the audio. (17)

Every slide should advance this arc. If a slide doesn't, cut it.

## 4. Rehearsal checklist (do this twice, out loud, timed)

- [ ] Full run-through with a timer; note the minute mark at slide 13 — should be
      ~16:00. If you hit it at 18:00, you're too slow; trim slides 9–11.
- [ ] Each speaker can give **their** slides without reading them.
- [ ] Hand-offs are verbal and smooth.
- [ ] **Demo dry-run on the presentation machine** (see §6) — audio actually
      plays through the room speakers.
- [ ] One person drives slides the whole time (the others don't fight for the
      clicker).
- [ ] Decide who fields which Q&A area (model Qs → Dima, eval Qs → Ilya, data/eng
      → Emir). Say so beforehand so you don't talk over each other.
- [ ] Have `slides.pdf` AND `slides.pptx` AND `slides.html` on a USB stick / in
      cloud — projectors are hostile.

## 5. Per-slide "what they'll fixate on"

Quick map of where questions cluster, so you pre-load the answer:

| Slide | Likely question | One-line defense |
|---|---|---|
| 3 (pivot) | "Why abandon SpeechT5?" | One model > two; better audio out of the box. |
| 7 (data) | "Synthetic data — isn't that circular?" | Demonstrates *adaptation*; real recordings would slot into the same pipeline. |
| 8 (challenge) | "What exactly did you implement?" | forward_train, MAS, 5 losses, MPD+MSD, GAN loop. |
| 9 (forward_train) | "Why segment-slice?" | Decoder is expensive; HiFi-GAN trick; convolutional so a crop is fine. |
| 10 (losses) | "Why 45× on mel?" | Spectral accuracy is primary; GAN terms are polish. |
| 13 (eval) | "Why Whisper isn't circular?" | Independent ASR; its error floor is constant across before/after. |
| 15 (status) | "So it doesn't work yet?" | Machinery is done + wired; only the multi-hour run remains. |

Full answers: `DISCUSSION_QA.md`.

## 6. The demo — don't let it kill you

Live demos fail in front of examiners. **Pre-render everything:**

- Have the wavs ready in a folder and a simple player open:
  `dl/audio/generated/test_0000.wav` (pretrained), `dl/droid_test.wav` (B1
  voice), `dl/audio/demo_4_excited.wav` (a fun effect).
- Order: **(1) pretrained VITS** → **(2) B1 droid voice** → **(3) one effect.**
  Three clips, ~10 s each. That's the whole point made audible.
- **Backup:** if audio fails, the slide text still tells the story — say
  "you'll hear the droid voice in the recording we'll share." Never spend >20 s
  fighting hardware.
- **Test the room's audio output before you start.** Mac volume up, correct
  output device, file associations work.

## 7. Handling questions (the 5–10 min discussion)

- **Repeat/rephrase the question** before answering — buys thinking time, makes
  sure you understood, and the room hears it.
- **Answer in one sentence, then offer depth.** "Yes — because X. Want the
  detail?" Examiners reward crisp answers over rambling.
- **If you don't know:** "We didn't measure that, but I'd expect …, because …."
  Reasoning from principles beats bluffing. Never invent a number.
- **Route by ownership** — the model person takes model questions, etc. But don't
  leave a teammate hanging; jump in if they stall.
- **The mock results trap.** If asked about `training_log.json`, say plainly:
  "that's a placeholder from the old SpeechT5 plan — not real results. Our real
  baseline is the pretrained VITS row: WER 5.5%."
- **Turn it around (good)**: "We considered phonemes here — is that the direction
  you'd push?" Shows you know the trade-space.

## 8. What earns marks (rubric alignment)

The brief asks for specific things — make sure each is *visibly* covered:

| Brief requirement | Where you show it | Say the words |
|---|---|---|
| Chosen pipeline justified | slides 2–4 | "Option C, VITS, end-to-end." |
| Trained on held-out test set | slides 12–13 | "held-out split / test manifest." |
| **Training budget** (epochs/batch/GPU-h) | slide 12 | "batch 16, lr 2e-4, N epochs, ~X GPU-hours." |
| Qualitative gallery | slide 17 (demo) | play the clips. |
| **Failure modes** | slide 16 | "mispronunciation, prosody, artifacts." |
| ASR → **WER/CER** | slide 13 | "Whisper, WER 5.5% / CER 1.6%." |
| Spectral metric (**MCD**) | slide 13 | "MCD 6.38 dB." |
| Optional: tokenizer study | slide 16 / Q&A | "char now; phonemes the obvious extension." |

If you tick every row out loud, you've covered the assignment explicitly — that's
the difference between a good and a top mark.

## 9. Slide-design sanity (already done, keep it)

- One idea per slide, ≤6 bullets, big font (the deck is 26 px body / 46 px H1).
- Tables and the pipeline diagram do the heavy lifting — point at them, don't
  read them.
- Don't add a wall of equations to the slides — keep the math in
  `TECHNICAL_APPENDIX.md` and pull it out *only* if asked.

## 10. Backup / "deep-dive" slides to have ready (optional)

If you expect a hard technical panel, keep 2–3 hidden slides after the thank-you
(or just know them from the appendix):

- **MAS in one diagram** — the DP grid + monotonic path.
- **The ELBO** — `log p(x) ≥ E[log p(x|z)] − KL(q‖p)` mapped to your losses.
- **MPD reshape** — 1-D audio → 2-D by period, why primes.

These never appear unless asked; they signal depth when they do.

## 11. Logistics morning-of

- [ ] Laptop charged + charger; HDMI/USB-C adapter.
- [ ] `slides.pdf` open and tested on the projector resolution (16:9).
- [ ] Audio tested through room speakers.
- [ ] Phones on silent; one person owns the clicker.
- [ ] Water. Breathe. You built a from-scratch VITS trainer — own it.

## 12. The 30-second version (if you're cut to almost nothing)

"We built end-to-end TTS by fine-tuning VITS onto a custom voice we created and
published. Hugging Face only ships VITS for inference, so we re-implemented the
whole training stack ourselves — alignment search, five losses, the HiFi-GAN
discriminators, the GAN loop. We evaluate with Whisper-based WER/CER for
intelligibility and MCD for spectral fidelity. The pipeline is built end-to-end;
the final fine-tuning run is the last step."
