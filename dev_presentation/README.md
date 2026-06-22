# Project 13 — Text to Speech · Presentation

Slide deck + speaker notes for the group presentation of the DL TTS project
(fine-tuning **VITS**). Built for a **15–20 min talk + 5–10 min discussion**.

Source repo being presented: `~/dev/dl` · github.com/Dmitri1S9/DL
Full background brief: `~/dev/dl-presentation-brief.md`

## Files

### ★ The deck — present from this
| File | What it is |
|---|---|
| **`presentation.html`** | **THE deck** — reveal.js, Academic Clean theme, fragment builds + **presenter view**. Open in any browser (works offline). |
| `presentation.pdf` | Static PDF export of the reveal deck (one page per slide). |
| `SCRIPT.md` | **Word-for-word 3-person script** — exactly what each person says, with hand-offs + timing. Also embedded as presenter notes inside `presentation.html`. |
| `reveal/` | Vendored reveal.js (so the HTML needs no internet). Don't edit. |

### Prep & reference docs
| File | What it is |
|---|---|
| `CHEATSHEET.md` | One-page quick-reference — **print this and hold it** while presenting. |
| `SPEAKER_NOTES.md` | Per-slide talk track (bullet cues) with **timing** + speaker split. |
| `PRESENTATION_PLAYBOOK.md` | How to *deliver* it — rehearsal, demo plan, examiner tips, rubric. |
| `DISCUSSION_QA.md` | ~50 anticipated Q&A answers, by difficulty tier. |
| `TECHNICAL_APPENDIX.md` | Deep dive — VITS math/ELBO, MAS, HiFi-GAN, losses, references. |
| `pdf/` | PDF versions of every doc above (for sharing / Telegram). |

### Alternate deck (Marp)
| File | What it is |
|---|---|
| `slides.pptx` | **PowerPoint/Keynote** version — use if someone needs to edit in PowerPoint. |
| `slides.pdf` / `slides.html` / `slides.md` | Earlier Marp render of the same content. `presentation.html` supersedes these. |

> Master background brief (source of truth behind everything): `~/dev/dl-presentation-brief.md`.

**Suggested reading order:** `SCRIPT` → `CHEATSHEET` → `PRESENTATION_PLAYBOOK`
→ `DISCUSSION_QA` → `TECHNICAL_APPENDIX` (only if you want the math).

## Presenting (reveal.js)

1. **Open `presentation.html`** in any browser (Chrome/Safari/Firefox). It's
   self-contained — no internet needed.
2. Keys: **`F`** = fullscreen · **`→` / `Space`** = next (also advances fragment
   builds) · **`←`** = back · **`S`** = **speaker view** (your script + next-slide
   preview + timer on your laptop, slides on the projector) · **`Esc`** = slide
   overview · **`B`** = blank screen.
3. Read `SCRIPT.md` to rehearse; it splits all 20 slides across the three
   speakers with hand-off lines and cumulative timing (~18 min talk).
4. **Demo (slide 17):** pre-load the audio so you don't fumble live —
   `dl/audio/generated/test_0000.wav`, `dl/droid_test.wav`,
   `dl/audio/demo_4_excited.wav`. Test room speakers first.

## Editing / re-rendering

- **Edit the reveal deck:** just edit `presentation.html` (plain HTML/CSS) and
  refresh the browser. To regenerate the PDF:
  ```bash
  cd ~/dev/dev_presentation
  export PUPPETEER_EXECUTABLE_PATH="$(ls -d ~/.cache/puppeteer/chrome/*/chrome-mac-arm64/*.app/Contents/MacOS/* | head -1)"
  npx -y decktape@latest reveal --size 1280x720 "file://$PWD/presentation.html" presentation.pdf
  ```
- **Re-render the docs to PDF** (after editing any `.md`):
  ```bash
  export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
  pandoc SCRIPT.md -o /tmp/x.html -s --metadata title=" " && weasyprint -s doc.css /tmp/x.html pdf/SCRIPT.pdf
  ```
- **Marp alternate deck:** edit `slides.md`, then
  `npx -y @marp-team/marp-cli@latest slides.md -o slides.pptx` (and `-o slides.pdf --pdf`).

## Deck outline (20 slides)

1. Title · 2. The task · 3. Four decisions · 4. What is VITS · 5. Components
(train vs. infer) · 6. Alignment / MAS · 7. Data story (B1 droid voice) ·
8. The core challenge (HF is inference-only) · 9. `forward_train` · 10. Five
losses · 11. Discriminator (MPD+MSD) · 12. GAN training loop · 13. Evaluation ·
14. Engineering (contract-first) · 15. Status · 16. Failure modes · 17. Demo ·
18. Team · 19. Takeaways · 20. Q&A.
