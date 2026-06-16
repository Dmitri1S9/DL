# Night worklog — Project 13 (TTS → B1 droid voice)

**Date:** 2026-06-16. **Goal:** finish a real fine-tune + metrics + an HTML presentation in one session.

## Project in one sentence
Take pretrained **VITS** (text→speech) and, using an **RVC B1-droid** voice-conversion model as a *teacher*, distil a new target voice (B1 Battle Droid) into VITS itself, so it natively produces droid speech from text. Then measure how well the voice transfer worked.

## Key decisions
- Droid voice is the **centerpiece**, not a garnish → **distillation** (bake teacher data, fine-tune), not runtime postprocess.
- Teacher = RVC B1 (`Homiebear/B1BattleDroid`), used **once** to build training data, not at runtime.
- Eval test set stays on **original LJSpeech audio** + same prompts → before/after is apples-to-apples.

## State of the repo at start of night (pulled origin/main @ 5d36961)
- ✅ B1 dataset built + on HF Hub: `Dmi1tr13/ljspeech-b1` (13,100 clips, 22050 Hz). Baking already done.
- ✅ VITS fine-tune code (`src/vits_finetune/`) implemented; Dima ran 2000 steps locally (commit 5d36961). README "Status" calling it a stub is **stale**.
- ✅ Inference (`model/synthesize.py`), eval (`evaluation/`, WER/CER/MCD), train/test split (`data/download.py`, last 500 held out).
- ✅ Colab runbook (`docs/training-on-colab.md`) updated for this flow.
- ⚠️ Caveat: decoder fine-tuned with L1-mel + KL + duration only (no adversarial loss) → fine-tuned audio may be buzzy/muffled. Acceptable; report it.

## Plan
1. Fine-tune on Colab on `Dmi1tr13/ljspeech-b1` → checkpoint (smoke ~50 steps, then real run).
2. Generate test set with pretrained vs finetuned.
3. Eval → WER/CER/MCD before/after table.
4. Build HTML presentation.
5. (optional) tune + rerun.

## Challenges encountered (for the presentation reflection)
- **Stale README:** Status table marked the VITS fine-tune loop as a stub, but `forward_train` + losses were actually implemented in a later commit. Had to read the code to find the true state.
- **Colab setup gotchas (env reproducibility):**
  - `colab_setup.sh` installs pip deps only; the VITS tokenizer needs the **system** package `espeak-ng` (`apt install espeak-ng`), which was missing → would crash the tokenizer.
  - `requirements.txt` pins `torch==2.11.0+cu128` to Dima's local RTX 5060. That wheel isn't available on Colab → install would break. Fix: use Colab's preinstalled torch, install the rest (`grep -vE '^(torch|pedalboard)' requirements.txt`).
- **Test-set leakage (caught & fixed).** The hold-out split lives in `data/download.py` (Emir's old LJSpeech flow), but the new B1 training flow — `vits_finetune/dataset.py` (Dima) — loads the *entire* `ljspeech-b1` (all 13,100), including the droid versions of the 500 test texts → the model was training on the eval set. Integration seam between two people's code. Fix: hold out the last 500 in the dataset (`select(range(len-500))`), so eval is honest.
- **espeak memory-mapping leak (`failed to map segment`).** The VITS tokenizer phonemizes every text via phonemizer→espeak, which `dlopen`s the espeak C lib on every call and never `dlclose`s it (ctypes). Each call leaks a memory mapping; after a few thousand calls the process hits the default `vm.max_map_count` (~65k) and crashes. First blamed the DataLoader workers and set `num_workers=0` — but it crashed *sooner* in a single process (fewer processes to spread the leak across), which confirmed the real cause. Fix: `sysctl -w vm.max_map_count=2000000`. (Diagnostic lesson: a crash that gets *worse* with fewer processes points to a per-process resource leak, not a concurrency bug.)

## Log
| step | what happened |
|---|---|
| recon | Pulled latest origin/main (@5d36961). Assessed state (above): dataset + training code already done; only the run + metrics + presentation remain. |
| step 1 (started) | Colab Free + T4. Cells: clone → install (espeak-ng + torch workaround) → lowered checkpoint_every to 250 → launched `vits_finetune.train --batch-size 4 --num-epochs 1`. Awaiting first loss logs. Built `colab_train.ipynb` (upload-to-Colab notebook with the fixes baked in). |
| step 1 (training confirmed) | Setup OK: GPU=T4, base VITS (145M) + B1 dataset (8 parquet shards, ~3.7GB, 13,100 ex) downloaded. Training is learning: total_loss 976→465 over steps 0→50 (recon 0.86→0.55, duration 827→371), no nan/inf. Speed ~0.76 s/step. **1 epoch = 3,275 steps ≈ ~40 min.** Checkpoints every 250 steps (step_0.pt saved). Note: `words count mismatch` spam = harmless phonemizer/espeak noise; DataLoader uses 4 workers vs 2 cores (warning only, didn't freeze). |
| step 1 (loss curve) | Loss still dropping: step 250 → total 410.4 (recon 0.456, kl 68.1, dur 321.8). Curve flattening (normal). `step_250.pt` saved → backing up to Google Drive. ETA full epoch ~35 more min. |
| step 1 (step 500) | total 282.9 (recon 0.392, kl 67.9, dur 197.4). Healthy descent: 976→465→410→283. ~0.78 s/step; 500/3275 (~15%). `step_500.pt` saved (~35s/save overhead). ETA finish ~21:35–21:40. |
| step 1 (restart on Pro) | Caught test leakage (above) → stopping the T4 run. Upgrading to Colab Pro + L4 GPU. Updated `colab_train.ipynb`: hold-out last 500, silenced log spam, checkpoint_every=250, num_workers=2, batch 16. Fresh run (no resume — data changed). 1 epoch now ≈ 788 steps (~15–20 min). |
| step 1 (L4 run, clean) | Log now clean (message-filter silenced `words count mismatch` — first attempt via logger-name setLevel didn't catch it). Healthy: step 0 total 877 (recon 0.83) → step 50 total 330 (recon 0.43). Speed ~0.68 s/step at batch 16 (L4 ~4× faster per clip than T4). Epoch ≈788 steps ≈ ~10 min. |
| step 1 (crash @ step 200) | espeak crashed in a DataLoader worker (`failed to map segment`). Loss was healthy up to step 200 (recon 0.37) — environment issue, not the model. Only `step_0.pt` saved → restart needed. Fix attempt: `num_workers=0`, re-run train (VM intact, no re-download). |
| step 1 (crash again, real fix) | With `num_workers=0` it crashed even sooner (~step 100) — same espeak `failed to map segment`. Real cause: per-call espeak `dlopen` leaks a memory mapping → hits `vm.max_map_count`. Fix: `sysctl -w vm.max_map_count=2000000`, then re-run train. Fallback if sysctl blocked: train on a ~2k-clip subset. |
| step 1 (Plan B: subset) | `sysctl` blocked on Colab (`Read-only file system`, limit stuck at 65530). Plan B = train on a 2,500-clip subset (first 2,500, test still held out) + `num_workers=2` (split the leak across 2 processes, ~1250 calls each, under the ~1600 crash threshold). ~156 steps, ~3 min. Legit fine-tune; matches the original "small-data" idea. (Full dataset would need a deeper fix: make espeak load once instead of per call.) |
| ✅ TRAINING DONE | Full epoch completed, no crash. `epoch_1.pt` saved. Loss: step 0 total 902 (recon 0.84, dur 754) → step 150 total 121 (recon 0.38, dur 35). Trained on 2,500-clip subset, 1 epoch, batch 16, L4. Next: save to Drive → listen (pretrained vs finetuned) → generate test set + WER/CER/MCD before/after. |
| ⚠️ finetuned inference = garbage | Listened: pretrained demo (same `vits_finetune.synthesize` code, no checkpoint) sounds normal; finetuned `epoch_1.pt` = noise. So codepath + checkpoint load are fine — the fine-tune broke generation. |
| 🔑 real diagnosis: too few steps | Emir: Dima already got good results — so the code is fine. Real difference: Dima trained **2000 steps** (commit "test after 2000 steps"); our run was only **156 steps** (2500÷16). VITS's inference path (stochastic duration predictor + prior) converges slowly — 156 steps is far too few → garbage; ~2000 converges it. (Earlier "LR too high" diagnosis was wrong.) Fix: mimic Dima — batch 2, default LR 2e-4, ~2000 steps. To dodge the espeak leak: 4000-clip subset (test still held out) ÷ 4 workers ≈ 1000 calls/worker, 1 epoch = 2000 steps. `droid_test.wav` in repo is likely Dima's good output. |
| 🤖 autonomous Colab run (browser) | Ran the clean `finetune`-branch notebook end-to-end on Colab **T4** (L4 unavailable at the time). 4000-clip subset, batch 2, default LR, ~2000-step target. Passed step 200 fine (espeak workaround held) — but the run **hung at ~step 1050**, right after the step_1000 checkpoint save (espeak memory-map accumulation across the 4 workers on the 2nd half). Interrupt didn't take (native-level hang) → Restart session (VM files persist, `step_1000.pt` survived). |
| 📊 OBJECTIVE RESULT (1000 steps, Whisper) | On the held-out prompt: **pretrained WER = 10%** ("roger roger, all units, proceed with the mission. standing by" — intelligible) vs **fine-tuned-1000 WER = 90%** ("boderaer or i'll get it. with magic, hey, it goes!" — garbage). At 1000 steps the droid fine-tune is **not yet intelligible**; `recon_loss` plateaued ~0.4 the whole run. Whisper = objective check (no human ears needed). |
| ↻ resume → 2000 steps | Resumed from `step_1000.pt` in a fresh process (fresh espeak budget) → reached step 2000, saved `step_2000.pt` (then hung again ~step 2050, same espeak limit; killed via Restart session). Total = 2000 steps (Dima's setting). |
| 📊 OBJECTIVE RESULT (2000 steps, Whisper) | **pretrained WER = 20%** ("...proceed with the mission standing by") vs **fine-tuned-2000 WER = 120%** ("sorry, i owe ya everfield if thy he monday get spien i" — garbage, >100% = insertions). **2000 steps did NOT fix it** — still unintelligible, no better than 1000. |
| ✅ CONCLUSION | The droid fine-tune does **not** produce intelligible speech with this setup, at either 1000 or 2000 steps. Training loss plateaued from ~step 100 (recon ~0.4, KL ~68 never converged) → the VITS *generation* path (flow/decoder + stochastic duration) isn't being trained to match. Likely cause: the simplified objective (recon + KL + duration) **lacks the adversarial/discriminator loss** that real VITS uses to train the decoder — so reconstruction looks fine but inference produces noise. A complete, objective negative result + clear next step (add the GAN loss / use an established VITS fine-tune recipe; verify what Dima's `droid_test.wav` actually came from). |

## Contributions & project evolution (from git history, 26 commits)

| Member | Commits | Area |
|---|---|---|
| Dima (Tsygankov Dmitrii / Dmitrii Ts) | 14 | model, RVC voice conversion, VITS migration, B1 dataset, fine-tuning |
| Emir (Wiped-Out) | 12 | infra, restructure/tooling, data + train/test split, evaluation (MCD/WER), Colab runbook, README |
| Ilya | 0 | — (no commits in history; evaluation code was committed by Emir — verify before the contributions slide) |

**Timeline / arc:**
1. **30 May (Dima)** — seed: initial TTS prototype with droid-voice + Russian-accent effects (the creative spark).
2. **4 Jun (Emir, ~11 commits)** — engineering foundation: `src/`+`tests/` restructure, contract-first pipeline (DTOs/mocks/Makefile), ruff+loguru, LJSpeech download + train/test split, **evaluation (pymcd MCD, WER/CER)**, Colab runbook + pinned deps, README. Made it a reproducible, gradeable DL project.
3. **4 Jun (Dima)** — modeling & droid pivot: RVC B1 voice conversion (teacher), SpeechT5→VITS migration (Option A→C), augmentation.
4. **13–14 Jun (Dima)** — data + fine-tuning: built & published `Dmi1tr13/ljspeech-b1`, data scripts, real VITS fine-tune loop, first 2000-step run.
5. **16 Jun (Emir, tonight)** — end-to-end fine-tune on Colab, before/after metrics, HTML presentation.

One-line arc: *Dima sparked the idea → Emir built the engineering base → Dima brought up the model + data → Emir drives it to a measured result and packages it.*
