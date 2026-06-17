# VITS fine-tune: достройка до настоящего GAN-обучения

**Дата:** 2026-06-17
**Статус:** утверждён, готов к плану реализации

## Проблема

Собственный цикл обучения в `src/vits_finetune/` тренирует декодер только по
реконструкционному mel-лоссу (L1) + KL + duration. Не хватает **adversarial**
части (дискриминатор + generator + feature-matching), без которой VITS-декодер
не может научиться выдавать реалистичный waveform — получается шум
(Whisper WER 90–120%). `forward_train` уже реализован полноценно
(text encoder → posterior → flow → MAS → duration → нарезка сегмента →
декодер → mel). Не хватает именно GAN-механизма.

## Цель

Достроить **свой** код (без чужого рецепта ylacombe, без обновления библиотек)
до полного объектива VITS: добавить дискриминатор HiFi-GAN, adversarial и
feature-matching лоссы, переписать цикл в GAN с двумя оптимизаторами. Локально —
smoke-тест на CPU; реальное обучение — на Colab GPU.

## Не входит в объём (YAGNI)

- Чужой рецепт `ylacombe/finetune-hf-vits` и `colab_finetune_proper.ipynb`
  (ноутбук удалён).
- Обновление `transformers`/`accelerate`.
- Multi-GPU, mixed precision (AMP), EMA весов.

## Архитектура изменений

### 1. `src/vits_finetune/discriminator.py` (новый)

Дискриминатор HiFi-GAN, реализованный с нуля (стандартная, документированная
архитектура):

- `MultiPeriodDiscriminator` — sub-дискриминаторы для периодов [2, 3, 5, 7, 11];
  каждый решейпит 1D-волну в 2D по периоду и прогоняет стек Conv2d (weight_norm).
- `MultiScaleDiscriminator` — 3 масштаба (исходный + 2 усреднённых через
  AvgPool1d); стек Conv1d, первый sub-дискриминатор со spectral_norm, остальные
  weight_norm.
- Объединяющий `VitsDiscriminator(nn.Module)`:
  `forward(real_wav, fake_wav) -> (d_real_outputs, d_fake_outputs,
  fmaps_real, fmaps_fake)`, где outputs — списки логитов по sub-дискриминаторам,
  fmaps — списки промежуточных карт признаков.

Вход — waveform `(B, 1, T_samples)` (как уже отдаёт `collate_fn`).

### 2. `src/vits_finetune/losses.py`

Сохраняем `recon_loss`, `kl_loss`. Добавляем:

- `discriminator_loss(d_real_outputs, d_fake_outputs) -> Tensor` — LS-GAN:
  `mean((1 - d_real)^2) + mean(d_fake^2)` по всем sub-дискриминаторам.
- `generator_adv_loss(d_fake_outputs) -> Tensor` — `mean((1 - d_fake)^2)`.
- `feature_matching_loss(fmaps_real, fmaps_fake) -> Tensor` — L1 между картами
  признаков (с `.detach()` по real-картам).

### 3. `src/vits_finetune/model.py`

`forward_train` дополнительно возвращает:

- `predicted_waveform` `(B, 1, segment_frames * hop_length)` — уже вычисляется.
- `target_waveform` `(B, 1, segment_frames * hop_length)` — реальный кусок
  `batch['waveform']`, вырезанный по тем же `starts` (в сэмплах:
  `start * hop_length`, длина `segment_frames * hop_length`).

Это даёт дискриминатору пару «настоящий/поддельный» с одной позиции.

### 4. `src/vits_finetune/train.py`

Переписать цикл в GAN, убрав декораторы `back_step_dec` / `stepof5GOAT` /
`pretrain_check` в пользу явных методов:

- Два оптимизатора: `optim_g` (параметры VITS-генератора), `optim_d` (параметры
  дискриминатора). Оба `AdamW`, `betas=(0.8, 0.99)`.
- Один шаг:
  1. `outputs = model.forward_train(batch)`.
  2. **Шаг дискриминатора:** `D(target_waveform, predicted_waveform.detach())`
     → `discriminator_loss` → `optim_d`.
  3. **Шаг генератора:** `D(target_waveform, predicted_waveform)` →
     `gen_adv + fm + mel·45 + kl + duration` → `optim_g`.
- Логирование, чекпоинты, resume сохранить (с учётом п. 6).

### 5. `src/vits_finetune/config.py`

Добавить веса/настройки adversarial-части:

- `fm_loss_weight: float = 2.0`
- `gen_loss_weight: float = 1.0`
- `disc_loss_weight: float = 1.0`
- `disc_learning_rate: float = 2e-4`

### 6. `src/vits_finetune/checkpoint.py`

Расширить `save_checkpoint` / `load_checkpoint`, чтобы сохранять и загружать
дискриминатор и оба оптимизатора (сейчас — одна модель + один оптимизатор).
Обратная совместимость не требуется (старые `.pt` от сломанного цикла — мусор).

### 7. `tests/test_vits_finetune_smoke.py` (новый)

CPU smoke-тест без скачивания данных и реальной VITS-модели, где возможно:

- Прогнать дискриминатор и все новые лоссы на крошечных фейковых тензорах →
  результаты конечны (не NaN/Inf), скаляры.
- Прогнать пару GAN-шагов (с мини-моделью или замоканным `forward_train`) →
  не падает, параметры дискриминатора и генератора реально обновляются.

### 8. `colab_train.ipynb`

Обновить под новый цикл: установка зависимостей → клон/подключение `src/` →
запуск `python -m vits_finetune.train` (GAN) → синтез → оценка WER через
`src/evaluation/`.

## Критерии готовности

- **Локально:** `pytest tests/test_vits_finetune_smoke.py` зелёный; `ruff check`
  чистый.
- **На Colab (вручную пользователем):** прогон обучения на GPU, синтез,
  WER заметно ниже прежних 90–120%.

## Открытые риски

- Точная сигнатура `self.vits.decoder` и `posterior_encoder` в установленной
  версии `transformers` — проверить при реализации (форма `z_segment`,
  выход декодера).
- Совпадение длин predicted/target waveform по сэмплам после нарезки —
  обрезать по минимуму, как уже делается для mel.
