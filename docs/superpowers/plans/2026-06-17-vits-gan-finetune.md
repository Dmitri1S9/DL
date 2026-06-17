# VITS GAN Fine-tune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Достроить собственный VITS fine-tune до полноценного GAN-обучения (дискриминатор HiFi-GAN + adversarial/feature-matching лоссы + двух-оптимизаторный цикл).

**Architecture:** В `transformers.VitsModel` нет дискриминатора. Пишем его с нуля (стандарт HiFi-GAN: MPD + MSD) как отдельный модуль `discriminator.py`, добавляем три adversarial-лосса в `losses.py`, `forward_train` начинает возвращать пару waveform (настоящий/поддельный сегмент), а цикл в `train.py` становится GAN: на каждом шаге сначала шаг дискриминатора, потом шаг генератора.

**Tech Stack:** PyTorch, torchaudio, HuggingFace transformers (VitsModel), pytest, ruff.

## Global Constraints

- НЕ обновлять `transformers`/`accelerate` и прочие версии библиотек.
- НЕ использовать чужой рецепт ylacombe.
- Стиль: одинарные кавычки, длина строки 88, `ruff` (`make lint` / `make format`).
- Import root — `src` (запуск с `PYTHONPATH=src`).
- Коммиты разрешены (Эмир дал разрешение на эту задачу); коммитить после каждой задачи.
- Логи через `core.logger`/стандартный logging, без новых `print` в библиотечном коде.

---

### Task 1: Дискриминатор HiFi-GAN (`discriminator.py`)

**Files:**
- Create: `src/vits_finetune/discriminator.py`
- Test: `tests/test_vits_finetune_smoke.py`

**Interfaces:**
- Produces: `VitsDiscriminator(periods=(2,3,5,7,11))`; метод
  `forward(real, fake) -> (d_real_outputs, d_fake_outputs, fmaps_real, fmaps_fake)`.
  `real`/`fake` — `(B, 1, T)`. `d_*_outputs` — `list[Tensor]` логитов
  `(B, N)` по sub-дискриминаторам. `fmaps_*` — `list[list[Tensor]]`
  карт признаков.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vits_finetune_smoke.py
import torch
from vits_finetune.discriminator import VitsDiscriminator


def test_discriminator_shapes_and_finite():
    disc = VitsDiscriminator()
    real = torch.randn(2, 1, 8192)
    fake = torch.randn(2, 1, 8192)
    d_real, d_fake, fmap_real, fmap_fake = disc(real, fake)
    # 5 period + 3 scale sub-discriminators
    assert len(d_real) == len(d_fake) == 8
    assert len(fmap_real) == len(fmap_fake) == 8
    for out in d_real + d_fake:
        assert out.shape[0] == 2
        assert torch.isfinite(out).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_vits_finetune_smoke.py::test_discriminator_shapes_and_finite -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vits_finetune.discriminator'`

- [ ] **Step 3: Write the implementation**

```python
# src/vits_finetune/discriminator.py
"""HiFi-GAN discriminator for VITS adversarial fine-tuning (written from scratch).

VITS is a GAN: the decoder (generator) only learns to produce realistic
waveforms when trained against a discriminator. ``transformers.VitsModel`` ships
the generator but not the discriminator, so we add the standard HiFi-GAN one:
a multi-period discriminator (MPD) + a multi-scale discriminator (MSD).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm

LRELU_SLOPE = 0.1


class DiscriminatorP(nn.Module):
    """One period sub-discriminator: reshapes the 1D wave to 2D by ``period``."""

    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3) -> None:
        super().__init__()
        self.period = period
        pad = (kernel_size - 1) // 2
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), (pad, 0))),
            weight_norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), (pad, 0))),
            weight_norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), (pad, 0))),
            weight_norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), (pad, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, (pad, 0))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, (1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap: list[torch.Tensor] = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), 'reflect')
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for layer in self.convs:
            x = F.leaky_relu(layer(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class DiscriminatorS(nn.Module):
    """One scale sub-discriminator operating directly on the 1D wave."""

    def __init__(self, use_spectral_norm: bool = False) -> None:
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap: list[torch.Tensor] = []
        for layer in self.convs:
            x = F.leaky_relu(layer(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: tuple[int, ...] = (2, 3, 5, 7, 11)) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(p) for p in periods])

    def forward(self, x: torch.Tensor) -> tuple[list, list]:
        outs, fmaps = [], []
        for d in self.discriminators:
            out, fmap = d(x)
            outs.append(out)
            fmaps.append(fmap)
        return outs, fmaps


class MultiScaleDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, x: torch.Tensor) -> tuple[list, list]:
        outs, fmaps = [], []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                x = self.meanpools[i - 1](x)
            out, fmap = d(x)
            outs.append(out)
            fmaps.append(fmap)
        return outs, fmaps


class VitsDiscriminator(nn.Module):
    """MPD + MSD combined; compares a real and a fake waveform segment."""

    def __init__(self, periods: tuple[int, ...] = (2, 3, 5, 7, 11)) -> None:
        super().__init__()
        self.mpd = MultiPeriodDiscriminator(periods)
        self.msd = MultiScaleDiscriminator()

    def forward(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> tuple[list, list, list, list]:
        r_mpd, fr_mpd = self.mpd(real)
        f_mpd, ff_mpd = self.mpd(fake)
        r_msd, fr_msd = self.msd(real)
        f_msd, ff_msd = self.msd(fake)
        return (
            r_mpd + r_msd,
            f_mpd + f_msd,
            fr_mpd + fr_msd,
            ff_mpd + ff_msd,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_vits_finetune_smoke.py::test_discriminator_shapes_and_finite -v`
Expected: PASS

- [ ] **Step 5: Lint + commit**

```bash
.venv/bin/ruff check src/vits_finetune/discriminator.py
git add src/vits_finetune/discriminator.py tests/test_vits_finetune_smoke.py
git commit -m "feat(vits): add HiFi-GAN discriminator (MPD + MSD)"
```

---

### Task 2: Adversarial-лоссы (`losses.py`)

**Files:**
- Modify: `src/vits_finetune/losses.py`
- Test: `tests/test_vits_finetune_smoke.py`

**Interfaces:**
- Consumes: выход `VitsDiscriminator.forward` из Task 1.
- Produces:
  - `discriminator_loss(d_real_outputs, d_fake_outputs) -> Tensor`
  - `generator_adv_loss(d_fake_outputs) -> Tensor`
  - `feature_matching_loss(fmaps_real, fmaps_fake) -> Tensor`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_vits_finetune_smoke.py
from vits_finetune.losses import (
    discriminator_loss,
    feature_matching_loss,
    generator_adv_loss,
)


def test_adversarial_losses_finite():
    disc = VitsDiscriminator()
    real = torch.randn(2, 1, 8192)
    fake = torch.randn(2, 1, 8192)
    d_real, d_fake, fmap_real, fmap_fake = disc(real, fake)

    loss_d = discriminator_loss(d_real, d_fake)
    loss_g = generator_adv_loss(d_fake)
    loss_fm = feature_matching_loss(fmap_real, fmap_fake)

    for loss in (loss_d, loss_g, loss_fm):
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_vits_finetune_smoke.py::test_adversarial_losses_finite -v`
Expected: FAIL with `ImportError: cannot import name 'discriminator_loss'`

- [ ] **Step 3: Write the implementation** (append to `losses.py`)

```python
def discriminator_loss(
    d_real_outputs: list[torch.Tensor],
    d_fake_outputs: list[torch.Tensor],
) -> torch.Tensor:
    """LS-GAN discriminator loss: push real -> 1, fake -> 0."""
    loss = torch.zeros((), device=d_real_outputs[0].device)
    for d_real, d_fake in zip(d_real_outputs, d_fake_outputs):
        loss = loss + torch.mean((1 - d_real) ** 2) + torch.mean(d_fake ** 2)
    return loss


def generator_adv_loss(d_fake_outputs: list[torch.Tensor]) -> torch.Tensor:
    """LS-GAN generator loss: push the discriminator's fake score -> 1."""
    loss = torch.zeros((), device=d_fake_outputs[0].device)
    for d_fake in d_fake_outputs:
        loss = loss + torch.mean((1 - d_fake) ** 2)
    return loss


def feature_matching_loss(
    fmaps_real: list[list[torch.Tensor]],
    fmaps_fake: list[list[torch.Tensor]],
) -> torch.Tensor:
    """L1 between real and fake discriminator feature maps (real detached)."""
    loss = torch.zeros((), device=fmaps_fake[0][0].device)
    for real_maps, fake_maps in zip(fmaps_real, fmaps_fake):
        for real_map, fake_map in zip(real_maps, fake_maps):
            loss = loss + torch.mean(torch.abs(real_map.detach() - fake_map))
    return loss
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_vits_finetune_smoke.py::test_adversarial_losses_finite -v`
Expected: PASS

- [ ] **Step 5: Lint + commit**

```bash
.venv/bin/ruff check src/vits_finetune/losses.py
git add src/vits_finetune/losses.py tests/test_vits_finetune_smoke.py
git commit -m "feat(vits): add adversarial + feature-matching losses"
```

---

### Task 3: GAN-шаги обучают параметры (тест + проверка обратного хода)

**Files:**
- Test: `tests/test_vits_finetune_smoke.py`

**Interfaces:**
- Consumes: Task 1 (`VitsDiscriminator`), Task 2 (лоссы).

Этот тест доказывает главное свойство GAN-цикла без скачивания VITS: и
дискриминатор, и «генератор» реально получают градиент и обновляются.

- [ ] **Step 1: Write the test**

```python
# append to tests/test_vits_finetune_smoke.py
def test_gan_step_updates_both_sides():
    torch.manual_seed(0)
    disc = VitsDiscriminator()
    # tiny stand-in generator: noise -> 8192-sample wave
    generator = torch.nn.Sequential(torch.nn.Conv1d(1, 1, 3, padding=1))
    opt_d = torch.optim.AdamW(disc.parameters(), lr=1e-3)
    opt_g = torch.optim.AdamW(generator.parameters(), lr=1e-3)

    real = torch.randn(2, 1, 8192)
    noise = torch.randn(2, 1, 8192)

    g_before = next(generator.parameters()).clone()
    d_before = next(disc.parameters()).clone()

    # discriminator step
    fake = generator(noise)
    opt_d.zero_grad()
    d_real, d_fake, _, _ = disc(real, fake.detach())
    discriminator_loss(d_real, d_fake).backward()
    opt_d.step()

    # generator step
    opt_g.zero_grad()
    _, d_fake2, fmap_real, fmap_fake = disc(real, fake)
    loss_g = generator_adv_loss(d_fake2) + feature_matching_loss(fmap_real, fmap_fake)
    loss_g.backward()
    opt_g.step()

    assert not torch.equal(d_before, next(disc.parameters()))
    assert not torch.equal(g_before, next(generator.parameters()))
```

- [ ] **Step 2: Run test**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_vits_finetune_smoke.py::test_gan_step_updates_both_sides -v`
Expected: PASS (если FAIL — баг в Task 1/2, чинить там)

- [ ] **Step 3: Commit**

```bash
git add tests/test_vits_finetune_smoke.py
git commit -m "test(vits): GAN step updates discriminator and generator"
```

---

### Task 4: `forward_train` возвращает `target_waveform` (`model.py`)

**Files:**
- Modify: `src/vits_finetune/model.py:131-155`
- Test: `tests/test_vits_finetune_smoke.py`

**Interfaces:**
- Produces: `forward_train` теперь добавляет в возвращаемый dict
  `predicted_waveform` `(B, 1, S)` и `target_waveform` `(B, 1, S)`, где
  `S = segment_frames * hop_length`, обрезанные до общего минимума по длине.

`_slice_segments` (уже есть в `model.py`) работает на `(B, C, T)` с
по-примерными `starts` и фиксированной длиной — переиспользуем его для волны.

- [ ] **Step 1: Write the test** (юнит-тест на нарезку волны, без VITS)

```python
# append to tests/test_vits_finetune_smoke.py
from vits_finetune.model import _slice_segments


def test_slice_segments_waveform_crop():
    wav = torch.arange(20, dtype=torch.float32).view(1, 1, 20)
    starts = torch.tensor([5])
    out = _slice_segments(wav, starts, 4)
    assert out.shape == (1, 1, 4)
    assert torch.equal(out[0, 0], torch.tensor([5.0, 6.0, 7.0, 8.0]))
```

- [ ] **Step 2: Run test to verify it passes (helper exists already)**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_vits_finetune_smoke.py::test_slice_segments_waveform_crop -v`
Expected: PASS

- [ ] **Step 3: Modify `forward_train`** — после строки с `predicted_mel` и
расчёта `num_frames`, добавить нарезку реальной волны и вернуть обе волны.
Заменить блок `return {...}` (model.py:145-155) на:

```python
        num_frames = min(predicted_mel.shape[-1], target_mel.shape[-1])

        wav_segment_size = segment_frames * self.training_config.hop_length
        wav_starts = starts * self.training_config.hop_length
        target_waveform = _slice_segments(
            batch['waveform'], wav_starts, wav_segment_size
        )
        num_samples = min(predicted_waveform.shape[-1], target_waveform.shape[-1])

        return {
            'predicted_mel': predicted_mel[..., :num_frames],
            'target_mel': target_mel[..., :num_frames],
            'predicted_waveform': predicted_waveform[..., :num_samples],
            'target_waveform': target_waveform[..., :num_samples],
            'posterior_mean': posterior_mean,
            'posterior_log_stddev': posterior_log_stddev,
            'prior_mean': prior_mean,
            'prior_log_stddev': prior_log_stddev,
            'z_mask': spec_mask,
            'duration_loss': duration_loss,
            'z_p': z_p,
        }
```

- [ ] **Step 4: Run the full smoke suite**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_vits_finetune_smoke.py -v`
Expected: PASS (все тесты)

- [ ] **Step 5: Lint + commit**

```bash
.venv/bin/ruff check src/vits_finetune/model.py
git add src/vits_finetune/model.py tests/test_vits_finetune_smoke.py
git commit -m "feat(vits): forward_train returns real+fake waveform segments"
```

---

### Task 5: Чекпоинт сохраняет дискриминатор и оба оптимизатора (`checkpoint.py`)

**Files:**
- Modify: `src/vits_finetune/checkpoint.py`

**Interfaces:**
- Produces: `save_checkpoint(..., discriminator=None, disc_optimizer=None, ...)`
  и `load_checkpoint(..., discriminator=None, disc_optimizer=None, ...)` —
  опциональные параметры, обратная совместимость сохранена.

- [ ] **Step 1: Modify `save_checkpoint`** — добавить параметры и запись:

```python
def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int = 0,
    discriminator: torch.nn.Module | None = None,
    disc_optimizer: torch.optim.Optimizer | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save model/optimizer (+ optional discriminator) and training progress."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'epoch': epoch,
    }
    if discriminator is not None:
        payload['discriminator'] = discriminator.state_dict()
    if disc_optimizer is not None:
        payload['disc_optimizer'] = disc_optimizer.state_dict()
    if extra is not None:
        payload['extra'] = extra
    torch.save(payload, path)
```

- [ ] **Step 2: Modify `load_checkpoint`** — добавить параметры и загрузку:

```python
def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    discriminator: torch.nn.Module | None = None,
    disc_optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load a checkpoint written by ``save_checkpoint``."""
    path = Path(path)
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if discriminator is not None and 'discriminator' in checkpoint:
        discriminator.load_state_dict(checkpoint['discriminator'])
    if disc_optimizer is not None and 'disc_optimizer' in checkpoint:
        disc_optimizer.load_state_dict(checkpoint['disc_optimizer'])
    return checkpoint
```

- [ ] **Step 3: Lint + commit**

```bash
.venv/bin/ruff check src/vits_finetune/checkpoint.py
git add src/vits_finetune/checkpoint.py
git commit -m "feat(vits): checkpoint discriminator and both optimizers"
```

---

### Task 6: Веса adversarial в конфиге (`config.py`)

**Files:**
- Modify: `src/vits_finetune/config.py:39-41`

**Interfaces:**
- Produces: `TrainingConfig` получает поля `fm_loss_weight=2.0`,
  `gen_loss_weight=1.0`, `disc_loss_weight=1.0`, `disc_learning_rate=2e-4`.

- [ ] **Step 1: Add fields** — в блок `--- loss weights ---` (после
`kl_loss_weight`) и в training-блок добавить:

```python
    # --- loss weights ---
    mel_loss_weight: float = 45.0
    kl_loss_weight: float = 1.0
    fm_loss_weight: float = 2.0
    gen_loss_weight: float = 1.0
    disc_loss_weight: float = 1.0

    # --- adversarial optimizer ---
    disc_learning_rate: float = 2e-4
```

- [ ] **Step 2: Sanity import**

Run: `PYTHONPATH=src .venv/bin/python -c "from vits_finetune.config import TrainingConfig; print(TrainingConfig().fm_loss_weight)"`
Expected: `2.0`

- [ ] **Step 3: Commit**

```bash
git add src/vits_finetune/config.py
git commit -m "feat(vits): add adversarial loss weights + disc lr to config"
```

---

### Task 7: GAN-цикл в `train.py` (убрать декораторы)

**Files:**
- Modify: `src/vits_finetune/train.py` (класс `Trainer` целиком, строки 31-157)

**Interfaces:**
- Consumes: Task 1 (`VitsDiscriminator`), Task 2 (лоссы), Task 4
  (`forward_train` с waveform), Task 5 (checkpoint), Task 6 (config-поля).

- [ ] **Step 1: Заменить тело класса `Trainer`** (от `class Trainer:` до конца
метода `train`) на явный GAN-цикл без декораторов:

```python
class Trainer:
    @staticmethod
    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description='Fine-tune VITS on a custom voice.')
        parser.add_argument('--dataset-repo-id', default=None, help='HF Hub dataset repo id.')
        parser.add_argument('--checkpoint-dir', type=Path, default=None)
        parser.add_argument('--batch-size', type=int, default=None)
        parser.add_argument('--learning-rate', type=float, default=None)
        parser.add_argument('--num-epochs', type=int, default=None)
        parser.add_argument('--num-workers', type=int, default=None, help='DataLoader workers.')
        parser.add_argument('--max-train-clips', type=int, default=None,
                            help='Cap training clips; test set always held out.')
        parser.add_argument('--resume', type=Path, default=None, help='Checkpoint .pt to resume from.')
        parser.add_argument('--device', default=None, help='"cuda" or "cpu" (default: auto-detect).')
        return parser.parse_args()

    @staticmethod
    def build_config(args: argparse.Namespace) -> TrainingConfig:
        overrides = {
            name: value
            for name in (
                'dataset_repo_id', 'checkpoint_dir',
                'batch_size', 'learning_rate', 'num_epochs',
                'num_workers', 'max_train_clips',
            )
            if (value := getattr(args, name, None)) is not None
        }
        return TrainingConfig(**overrides)

    def __init__(self, config: TrainingConfig, model_config: VitsModelConfig,
                 args: argparse.Namespace) -> None:
        self.device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        logger.info(f'Using device: {self.device}')

        torch.manual_seed(config.seed)
        self.config = config
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.pretrained_model_name)
        self.dataset = VitsFinetuneDataset(config, self.tokenizer)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config.num_workers,
        )
        self.model = VitsFinetuneModel(model_config, config).to(self.device)
        self.discriminator = VitsDiscriminator().to(self.device)
        self.optim_g = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, betas=(0.8, 0.99))
        self.optim_d = torch.optim.AdamW(
            self.discriminator.parameters(), lr=config.disc_learning_rate, betas=(0.8, 0.99))
        self.start_epoch, self.global_step = 0, 0

    def train_step(self, batch: dict) -> dict:
        outputs = self.model.forward_train(batch)
        real = outputs['target_waveform']
        fake = outputs['predicted_waveform']

        # --- discriminator step ---
        self.optim_d.zero_grad()
        d_real, d_fake, _, _ = self.discriminator(real, fake.detach())
        loss_d = self.config.disc_loss_weight * discriminator_loss(d_real, d_fake)
        loss_d.backward()
        self.optim_d.step()

        # --- generator step ---
        self.optim_g.zero_grad()
        _, d_fake_g, fmap_real, fmap_fake = self.discriminator(real, fake)
        recon = recon_loss(outputs['predicted_mel'], outputs['target_mel'])
        kl = kl_loss(
            outputs['z_p'], outputs['posterior_log_stddev'],
            outputs['prior_mean'], outputs['prior_log_stddev'], outputs['z_mask'],
        )
        dur = outputs['duration_loss']
        adv = generator_adv_loss(d_fake_g)
        fm = feature_matching_loss(fmap_real, fmap_fake)
        loss_g = (
            self.config.mel_loss_weight * recon
            + self.config.kl_loss_weight * kl
            + dur
            + self.config.gen_loss_weight * adv
            + self.config.fm_loss_weight * fm
        )
        loss_g.backward()
        self.optim_g.step()

        return {
            'loss_d': loss_d.item(), 'loss_g': loss_g.item(),
            'recon': recon.item(), 'kl': kl.item(), 'dur': dur.item(),
            'adv': adv.item(), 'fm': fm.item(),
        }

    def _save(self, name: str, epoch: int) -> None:
        path = self.config.checkpoint_dir / name
        save_checkpoint(
            path, self.model, self.optim_g, self.global_step, epoch,
            discriminator=self.discriminator, disc_optimizer=self.optim_d,
        )
        logger.info(f'Saved checkpoint to {path}')

    def train(self) -> None:
        if self.args.resume:
            ckpt = load_checkpoint(
                self.args.resume, self.model, self.optim_g,
                discriminator=self.discriminator, disc_optimizer=self.optim_d,
                map_location=self.device,
            )
            self.start_epoch = ckpt.get('epoch', 0)
            self.global_step = ckpt.get('step', 0)
            logger.info(f'Resumed from {self.args.resume} at epoch {self.start_epoch}')

        for epoch in range(self.start_epoch, self.config.num_epochs):
            for batch in self.dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss_dict = self.train_step(batch)

                if self.global_step % self.config.log_every == 0:
                    logger.info(
                        f'Epoch {epoch} | Step {self.global_step} | '
                        + ' | '.join(f'{k}: {v:.4f}' for k, v in loss_dict.items())
                    )
                if self.global_step % self.config.checkpoint_every == 0:
                    self._save(f'step_{self.global_step}.pt', epoch)
                self.global_step += 1

            self._save(f'epoch_{epoch + 1}.pt', epoch + 1)
            logger.info(f'End of epoch {epoch}')
```

- [ ] **Step 2: Обновить импорты** в начале `train.py` — заменить строку
импорта лоссов и добавить дискриминатор:

```python
from vits_finetune.discriminator import VitsDiscriminator
from vits_finetune.losses import (
    discriminator_loss,
    feature_matching_loss,
    generator_adv_loss,
    kl_loss,
    recon_loss,
)
```

- [ ] **Step 3: Проверить, что модуль импортируется и парсер args жив**

Run: `PYTHONPATH=src .venv/bin/python -c "from vits_finetune.train import Trainer; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Lint + commit**

```bash
.venv/bin/ruff check src/vits_finetune/train.py
git add src/vits_finetune/train.py
git commit -m "refactor(vits): rewrite training loop as GAN with two optimizers"
```

---

### Task 8: Обновить `colab_train.ipynb` под GAN-цикл

**Files:**
- Modify: `colab_train.ipynb`

**Interfaces:**
- Consumes: новый `python -m vits_finetune.train` (GAN), `src/evaluation/`.

- [ ] **Step 1: Прочитать текущий ноутбук**, понять текущие ячейки (clone/setup,
запуск train, eval).

- [ ] **Step 2: Обновить ячейки** так, чтобы они:
  - клонировали репо и ставили `requirements.txt` + system `espeak-ng`;
  - запускали `!PYTHONPATH=src python -m vits_finetune.train --num-epochs N --batch-size B`;
  - после обучения синтезировали через `src/model/synthesize.py` или
    `vits_finetune/synthesize.py` и считали WER через `src/evaluation/`.

- [ ] **Step 3: Commit**

```bash
git add colab_train.ipynb
git commit -m "docs(colab): update notebook for GAN training loop + src eval"
```

---

### Task 9: Финальная проверка и push

- [ ] **Step 1: Прогнать весь smoke-набор и линт**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_vits_finetune_smoke.py -v && .venv/bin/ruff check src/vits_finetune/`
Expected: всё зелёное

- [ ] **Step 2: Прогнать существующий smoke-тест проекта (не сломали ли)**

Run: `make smoke` (или `PYTHONPATH=src .venv/bin/pytest tests/ -v`)
Expected: PASS

- [ ] **Step 3: Push ветки**

```bash
git push origin finetune
```

---

## Self-Review

**Spec coverage:**
- discriminator.py → Task 1 ✅
- losses (3 новых) → Task 2 ✅
- forward_train target_waveform → Task 4 ✅
- train.py GAN-цикл без декораторов → Task 7 ✅
- config веса → Task 6 ✅
- checkpoint disc+2 optim → Task 5 ✅
- smoke-тест → Tasks 1–4 ✅
- colab_train.ipynb → Task 8 ✅
- критерии готовности (pytest+ruff) → Task 9 ✅

**Placeholder scan:** код приведён полностью в каждом шаге; Task 8 описывает
правки ноутбука словами, т.к. точные ячейки читаются на месте (Step 1).

**Type consistency:** `VitsDiscriminator.forward -> (d_real, d_fake, fmap_real,
fmap_fake)` используется одинаково в Tasks 2, 3, 7. Имена лоссов
(`discriminator_loss`, `generator_adv_loss`, `feature_matching_loss`) совпадают
во всех задачах. Ключи `predicted_waveform`/`target_waveform` из Task 4
читаются в Task 7.
