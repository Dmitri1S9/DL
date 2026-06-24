"""Parse models/vits_finetune/train.log and plot each loss metric with a trend line.

Usage: PYTHONPATH=src .venv/Scripts/python.exe scripts/plot_train_log.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LOG_PATH = Path('models/vits_finetune/train.log')
OUT_DIR = Path('log_stat')
WINDOW = 100  # rolling-mean window (steps of log_every=50 -> ~5000 step span)

# metric name -> regex capturing the value
PATTERNS = {
    'd': re.compile(r'\| d: ([\d.]+)'),
    'g': re.compile(r'\| g: ([\d.]+)'),
    'recon': re.compile(r'recon: ([\d.]+)'),
    'kl': re.compile(r'kl: ([\d.]+)'),
    'dur': re.compile(r'dur: ([\d.]+)'),
    'adv': re.compile(r'adv: ([\d.]+)'),
    'fm': re.compile(r'fm: ([\d.]+)'),
}

# metric name -> full description used in plot title/axis (no abbreviations)
FULL_NAMES = {
    'd': 'Discriminator loss',
    'g': 'Generator loss (total)',
    'recon': 'Reconstruction loss (L1 on mel-spectrogram)',
    'kl': 'KL divergence loss (posterior vs. prior)',
    'dur': 'Duration predictor loss',
    'adv': 'Adversarial loss (generator side)',
    'fm': 'Feature-matching loss',
}

STEP_RE = re.compile(r'Step (\d+)')


def parse_log() -> dict[str, tuple[list[int], list[float]]]:
    series: dict[str, tuple[list[int], list[float]]] = {
        name: ([], []) for name in PATTERNS
    }
    last_step = None
    for line in LOG_PATH.read_text(encoding='utf-8').splitlines():
        m = STEP_RE.search(line)
        if m:
            last_step = int(m.group(1))
        if last_step is None:
            continue
        for name, pattern in PATTERNS.items():
            m = pattern.search(line)
            if m:
                steps, values = series[name]
                steps.append(last_step)
                values.append(float(m.group(1)))
    return series


def plot_metric(name: str, steps: list[int], values: list[float]) -> None:
    steps_arr = np.array(steps)
    values_arr = np.array(values)

    window = min(WINDOW, max(2, len(values_arr) // 5))
    # expanding mean for the first `window` points so the trend line covers
    # the full range starting at step 0, instead of only starting once a
    # full window of data is available.
    trend = np.array(
        [
            values_arr[max(0, i - window + 1) : i + 1].mean()
            for i in range(len(values_arr))
        ]
    )
    trend_steps = steps_arr

    # expected trend: linear fit on the rolling mean, extrapolated 20% past the data
    coeffs = np.polyfit(trend_steps, trend, 1)
    extra_span = int((steps_arr[-1] - steps_arr[0]) * 0.2)
    fit_steps = np.array([steps_arr[0], steps_arr[-1] + extra_span])
    fit_values = np.polyval(coeffs, fit_steps)

    full_name = FULL_NAMES[name]

    plt.figure(figsize=(10, 5))
    plt.plot(
        steps_arr,
        values_arr,
        linewidth=0.5,
        alpha=0.4,
        color='tab:blue',
        label=full_name,
    )
    plt.plot(
        trend_steps,
        trend,
        linewidth=2,
        color='tab:red',
        label=f'rolling mean (window={window})',
    )
    plt.plot(
        fit_steps,
        fit_values,
        linewidth=1.5,
        color='black',
        linestyle='--',
        label=f'expected trend (slope={coeffs[0]:.2e}/step)',
    )
    plt.axvline(steps_arr[-1], color='gray', linewidth=0.8, linestyle=':')
    plt.xlabel('training step')
    plt.ylabel(full_name)
    plt.title(f'{full_name} over training')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = OUT_DIR / f'{name}.png'
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f'saved {out_path}')


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    series = parse_log()
    for name, (steps, values) in series.items():
        if not values:
            print(f'skip {name}: no data')
            continue
        plot_metric(name, steps, values)


if __name__ == '__main__':
    main()
