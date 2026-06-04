"""Project-wide logger (loguru).

Import the configured object directly:

    from core.logger import logger

Console level can be overridden via the LOG_LEVEL env var (default: INFO).
Everything from DEBUG up is always written to logs/run_<date>.log.
"""

import os
import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]  # core -> src -> project root
LOG_DIR = ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)

logger.remove()  # drop loguru's default handler so we fully control sinks

# Console: colored, human-readable, level controlled by LOG_LEVEL.
logger.add(
    sys.stderr,
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format=(
        '<green>{time:HH:mm:ss}</green> | '
        '<level>{level: <8}</level> | '
        '<level>{message}</level>'
    ),
)

# File: full DEBUG trail, rotated and compressed, kept for two weeks.
logger.add(
    LOG_DIR / 'run_{time:YYYY-MM-DD}.log',
    level='DEBUG',
    rotation='10 MB',
    retention='14 days',
    compression='zip',
    encoding='utf-8',
)

__all__ = ['logger']
