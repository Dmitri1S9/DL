"""Manifest persistence: read/write the JSONL test manifest.

The schema (TestItem) lives in core.dto; this module is just the I/O over it.
The manifest is the hand-off point between data.prepare (writes) and both
model.synthesize and evaluation.evaluate (read).
"""

from collections.abc import Iterable
from pathlib import Path

from core.dto import TestItem


def write_manifest(items: Iterable[TestItem], path: Path) -> None:
    """Write items as JSONL to ``path`` (one TestItem per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for item in items:
            f.write(item.model_dump_json() + '\n')


def read_manifest(path: Path) -> list[TestItem]:
    """Read a JSONL manifest back into validated TestItem objects."""
    with Path(path).open(encoding='utf-8') as f:
        return [TestItem.model_validate_json(line) for line in f if line.strip()]
