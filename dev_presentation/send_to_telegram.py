#!/usr/bin/env python3
"""Send the DL-presentation PDFs to Telegram via OpenClaw's bot credentials.

Reuses OpenClaw's token/chat_id resolution (lib.telegram) and adds a
sendDocument (multipart) call, which the built-in send_telegram() lacks.

Usage:
    python3 send_to_telegram.py            # send all PDFs + a caption message
    DRY_RUN=true python3 send_to_telegram.py   # preview, send nothing
"""

from __future__ import annotations

import mimetypes
import os
import ssl
import sys
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path.home() / 'dev' / 'openclaw'))
from lib.telegram import get_bot_token, get_chat_id, send_telegram

HERE = Path(__file__).resolve().parent
PDF = HERE / 'pdf'

# (file, human caption) — order = send order
DOCS = [
    (HERE / 'slides.pdf', '🎤 Slides — the deck (present from this)'),
    (PDF / 'CHEATSHEET.pdf', '📇 Cheat sheet — print & hold while presenting'),
    (PDF / 'SPEAKER_NOTES.pdf', '🗒️ Speaker notes — per-slide script + timing'),
    (PDF / 'PRESENTATION_PLAYBOOK.pdf', '🎯 Playbook — delivery, demo, examiner tips'),
    (PDF / 'DISCUSSION_QA.pdf', '❓ Discussion Q&A — ~50 answers by difficulty'),
    (PDF / 'TECHNICAL_APPENDIX.pdf', '🧠 Technical appendix — VITS/MAS/HiFi-GAN math'),
    (
        PDF / 'dl-presentation-brief.pdf',
        '📘 Master brief — everything behind the slides',
    ),
]


def send_document(path: Path, caption: str = '', chat_id: str = '') -> bool:
    """POST a file to Telegram's sendDocument as multipart/form-data."""
    if os.environ.get('DRY_RUN', 'false').lower() == 'true':
        print(
            f'[DRY RUN] would send {path.name} ({path.stat().st_size // 1024} KB) — {caption}'
        )
        return True

    token = get_bot_token()
    chat_id = chat_id or get_chat_id()
    if not token or not chat_id:
        print('ERROR: Telegram not configured (token/chat_id)')
        return False
    if not path.exists():
        print(f'ERROR: missing {path}')
        return False

    boundary = f'----oc{uuid.uuid4().hex}'
    ctype = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
    parts: list[bytes] = []

    def field(name: str, value: str) -> None:
        parts.extend(
            [
                f'--{boundary}'.encode(),
                f'Content-Disposition: form-data; name="{name}"'.encode(),
                b'',
                value.encode(),
            ]
        )

    field('chat_id', str(chat_id))
    if caption:
        field('caption', caption)
    parts.extend(
        [
            f'--{boundary}'.encode(),
            f'Content-Disposition: form-data; name="document"; filename="{path.name}"'.encode(),
            f'Content-Type: {ctype}'.encode(),
            b'',
            path.read_bytes(),
            f'--{boundary}--'.encode(),
            b'',
        ]
    )
    body = b'\r\n'.join(parts)

    req = urllib.request.Request(
        f'https://api.telegram.org/bot{token}/sendDocument',
        data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
        method='POST',
    )
    try:
        urllib.request.urlopen(req, timeout=60, context=ssl.create_default_context())
        return True
    except Exception as e:
        print(f'ERROR sending {path.name}: {e}')
        return False


def main() -> int:
    intro = (
        '📦 <b>DL Project 13 — Text to Speech: presentation package</b>\n'
        'Fine-tuning VITS · 192.151 Intro to DL, 2026S.\n\n'
        'Sending 7 files: the slide deck + cheat sheet, speaker notes, '
        'delivery playbook, Q&A bank, technical appendix, and the master brief.'
    )
    ok_intro = send_telegram(intro)
    print(f'intro message: {"sent" if ok_intro else "FAILED"}')

    sent = 0
    for path, caption in DOCS:
        if send_document(path, caption):
            print(f'  ✓ {path.name}')
            sent += 1
        else:
            print(f'  ✗ {path.name}')
    print(f'\n{sent}/{len(DOCS)} documents sent.')
    return 0 if sent == len(DOCS) else 1


if __name__ == '__main__':
    raise SystemExit(main())
