"""Bundle presentation.html into ONE self-contained file.

Inlines reveal.css / theme / plugins and the reveal.js + notes JS, and
base64-embeds every referenced image and audio clip. The result,
presentation_standalone.html, can be sent as a single file — no assets/ or
reveal/ folder needed (fixes the recurring "you must ship the images too" problem).

Run:  python build_standalone.py
"""

from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'presentation.html'
OUT = HERE / 'presentation_standalone.html'


def read(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or 'application/octet-stream'
    b64 = base64.b64encode(path.read_bytes()).decode('ascii')
    return f'data:{mime};base64,{b64}'


def inline_css(html: str) -> str:
    def repl(m: re.Match) -> str:
        href = m.group(1)
        if not href.startswith('reveal/'):
            return m.group(0)
        css = read(HERE / href)
        return f'<style>\n{css}\n</style>'

    return re.sub(r'<link[^>]*\bhref="([^"]+)"[^>]*>', repl, html)


def inline_js(html: str) -> str:
    def repl(m: re.Match) -> str:
        src = m.group(1)
        if not src.startswith('reveal/'):
            return m.group(0)
        js = read(HERE / src).replace('</script>', '<\\/script>')
        return f'<script>\n{js}\n</script>'

    return re.sub(r'<script[^>]*\bsrc="([^"]+)"[^>]*>\s*</script>', repl, html)


def embed_assets(html: str) -> str:
    def repl(m: re.Match) -> str:
        rel = m.group(1)
        path = HERE / rel
        if not path.exists():
            print(f'  ! missing asset (left as-is): {rel}')
            return m.group(0)
        return f'src="{data_uri(path)}"'

    return re.sub(r'src="(assets/[^"]+)"', repl, html)


def main() -> None:
    html = read(SRC)
    html = inline_css(html)
    html = inline_js(html)
    html = embed_assets(html)

    # sanity: nothing external left except data: URIs
    leftovers = re.findall(r'(?:src|href)="(?!data:)(reveal/|assets/)[^"]*"', html)
    if leftovers:
        print(f'  ! WARNING, {len(leftovers)} external refs remain: {set(leftovers)}')

    OUT.write_text(html, encoding='utf-8')
    mb = OUT.stat().st_size / 1e6
    print(f'Wrote {OUT.name} ({mb:.1f} MB) — self-contained, send as one file.')


if __name__ == '__main__':
    main()
