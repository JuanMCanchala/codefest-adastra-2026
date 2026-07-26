"""Extraccion de HTML con trafilatura (mejor limpieza de boilerplate)."""
from __future__ import annotations

from pathlib import Path


def extract_html(path: str | Path) -> str:
    import trafilatura  # import diferido
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    text = trafilatura.extract(raw, include_comments=False, include_tables=True)
    return text or ""
