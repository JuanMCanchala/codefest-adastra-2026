"""Extraccion de texto por formato (Seccion 2.1) y dispatch.

Cada extractor recibe una ruta y devuelve un Document con texto plano en orden
de lectura + metadata de origen. La CLAVE es preservar `fuente` (ruta/URL
original de ADL) porque el F1@3 se empareja por ese campo (spec 10.2.1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    doc_id: str
    fuente: str            # ruta/URL original tal como la entrega ADL  <- CLAVE
    formato: str           # pdf | html | md | json | csv | xlsx | image | pbf
    text: str              # texto plano extraido, orden de lectura preservado
    fenomeno: int = 0      # 1, 2, 3 (se asigna segun la carpeta/fuente del corpus)
    idioma: str | None = None
    titulo: str | None = None
    fecha: str | None = None
    extra: dict = field(default_factory=dict)


EXT_TO_FORMAT = {
    ".pdf": "pdf",
    ".html": "html", ".htm": "html",
    ".md": "md", ".txt": "md",
    ".json": "json",
    ".csv": "csv",
    ".xlsx": "xlsx", ".xls": "xlsx",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".tiff": "image",
    ".pbf": "pbf",
}


def detect_format(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    fmt = EXT_TO_FORMAT.get(ext)
    if fmt is None:
        raise ValueError(f"formato no reconocido para {path}")
    return fmt


def extract(path: str | Path, doc_id: str, fuente: str, fenomeno: int = 0) -> Document:
    """Dispatch por formato. Importa el extractor concreto de forma diferida."""
    fmt = detect_format(path)
    if fmt == "pdf":
        from .pdf import extract_pdf
        text = extract_pdf(path)
    elif fmt == "html":
        from .html_ext import extract_html
        text = extract_html(path)
    elif fmt in ("md", "json", "csv", "xlsx", "image", "pbf"):
        from .misc import extract_generic
        text = extract_generic(path, fmt)
    else:
        raise ValueError(f"sin extractor para formato {fmt}")
    return Document(doc_id=doc_id, fuente=fuente, formato=fmt, text=text, fenomeno=fenomeno)
