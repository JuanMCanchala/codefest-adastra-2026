"""Extractores para MD/TXT, JSON, CSV/XLSX, imagenes (OCR) y PBF (Seccion 2.1)."""
from __future__ import annotations

import json
from pathlib import Path


def extract_generic(path: str | Path, fmt: str) -> str:
    if fmt == "md":
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    if fmt == "json":
        return _extract_json(path)
    if fmt == "csv":
        return _extract_tabular(path, "csv")
    if fmt == "xlsx":
        return _extract_tabular(path, "xlsx")
    if fmt == "image":
        return _extract_ocr(path)
    if fmt == "pbf":
        return _extract_pbf(path)
    raise ValueError(f"formato sin extractor: {fmt}")


def _extract_json(path: str | Path) -> str:
    """Selecciona campos de texto (title, body_text, body_paragraphs) y concatena
    respetando el orden; url/date/authors/tags quedan como metadata aparte."""
    data = json.loads(Path(path).read_text(encoding="utf-8", errors="ignore"))
    text_keys = ("title", "body_text", "body_paragraphs", "content", "text", "summary")
    parts: list[str] = []

    def collect(obj):
        if isinstance(obj, dict):
            for k in text_keys:
                if k in obj:
                    v = obj[k]
                    if isinstance(v, list):
                        parts.extend(str(x) for x in v)
                    elif v:
                        parts.append(str(v))
        elif isinstance(obj, list):
            for item in obj:
                collect(item)

    collect(data)
    return "\n\n".join(parts)


def _extract_tabular(path: str | Path, fmt: str) -> str:
    """Cada fila -> 'columna: valor' separados por delimitador (spec CSV/XLSX).
    Cada fila es una unidad de fragmentacion independiente."""
    import pandas as pd  # import diferido
    df = pd.read_csv(path) if fmt == "csv" else pd.read_excel(path)
    rows = []
    for _, row in df.iterrows():
        cells = [f"{col}: {val}" for col, val in row.items() if str(val).strip() and str(val) != "nan"]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _extract_ocr(path: str | Path) -> str:
    """OCR multilingue sobre imagenes con texto (infografias/graficos)."""
    from paddleocr import PaddleOCR  # import diferido
    ocr = PaddleOCR(use_angle_cls=True, lang="es")  # 'es' cubre latin ES/EN/PT
    result = ocr.ocr(str(path), cls=True)
    lines = []
    for page in result or []:
        for _box, (text, _conf) in page or []:
            lines.append(text)
    return "\n".join(lines)


def _extract_pbf(path: str | Path) -> str:  # pragma: no cover - requiere pyosmium
    """Recorre capas del mapa y vuelca atributos como 'clave: valor'.
    Se queda con una sola version del elemento para no duplicar por zoom."""
    raise NotImplementedError("Activar cuando ADL entregue archivos .pbf reales")
