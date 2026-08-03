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


# Claves que NO aportan texto recuperable (rutas, ids, tamaños, marcas de tiempo).
# Se excluyen porque meterlas en el indice solo añade ruido al emparejamiento.
_JSON_SKIP_KEYS = {
    "url", "url_page", "url_pdf", "urls", "link", "links", "pdf_links", "href",
    "path", "filename", "file", "filepath", "image", "images", "thumbnail",
    "id", "study_id", "doc_id", "uuid", "slug", "hash",
    "size_bytes", "size_mb", "status", "scraped_at", "timestamp", "updated_at",
    "contenido_limitado", "mime", "extension",
}
# Claves cuyo texto va PRIMERO (encabezan el documento).
_JSON_TITLE_KEYS = ("title", "titulo", "nombre", "headline", "subtitle", "subtitulo")


def _extract_json(path: str | Path) -> str:
    """Extrae el texto de un JSON sea cual sea su esquema.

    Los JSON del corpus son heterogeneos: unos son articulos
    ({title, body_text, body_paragraphs}), otros paginas con {sections, lists} y
    otros catalogos (listas de fichas con {titulo, autores, pais, año}). En vez
    de buscar un conjunto fijo de claves, recorremos la estructura y recogemos
    todo el texto util, descartando las claves que solo contienen metadata
    tecnica (_JSON_SKIP_KEYS). Asi no perdemos contenido por un esquema no
    previsto, que en un corpus de ~950 JSON es lo mas probable.
    """
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return ""

    titles: list[str] = []
    body: list[str] = []
    seen: set[str] = set()

    def add(value: str, is_title: bool) -> None:
        value = value.strip()
        # descartar fragmentos triviales y URLs sueltas
        if len(value) < 3 or value.startswith(("http://", "https://", "data:")):
            return
        if value in seen:
            return
        seen.add(value)
        (titles if is_title else body).append(value)

    def walk(obj, key: str | None = None, depth: int = 0) -> None:
        if depth > 12:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k).lower() in _JSON_SKIP_KEYS:
                    continue
                walk(v, str(k).lower(), depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                walk(item, key, depth + 1)
        elif isinstance(obj, str):
            add(obj, is_title=(key in _JSON_TITLE_KEYS))
        elif isinstance(obj, (int, float)) and key:
            # numeros sueltos solo tienen sentido con su etiqueta (ej. "año: 2024")
            add(f"{key}: {obj}", is_title=False)

    walk(data)
    return "\n\n".join(titles + body)


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


def _extract_pbf(path: str | Path) -> str:
    """Extrae los atributos de un vector tile (.pbf) como pares 'clave: valor'.

    El corpus trae ~73 mapas. Cada tile contiene capas con elementos (municipios,
    zonas) y sus atributos. Un mismo elemento se repite en varios niveles de zoom,
    asi que se deduplican los valores para no inflar el indice (Seccion 2.1).

    Si no hay libreria de vector tiles instalada, se devuelve cadena vacia en vez
    de romper la indexacion: 73 de 1826 documentos no justifican tumbar el build.
    """
    try:
        import mapbox_vector_tile  # type: ignore
    except ImportError:
        return ""

    raw = Path(path).read_bytes()
    try:
        tile = mapbox_vector_tile.decode(raw)
    except Exception:
        return ""

    seen: set[str] = set()
    parts: list[str] = []
    for layer_name, layer in (tile or {}).items():
        for feature in layer.get("features", []):
            for key, value in (feature.get("properties") or {}).items():
                text = f"{key}: {value}".strip()
                if len(text) > 3 and text not in seen:
                    seen.add(text)
                    parts.append(text)
        if parts:
            parts.append(f"[capa: {layer_name}]")
    return "\n".join(parts)
