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
    Cada fila es una unidad de fragmentacion independiente.

    No todos los .csv del corpus son csv: AIINDEX_lit-covid-ai-covid-literature
    esta separado por tabuladores, y el lector estricto de pandas lo rechazaba con
    ParserError, dejando fuera del indice sus 15.115 registros. Ante un fallo se
    reintenta dejando que pandas deduzca el separador y descartando las filas
    malformadas: es preferible indexar un documento incompleto que perderlo.
    """
    import pandas as pd  # import diferido
    if fmt == "xlsx":
        df = pd.read_excel(path)
    else:
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")
    rows = []
    for _, row in df.iterrows():
        cells = [f"{col}: {val}" for col, val in row.items() if str(val).strip() and str(val) != "nan"]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


_OCR_READER = None


def _extract_ocr(path: str | Path) -> str:
    """OCR sobre imagenes con texto (infografias, capturas, graficos).

    Usa EasyOCR sobre GPU, el mismo motor ya medido en scripts/ocr_scanned.py:
    deteccion CRAFT + reconocimiento CRNN, modelos clasicos y NO generativos, asi
    que no entran en la restriccion de la Seccion 8.3. Se descarto PaddleOCR
    porque en Windows obliga a desactivar oneDNN y pasa de 6,5 s a 115 s por
    imagen.

    El lector se construye una sola vez por proceso: cargar los pesos cuesta
    varios segundos y hacerlo por imagen dominaria el tiempo total.
    """
    global _OCR_READER
    if _OCR_READER is None:
        import easyocr  # import diferido
        import torch
        _OCR_READER = easyocr.Reader(["es", "en"], gpu=torch.cuda.is_available(),
                                     verbose=False)
    try:
        lineas = _OCR_READER.readtext(str(path), detail=0, paragraph=True)
    except Exception:
        return ""
    return "\n".join(str(t) for t in lineas if t)


def _extract_pbf(path: str | Path) -> str:
    """Extrae los atributos de un .pbf como pares 'clave: valor'.

    Las FAQ del reto responden que .pbf es "el formato de OpenStreetMap
    (Protocolbuffer Binary Format)" y sugieren pyrosm/pyosmium. Los 73 .pbf de
    ESTE corpus no lo son: viven en una piramide de teselas
    (Amazon_Underworld/tiles/{z}/{x}/{y}.pbf) y sus bytes iniciales son
    `1a .. 0a 10 "au_compilado_R02"`, es decir campo 3 (layers) y campo 1 (name)
    del esquema Mapbox Vector Tile. Un .osm.pbf empieza por la longitud del
    BlobHeader seguida de la cadena "OSMHeader", que aqui no aparece. Decodificado
    como vector tile, un solo archivo entrega 251 elementos con atributos utiles
    (municipio, pais, poblacion y presencia de grupos armados), asi que se trata
    como vector tile y se deja OSM como respaldo por si algun archivo si lo fuera.

    Un mismo elemento se repite en varios niveles de zoom, asi que se deduplican
    los valores para no inflar el indice (Seccion 2.1). Si no hay libreria
    disponible se devuelve cadena vacia en vez de romper la indexacion: 73 de
    1826 documentos no justifican tumbar el build.
    """
    raw = Path(path).read_bytes()
    texto = _decode_vector_tile(raw)
    if not texto:
        texto = _decode_osm_pbf(path)
    return texto


def _decode_vector_tile(raw: bytes) -> str:
    try:
        import mapbox_vector_tile  # type: ignore
    except ImportError:
        return ""
    try:
        tile = mapbox_vector_tile.decode(raw)
    except Exception:
        return ""

    seen: set[str] = set()
    parts: list[str] = []
    for layer_name, layer in (tile or {}).items():
        for feature in layer.get("features", []):
            props = feature.get("properties") or {}
            campos = [f"{k}: {v}".strip() for k, v in props.items()
                      if str(v).strip() not in ("", "None", "nan")]
            if not campos:
                continue
            # Un registro por elemento, como en CSV/XLSX: cada municipio es una
            # unidad de fragmentacion con sentido propio. Aplanar los atributos
            # sueltos (lo que se hacia antes) dejaba un texto sin ninguna frontera
            # de oracion, y el documento entero acababa en un unico fragmento de
            # cientos de KB que el encoder truncaba a sus primeros 8192 tokens.
            registro = f"capa {layer_name} | " + " | ".join(campos) + "."
            # El mismo elemento se repite en varios niveles de zoom: se deduplica
            # por registro completo, no por atributo.
            if registro not in seen:
                seen.add(registro)
                parts.append(registro)
    return "\n".join(parts)


def _decode_osm_pbf(path: str | Path) -> str:
    """Respaldo para .pbf que si sean de OpenStreetMap: recoge las etiquetas
    (name, amenity, boundary...) de nodos, vias y relaciones."""
    try:
        import osmium  # type: ignore
    except ImportError:
        return ""

    seen: set[str] = set()
    parts: list[str] = []

    class _Handler(osmium.SimpleHandler):
        def _tags(self, obj) -> None:
            for tag in obj.tags:
                text = f"{tag.k}: {tag.v}".strip()
                if len(text) > 3 and text not in seen:
                    seen.add(text)
                    parts.append(text)

        node = way = relation = _tags

    try:
        _Handler().apply_file(str(path))
    except Exception:
        return ""
    return "\n".join(parts)
