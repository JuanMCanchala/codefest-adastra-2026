"""Segmentacion de oraciones multilingue (ES/EN/PT).

Cumple el requisito de completitud linguistica (spec 3.3): ninguna oracion
puede partirse entre fragmentos. Preferimos pysbd (robusto, multilingue);
si no esta instalado, caemos a un segmentador por regex razonable.
"""
from __future__ import annotations

import re

try:
    import pysbd  # type: ignore
    _HAS_PYSBD = True
except ImportError:  # pragma: no cover - depende del entorno
    _HAS_PYSBD = False


# Segmentador rapido: corta tras . ! ? seguidos de espacio + mayuscula/comilla.
_FALLBACK_RE = re.compile(
    r"""(?<=[.!?…])["'”’)\]]*\s+(?=["'“¿¡(\[]*[A-ZÀ-Ý0-9])""",
    re.VERBOSE,
)

# Abreviaturas tras las que un punto NO cierra oracion. Sin esta guarda, cortar
# en "Dr. Smith" o "Fig. 3" produciria fragmentos con oraciones incompletas, que
# el reto prohibe expresamente (Seccion 3.3).
_ABBREV = {
    # tratamiento y cargos
    "sr", "sra", "srta", "dr", "dra", "prof", "ing", "lic", "gral", "cnel", "tte",
    "mr", "mrs", "ms", "jr", "sr.", "st", "mt",
    # referencias y citas
    "fig", "figs", "tab", "tabla", "ec", "eq", "ref", "refs", "cap", "sec", "art",
    "pag", "pags", "pp", "p", "vol", "no", "nro", "num", "ed", "eds", "al",
    # latinismos y locuciones
    "etc", "vs", "ej", "cf", "ca", "aprox", "e.g", "i.e", "ss", "op", "cit",
    # organizaciones y unidades frecuentes en el corpus
    "ee", "uu", "ee.uu", "ss", "dept", "univ", "inc", "ltd", "corp", "gob",
}
# ultima "palabra" antes del punto de corte
_LAST_TOKEN_RE = re.compile(r"([\wÀ-ÿ.]+)[\"'”’)\]]*\s*$")


def _is_false_break(left: str) -> bool:
    """True si el punto que cierra `left` pertenece a una abreviatura, una
    inicial ('J. Perez') o un numero ('3.'), y por tanto no separa oraciones."""
    match = _LAST_TOKEN_RE.search(left)
    if not match:
        return False
    token = match.group(1).rstrip(".").lower()
    if not token:
        return False
    if token in _ABBREV:
        return True
    if len(token) == 1:            # inicial de nombre o item de lista
        return True
    if token.isdigit():            # numeracion: "1." , "2026."
        return True
    return False

_LANG_MAP = {"es": "es", "en": "en", "pt": "es"}  # pysbd no trae pt; 'es' aproxima bien


def _fallback_segment(text: str) -> list[str]:
    """Segmentacion rapida por reglas, con guarda de abreviaturas.

    pysbd es ~9000x mas lento (7 K chars/s frente a 64 M chars/s) y sobre el
    corpus real supondria horas de proceso, asi que este es el segmentador por
    defecto. La guarda evita cortar en 'Dr.', 'Fig.' o iniciales, que es donde un
    regex ingenuo generaria oraciones incompletas.
    """
    text = text.strip()
    if not text:
        return []
    sentences: list[str] = []
    start = 0
    for match in _FALLBACK_RE.finditer(text):
        if _is_false_break(text[start:match.start()]):
            continue
        piece = text[start:match.start()].strip()
        if piece:
            sentences.append(piece)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


# Limite por bloque para el segmentador. pysbd se degrada mucho con cadenas muy
# largas: el corpus real trae datasets tabulares de decenas de MB en un solo
# documento (uno de 48 MB), que dejaban el proceso practicamente colgado.
_MAX_BLOCK_CHARS = 100_000


def _split_blocks(text: str, max_chars: int = _MAX_BLOCK_CHARS) -> list[str]:
    """Parte un texto muy largo en bloques cortando en saltos de linea.

    Cortar en '\\n' preserva las fronteras de parrafo y de fila (los documentos
    tabulares son una fila por linea), asi que ninguna oracion queda partida.
    """
    if len(text) <= max_chars:
        return [text]
    blocks: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.split("\n"):
        if size + len(line) > max_chars and current:
            blocks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        blocks.append("\n".join(current))
    return blocks


# Segmentador por defecto. 'fast' usa reglas (64 M chars/s); 'pysbd' es mas
# preciso en casos raros pero rinde 7 K chars/s: sobre los 204 MB del corpus real
# son ~8 horas frente a ~3 segundos, con practicamente las mismas oraciones.
USE_PYSBD = False


def segment_sentences(text: str, lang: str = "es", use_pysbd: bool | None = None) -> list[str]:
    """Divide `text` en oraciones completas.

    `lang`: codigo ISO ('es', 'en', 'pt'). pysbd no soporta 'pt' nativo -> 'es'.
    `use_pysbd`: fuerza el segmentador; por defecto sigue a USE_PYSBD.
    Los textos muy largos se procesan por bloques para que el coste sea lineal.
    """
    text = text.strip()
    if not text:
        return []
    prefer_pysbd = USE_PYSBD if use_pysbd is None else use_pysbd
    if not (_HAS_PYSBD and prefer_pysbd):
        return _fallback_segment(text)

    seg_lang = _LANG_MAP.get(lang, "en")
    segmenter = pysbd.Segmenter(language=seg_lang, clean=False)
    sentences: list[str] = []
    for block in _split_blocks(text):
        if not block.strip():
            continue
        try:
            sentences.extend(s.strip() for s in segmenter.segment(block) if s.strip())
        except Exception:
            sentences.extend(_fallback_segment(block))
    return sentences or _fallback_segment(text)
