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


# Fallback: corta tras . ! ? (y variantes) seguidos de espacio + mayuscula/comilla.
# No es perfecto, pero nunca deja oraciones a medias en el caso comun.
_FALLBACK_RE = re.compile(
    r"""(?<=[.!?…])["'”’)\]]*\s+(?=["'“¿¡(\[]*[A-ZÀ-Ý0-9])""",
    re.VERBOSE,
)

_LANG_MAP = {"es": "es", "en": "en", "pt": "es"}  # pysbd no trae pt; 'es' aproxima bien


def _fallback_segment(text: str) -> list[str]:
    parts = _FALLBACK_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def segment_sentences(text: str, lang: str = "es") -> list[str]:
    """Divide `text` en oraciones completas.

    `lang`: codigo ISO ('es', 'en', 'pt'). pysbd no soporta 'pt' nativo -> 'es'.
    """
    text = text.strip()
    if not text:
        return []
    if _HAS_PYSBD:
        seg_lang = _LANG_MAP.get(lang, "en")
        segmenter = pysbd.Segmenter(language=seg_lang, clean=False)
        sentences = [s.strip() for s in segmenter.segment(text) if s.strip()]
        return sentences or _fallback_segment(text)
    return _fallback_segment(text)
