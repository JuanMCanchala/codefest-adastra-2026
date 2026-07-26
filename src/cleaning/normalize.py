"""Limpieza y normalizacion del texto extraido (Seccion 2.2 del reto).

Elimina lo que no aporta significado y contamina el indice:
  - caracteres de control y espacios redundantes,
  - lineas de indice (TOC) con puntos de relleno '. . . . 21',
  - encabezados/pies de pagina repetidos en muchas paginas (boilerplate),
  - numeracion de paginas suelta.

Es una etapa separada entre extraccion y chunking (ver Figura 1 del PDF).
Sin esta limpieza, los puntos del indice rompen la segmentacion de oraciones y
generan fragmentos basura que bajan NDCG y F1.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

# 3+ puntos (con o sin espacios) = relleno de indice -> se colapsa
_DOT_LEADER = re.compile(r"(?:\s*\.\s*){3,}")
# linea que es solo ruido: puntos, espacios, digitos, guiones, bullets
_NOISE_ONLY = re.compile(r"^[\s.\-–—•·|0-9]*$")
# entrada de indice: '... texto ....... 21' (termina en numero de pagina)
_TOC_ENTRY = re.compile(r".+\.\s*\.\s*\.\s*.*\d+\s*$")
_MULTISPACE = re.compile(r"[ \t]{2,}")
# guion de corte de linea: letra + '-' al final de la linea
_HYPHEN_EOL = re.compile(r"[A-Za-zÀ-ÿ]-$")


def _fix_unicode(text: str) -> str:
    """Normaliza a NFC y elimina caracteres de control (menos \n y \t)."""
    text = unicodedata.normalize("NFC", text)
    return "".join(
        ch for ch in text
        if ch in ("\n", "\t") or not unicodedata.category(ch).startswith("C")
    )


def _is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if _NOISE_ONLY.match(stripped):
        return True
    if _DOT_LEADER.search(stripped) and _TOC_ENTRY.match(stripped):
        return True
    return False


def _detect_boilerplate(lines: list[str], min_repeats: int = 4, max_len: int = 70) -> set[str]:
    """Lineas cortas que se repiten muchas veces = encabezados/pies de pagina."""
    counts = Counter(l.strip() for l in lines if l.strip())
    return {
        line for line, n in counts.items()
        if n >= min_repeats and len(line) <= max_len
    }


def clean_document(text: str, drop_boilerplate: bool = True) -> str:
    """Devuelve el texto limpio y reflujado (lineas unidas dentro del parrafo).

    Reflujo: el PDF corta lineas a mitad de oracion; unimos con espacio para que
    el segmentador de oraciones funcione. Los saltos dobles marcan parrafos.
    """
    text = _fix_unicode(text)
    raw_lines = text.split("\n")

    boilerplate = _detect_boilerplate(raw_lines) if drop_boilerplate else set()

    kept: list[str] = []
    for line in raw_lines:
        stripped = line.strip()
        if stripped in boilerplate:
            continue
        if _is_noise_line(line):
            continue
        line = _DOT_LEADER.sub(" ", line)          # colapsa puntos de relleno
        line = _MULTISPACE.sub(" ", line).strip()
        if line:
            kept.append(line)

    # Reflujo con des-hifenacion: 'conges-\ntion' -> 'congestion' (guion de corte
    # de linea del PDF). Solo cuando el guion cierra la linea y la siguiente
    # empieza en minuscula (no toca compuestos reales como 'space-debris').
    parts: list[str] = []
    for line in kept:
        if parts and _HYPHEN_EOL.search(parts[-1]) and line[:1].islower():
            parts[-1] = parts[-1][:-1] + line
        else:
            parts.append(line)

    reflowed = " ".join(parts)
    reflowed = _MULTISPACE.sub(" ", reflowed).strip()
    return reflowed


def clean_pages(pages: list[str], drop_boilerplate: bool = True) -> str:
    """Variante cuando se dispone del texto por pagina (mejor deteccion de
    encabezados/pies). Une las paginas ya limpias."""
    joined = "\n".join(pages)
    return clean_document(joined, drop_boilerplate=drop_boilerplate)
