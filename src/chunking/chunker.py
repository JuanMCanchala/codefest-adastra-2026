"""Chunking de 2 niveles.

Nivel 1 (indice): agrupa oraciones completas hasta ~index_max_tokens tokens,
con solapamiento opcional. Es lo que se codifica y se guarda en FAISS.

Nivel 2 (salida): al construir resultados.jsonl, un chunk recuperado que exceda
250 palabras se divide en sub-fragmentos que respetan oraciones completas y el
limite (spec 9.2.1). Todos los sub-fragmentos conservan el chunk_id original
como clave de trazabilidad.

Nota sobre num_tokens: aqui aproximamos tokens por palabras (whitespace) para
no atar el chunking a un tokenizer concreto. En produccion, num_tokens se
recalcula con el tokenizer del encoder antes de indexar.
"""
from __future__ import annotations

from dataclasses import dataclass

from .sentences import segment_sentences


def approx_tokens(text: str) -> int:
    """Aproximacion barata de tokens (~= palabras). El valor real se calcula
    con el tokenizer del encoder en la fase de indexacion."""
    return len(text.split())


@dataclass
class RawChunk:
    text: str
    posicion: int          # indice ordinal dentro del documento (empieza en 0)
    num_tokens: int


def chunk_document(
    text: str,
    lang: str = "es",
    index_max_tokens: int = 384,
    overlap_sentences: int = 1,
) -> list[RawChunk]:
    """Nivel 1: divide un documento en chunks de oraciones completas.

    Nunca parte una oracion. Si una sola oracion supera index_max_tokens, se
    emite sola (se dividira despues en el nivel de salida si hace falta).
    """
    sentences = segment_sentences(text, lang=lang)
    chunks: list[RawChunk] = []
    if not sentences:
        return chunks

    current: list[str] = []
    current_tokens = 0
    position = 0

    def flush(sent_buffer: list[str]) -> None:
        nonlocal position
        if not sent_buffer:
            return
        chunk_text = " ".join(sent_buffer).strip()
        chunks.append(RawChunk(chunk_text, position, approx_tokens(chunk_text)))
        position += 1

    for sent in sentences:
        sent_tokens = approx_tokens(sent)
        if current and current_tokens + sent_tokens > index_max_tokens:
            flush(current)
            # solapamiento: arrancar el siguiente chunk con las ultimas N oraciones
            if overlap_sentences > 0:
                current = current[-overlap_sentences:]
                current_tokens = sum(approx_tokens(s) for s in current)
            else:
                current = []
                current_tokens = 0
        current.append(sent)
        current_tokens += sent_tokens

    flush(current)
    return chunks


def split_for_output(
    text: str,
    lang: str = "es",
    max_words: int = 250,
) -> list[str]:
    """Nivel 2: divide un texto en sub-fragmentos de <= max_words palabras,
    respetando oraciones completas (spec 9.2.1).

    Si una sola oracion supera max_words (caso raro), se corta por palabras
    como ultimo recurso para no violar el limite duro.
    """
    if len(text.split()) <= max_words:
        return [text.strip()]

    sentences = segment_sentences(text, lang=lang)
    out: list[str] = []
    current: list[str] = []
    current_words = 0

    for sent in sentences:
        n = len(sent.split())
        if n > max_words:
            # oracion gigante: emitir lo acumulado y trocear por palabras
            if current:
                out.append(" ".join(current).strip())
                current, current_words = [], 0
            words = sent.split()
            for i in range(0, len(words), max_words):
                out.append(" ".join(words[i:i + max_words]))
            continue
        if current and current_words + n > max_words:
            out.append(" ".join(current).strip())
            current, current_words = [], 0
        current.append(sent)
        current_words += n

    if current:
        out.append(" ".join(current).strip())
    return [c for c in out if c]
