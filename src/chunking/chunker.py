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


def _cap_sentences(sentences: list[str], max_tokens: int) -> list[str]:
    """Trocea por palabras las 'oraciones' que exceden el presupuesto del chunk.

    No es un caso teorico: los volcados de datos geoespaciales (.pbf) y algunos
    HTML mal formados llegan sin un solo punto, de modo que el segmentador
    devuelve el documento completo como una unica oracion. Sin este tope, ese
    documento se indexaba como un solo fragmento de cientos de KB del que el
    encoder solo veia sus primeros 8192 tokens: el resto quedaba fuera del
    indice sin aviso. Cortar por palabras degrada la lectura de ese fragmento,
    pero es preferible a perder el documento.
    """
    out: list[str] = []
    for sent in sentences:
        words = sent.split()
        if len(words) <= max_tokens:
            out.append(sent)
            continue
        for i in range(0, len(words), max_tokens):
            out.append(" ".join(words[i:i + max_tokens]))
    return out


def chunk_document(
    text: str,
    lang: str = "es",
    index_max_tokens: int = 384,
    overlap_sentences: int = 1,
) -> list[RawChunk]:
    """Nivel 1: divide un documento en chunks de oraciones completas.

    No parte oraciones salvo que una sola supere index_max_tokens, en cuyo caso
    se trocea por palabras para que ninguna quede fuera del indice.
    """
    sentences = _cap_sentences(segment_sentences(text, lang=lang), index_max_tokens)
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


def filter_min_chars(chunks: list[RawChunk], min_chars: int) -> list[RawChunk]:
    """Descarta fragmentos demasiado cortos, salvo que sean todo lo que hay.

    El umbral existe para que restos de encabezado no ocupen un hueco del top-10.
    Pero cuatro paginas del corpus (CENIA_fechas y companeras) son menus de
    navegacion cuyo unico texto util es el titulo, de unos 14 caracteres: al
    filtrarlos, el documento entero desaparecia del indice y ninguna consulta
    podria recuperarlo jamas. Un documento con un fragmento pobre es peor que uno
    bueno y mucho mejor que uno ausente.
    """
    filtrados = [c for c in chunks if len(c.text) >= min_chars]
    if not filtrados and chunks:
        return [max(chunks, key=lambda c: len(c.text))]
    return filtrados


# Fronteras secundarias, por orden de preferencia, para partir un texto que el
# segmentador entrego como una sola "oracion" larguisima. Casi siempre no es una
# oracion real, sino una lista de vinetas, una tabla o un bloque de referencias.
_FRONTERAS_SECUNDARIAS = ("•", "; ", ": ", " | ", " – ", " — ")


def _partir_larga(sent: str, max_words: int) -> list[str]:
    """Trocea una 'oracion' que excede el limite, en el punto menos malo.

    La Seccion 3.3 es un requisito obligatorio: ningun fragmento puede contener
    oraciones incompletas. Cortar por palabras lo incumple de forma flagrante
    -parte la frase a mitad-, asi que solo se recurre a ello si el texto no
    ofrece ninguna frontera aprovechable.
    """
    for marca in _FRONTERAS_SECUNDARIAS:
        if marca not in sent:
            continue
        piezas = [p.strip() for p in sent.split(marca) if p.strip()]
        if len(piezas) < 2:
            continue
        salida: list[str] = []
        actual: list[str] = []
        n = 0
        for pieza in piezas:
            k = len(pieza.split())
            if actual and n + k > max_words:
                salida.append(" ".join(actual))
                actual, n = [], 0
            actual.append(pieza)
            n += k
        if actual:
            salida.append(" ".join(actual))
        if all(len(s.split()) <= max_words for s in salida):
            return salida

    palabras = sent.split()   # ultimo recurso
    return [" ".join(palabras[i:i + max_words]) for i in range(0, len(palabras), max_words)]


def split_for_output(
    text: str,
    lang: str = "es",
    max_words: int = 250,
) -> list[str]:
    """Nivel 2: divide un texto en sub-fragmentos de <= max_words palabras,
    respetando oraciones completas (spec 9.2.1).

    Si una sola oracion supera max_words, se busca una frontera secundaria
    (vinetas, punto y coma, dos puntos) antes de recurrir al corte por palabras.
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
            # oracion gigante: emitir lo acumulado y buscarle una frontera
            if current:
                out.append(" ".join(current).strip())
                current, current_words = [], 0
            out.extend(_partir_larga(sent, max_words))
            continue
        if current and current_words + n > max_words:
            out.append(" ".join(current).strip())
            current, current_words = [], 0
        current.append(sent)
        current_words += n

    if current:
        out.append(" ".join(current).strip())
    return [c for c in out if c]
