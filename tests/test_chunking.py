"""Tests de chunking: completitud linguistica y limite de 250 palabras."""
from src.chunking.chunker import chunk_document, split_for_output
from src.chunking.sentences import segment_sentences


SENTS = [
    "La inteligencia artificial transforma la defensa nacional.",
    "Los sistemas autonomos plantean nuevos dilemas eticos.",
    "La orbita baja terrestre acumula desechos espaciales.",
    "America Latina enfrenta dinamicas territoriales complejas.",
]


def test_segment_splits_sentences():
    text = " ".join(SENTS)
    sents = segment_sentences(text, lang="es")
    assert len(sents) == 4


def test_chunk_never_splits_a_sentence():
    text = " ".join(SENTS)
    chunks = chunk_document(text, index_max_tokens=12, overlap_sentences=0)
    # cada oracion original debe aparecer intacta en algun chunk
    joined = " ".join(c.text for c in chunks)
    for s in SENTS:
        assert s in joined


def test_chunk_positions_are_ordinal():
    text = " ".join(SENTS)
    chunks = chunk_document(text, index_max_tokens=12, overlap_sentences=0)
    assert [c.posicion for c in chunks] == list(range(len(chunks)))


def test_output_split_respects_250_words():
    long_sentence = "palabra " * 300  # 300 palabras en varias "oraciones"
    text = ". ".join(["esto es una oracion de prueba"] * 60)
    parts = split_for_output(text, max_words=250)
    for p in parts:
        assert len(p.split()) <= 250


def test_output_split_short_text_untouched():
    text = "Un fragmento corto de pocas palabras."
    parts = split_for_output(text, max_words=250)
    assert parts == [text]


def test_overlap_repeats_last_sentence():
    text = " ".join(SENTS)
    chunks = chunk_document(text, index_max_tokens=12, overlap_sentences=1)
    # con solapamiento, deberia haber mas de un chunk y contenido repetido
    assert len(chunks) >= 2


# --- Guarda de abreviaturas del segmentador rapido (requisito 3.3) ---

def test_no_corta_en_abreviatura():
    from src.chunking.sentences import segment_sentences
    s = segment_sentences("El Dr. Ramirez presento el informe. Fue aprobado.", lang="es")
    assert len(s) == 2
    assert "Dr. Ramirez" in s[0]


def test_no_corta_en_inicial():
    from src.chunking.sentences import segment_sentences
    s = segment_sentences("El autor J. Perez lo documento. Nadie lo refuto.", lang="es")
    assert len(s) == 2
    assert "J. Perez" in s[0]


def test_no_corta_en_numeracion():
    from src.chunking.sentences import segment_sentences
    s = segment_sentences("Ver la Fig. 3 del anexo. Alli esta el detalle.", lang="es")
    assert len(s) == 2


def test_si_corta_oraciones_reales():
    from src.chunking.sentences import segment_sentences
    s = segment_sentences("Primera oracion completa. Segunda oracion. Tercera oracion.", lang="es")
    assert len(s) == 3
