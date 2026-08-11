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


# --------------------------------------------------------------------------- #
#  Garantias de cobertura del indice (correcciones C14 y C15)
# --------------------------------------------------------------------------- #
def test_documento_sin_puntos_no_queda_en_un_solo_fragmento():
    """Los volcados geoespaciales (.pbf) llegan sin un solo punto: el segmentador
    devuelve el documento entero como una 'oracion'. Sin tope, se indexaba como un
    unico fragmento del que el encoder solo veia sus primeros 8192 tokens."""
    from src.chunking.chunker import chunk_document
    texto = " ".join(f"campo{i}: valor{i}" for i in range(2000))   # ni un punto
    chunks = chunk_document(texto, index_max_tokens=256, overlap_sentences=0)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 256 for c in chunks)


def test_oracion_gigante_no_pierde_palabras():
    from src.chunking.chunker import chunk_document
    palabras = [f"p{i}" for i in range(1000)]
    chunks = chunk_document(" ".join(palabras), index_max_tokens=100, overlap_sentences=0)
    recuperadas = " ".join(c.text for c in chunks).split()
    assert recuperadas == palabras


def test_filtro_minimo_no_deja_un_documento_sin_fragmentos():
    """Cuatro paginas del corpus solo tienen titulo (13-31 caracteres). Filtrarlas
    borraba el documento del indice, y un documento ausente no se recupera nunca."""
    from src.chunking.chunker import RawChunk, filter_min_chars
    chunks = [RawChunk("Fechas - CENIA", 0, 2)]
    assert filter_min_chars(chunks, min_chars=40) == chunks


def test_filtro_minimo_si_descarta_cuando_hay_alternativa():
    from src.chunking.chunker import RawChunk, filter_min_chars
    largo = RawChunk("x" * 100, 1, 1)
    out = filter_min_chars([RawChunk("corto", 0, 1), largo], min_chars=40)
    assert out == [largo]


# --------------------------------------------------------------------------- #
#  Completitud linguistica (Seccion 3.3, requisito obligatorio)
# --------------------------------------------------------------------------- #
def test_lista_larga_se_parte_por_vinetas_no_por_palabras():
    """Una lista de vinetas llega como una sola 'oracion'. Partirla por palabras
    dejaria frases cortadas a la mitad, que la Seccion 3.3 prohibe."""
    from src.chunking.chunker import split_for_output
    items = [f"• Punto numero {i} con su texto explicativo correspondiente" for i in range(40)]
    partes = split_for_output(" ".join(items), max_words=60)
    assert all(len(p.split()) <= 60 for p in partes)
    # ninguna parte debe empezar a mitad de un item
    assert all(p.lstrip().startswith(("•", "Punto")) for p in partes)


def test_corte_por_palabras_solo_si_no_hay_frontera():
    from src.chunking.chunker import _partir_larga
    sin_fronteras = " ".join(f"palabra{i}" for i in range(300))
    partes = _partir_larga(sin_fronteras, max_words=100)
    assert len(partes) == 3
    assert " ".join(partes).split() == sin_fronteras.split()


def test_cierre_de_oracion_detecta_fragmentos_truncados():
    from src.retrieval.pipeline import _cierra_oracion
    assert _cierra_oracion("Esto es una oracion completa.")
    assert _cierra_oracion('Cita entre comillas."')
    assert _cierra_oracion("¿Una pregunta?")
    assert not _cierra_oracion("Esto quedo cortado a mitad de")
    assert not _cierra_oracion("• Promover")


def test_apertura_de_oracion_detecta_colas_de_frase():
    """Los PDF escaneados pierden la puntuacion en el OCR y el troceo cae a
    mitad de frase: 'ley se valen de bandas criminales...' es la cola de
    'grupos al margen de la ley se valen de...'."""
    from src.retrieval.pipeline import _abre_oracion, _bien_formado
    assert _abre_oracion("Una oracion normal.")
    assert _abre_oracion("¿Y una pregunta?")
    assert _abre_oracion("2024 fue el año del cambio.")
    assert _abre_oracion("«Una cita textual.»")
    assert not _abre_oracion("ley se valen de bandas criminales.")
    # bien formado exige ambas cosas
    assert _bien_formado("Abre y cierra correctamente.")
    assert not _bien_formado("Abre bien pero no cierra")
    assert not _bien_formado("cierra bien pero no abre.")
