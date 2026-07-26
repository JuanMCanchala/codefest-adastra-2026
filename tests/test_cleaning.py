"""Tests de la limpieza/normalizacion (Seccion 2.2)."""
from src.cleaning.normalize import clean_document, _detect_boilerplate


def test_removes_toc_dot_leaders():
    text = (
        "1. Introduccion\n"
        "1.1. Contexto del reto . . . . . . . . . . . . . . 3\n"
        "10.2.1. NDCG@10 para fragmentos . . . . . . . 21\n"
        "Este es contenido real que debe permanecer intacto.\n"
    )
    out = clean_document(text)
    assert "contenido real que debe permanecer" in out
    assert "NDCG@10 para fragmentos" not in out   # linea de indice eliminada
    assert "." * 5 not in out


def test_dehyphenates_line_breaks():
    text = "la conges-\ntion de la orbita baja terrestre es un problema."
    out = clean_document(text)
    assert "congestion" in out
    assert "conges- tion" not in out


def test_keeps_real_hyphenated_compounds():
    text = "el sistema space-debris orbital fue analizado."
    out = clean_document(text)
    assert "space-debris" in out          # no se une (no es guion de corte)


def test_detects_repeated_headers():
    lines = ["CODEFEST AD ASTRA 2026"] * 6 + ["contenido unico de una pagina"]
    bp = _detect_boilerplate(lines)
    assert "CODEFEST AD ASTRA 2026" in bp
    assert "contenido unico de una pagina" not in bp


def test_removes_control_chars_and_collapses_spaces():
    text = "texto   con\x00 caracteres\x07 de    control"
    out = clean_document(text)
    assert "\x00" not in out and "\x07" not in out
    assert "   " not in out
