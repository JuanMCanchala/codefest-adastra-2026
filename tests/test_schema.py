"""Tests del esquema estricto de resultados.jsonl (spec 9.3.2)."""
from src.schema import (
    validate_chunk_meta,
    validate_query_result,
    validate_resultados,
    QueryResult,
    DocResult,
    FragmentResult,
)


def _valid_query_obj(qid="q001"):
    return {
        "query_id": qid,
        "documents": [{"rank": i, "doc_id": f"DOC-{i:03d}"} for i in range(1, 4)],
        "fragments": [
            {"rank": i, "chunk_id": f"DOC-001-chunk-{i:03d}", "doc_id": "DOC-001",
             "text": "texto corto del fragmento"}
            for i in range(1, 11)
        ],
    }


def test_valid_query_passes():
    assert validate_query_result(_valid_query_obj()) == []


def test_wrong_number_of_documents_fails():
    obj = _valid_query_obj()
    obj["documents"] = obj["documents"][:2]
    errors = validate_query_result(obj)
    assert any("documents" in e for e in errors)


def test_wrong_number_of_fragments_fails():
    obj = _valid_query_obj()
    obj["fragments"] = obj["fragments"][:9]
    errors = validate_query_result(obj)
    assert any("fragments" in e for e in errors)


def test_fragment_over_250_words_fails():
    obj = _valid_query_obj()
    obj["fragments"][0]["text"] = "palabra " * 251
    errors = validate_query_result(obj)
    assert any("250" in e for e in errors)


def test_chunk_meta_missing_field():
    errors = validate_chunk_meta({"doc_id": "X"})
    assert any("fuente" in e for e in errors)


def test_chunk_meta_bad_fenomeno():
    obj = {
        "doc_id": "X", "chunk_id": "Y", "fuente": "f.pdf", "formato": "pdf",
        "fenomeno": 4, "posicion": 0, "num_tokens": 10, "texto": "t",
    }
    errors = validate_chunk_meta(obj)
    assert any("fenomeno" in e for e in errors)


def test_full_file_needs_50_lines_in_order():
    objs = [_valid_query_obj(f"q{i:03d}") for i in range(1, 51)]
    assert validate_resultados(objs) == []


def test_full_file_wrong_order_fails():
    objs = [_valid_query_obj(f"q{i:03d}") for i in range(1, 51)]
    objs[0]["query_id"] = "q999"
    errors = validate_resultados(objs)
    assert any("q001" in e for e in errors)


def test_query_result_dataclass_serializes():
    qr = QueryResult(
        query_id="q001",
        documents=[DocResult(1, "DOC-001"), DocResult(2, "DOC-002"), DocResult(3, "DOC-003")],
        fragments=[FragmentResult(i, f"c{i}", "DOC-001", "t") for i in range(1, 11)],
    )
    assert validate_query_result(qr.to_json()) == []
