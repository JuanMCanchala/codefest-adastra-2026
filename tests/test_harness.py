"""Regresion del arnes: NDCG@10 debe quedar en [0,1] aunque varios fragmentos
provengan del mismo documento relevante (bug detectado en el smoke)."""
from src.eval.harness import evaluate
from src.schema import QueryResult, DocResult, FragmentResult


def _fake_result(query_id, frag_docids, doc_ids):
    return QueryResult(
        query_id=query_id,
        documents=[DocResult(i + 1, d) for i, d in enumerate(doc_ids)],
        fragments=[FragmentResult(i + 1, f"c{i}", d, "t") for i, d in enumerate(frag_docids)],
    )


def test_ndcg_bounded_when_multiple_fragments_same_doc():
    # DOC-1 (fuente a.md) tiene 3 chunks, todos relevantes con grado 3.
    eval_set = [{
        "query_id": "q001",
        "text": "consulta",
        "relevantes": {"a.md": 3},
    }]
    docid_to_fuente = {"DOC-1": "a.md"}
    all_chunk_fuentes = ["a.md", "a.md", "a.md"]  # 3 chunks de la misma fuente

    # el retriever devuelve los 3 fragmentos del mismo doc relevante
    def retrieve_fn(qid, text):
        return _fake_result(qid, ["DOC-1", "DOC-1", "DOC-1"], ["DOC-1"])

    metrics = evaluate(eval_set, retrieve_fn, docid_to_fuente, all_chunk_fuentes, k=10)
    assert 0.0 <= metrics["mean_ndcg@10"] <= 1.0
    assert metrics["mean_ndcg@10"] == 1.0   # ranking perfecto


def test_f1_uses_fuente_not_docid():
    eval_set = [{"query_id": "q001", "text": "q", "relevantes": {"a.md": 3, "b.md": 2}}]
    docid_to_fuente = {"DOC-1": "a.md", "DOC-2": "b.md", "DOC-9": "z.md"}
    all_chunk_fuentes = ["a.md", "b.md", "z.md"]

    def retrieve_fn(qid, text):
        return _fake_result(qid, ["DOC-1"], ["DOC-1", "DOC-2", "DOC-9"])

    metrics = evaluate(eval_set, retrieve_fn, docid_to_fuente, all_chunk_fuentes, k=10)
    # 2 de 2 relevantes recuperados en top-3: P=2/3, R=2/min(2,3)=1 -> F1=0.8
    assert abs(metrics["mean_f1@3"] - 0.8) < 1e-9
