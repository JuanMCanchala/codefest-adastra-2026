"""Tests de fusion (ecuaciones 5-7) y agregacion chunk->doc (Seccion 8.6)."""
import math

from src.retrieval.fusion import combsum, combmnz, rrf, fuse
from src.retrieval.aggregate import aggregate_documents


def test_combsum_adds_scores():
    r1 = [("a", 0.9), ("b", 0.5)]
    r2 = [("a", 0.7), ("c", 0.4)]
    fused = dict(combsum([r1, r2]))
    assert math.isclose(fused["a"], 1.6)
    assert math.isclose(fused["b"], 0.5)
    assert math.isclose(fused["c"], 0.4)


def test_combmnz_rewards_consistency():
    # 'a' aparece en ambos (mnz x2), 'b' solo en uno
    r1 = [("a", 0.5), ("b", 0.9)]
    r2 = [("a", 0.5)]
    fused = dict(combmnz([r1, r2]))
    assert math.isclose(fused["a"], (0.5 + 0.5) * 2)   # 2.0
    assert math.isclose(fused["b"], 0.9 * 1)            # 0.9
    # pese a que b tenia score alto, a gana por consistencia
    assert fused["a"] > fused["b"]


def test_rrf_uses_rank_not_score():
    # item con score bajo pero primer puesto en ambos indices gana
    r1 = [("x", 0.01), ("y", 0.99)]
    r2 = [("x", 0.02), ("y", 0.98)]
    fused = dict(rrf([r1, r2], k0=60))
    assert fused["x"] > fused["y"]  # x es rank 1 en ambos


def test_fuse_single_ranking_passthrough():
    r = [("a", 0.5), ("b", 0.4)]
    assert fuse([r], method="rrf") == r


def test_aggregate_max_pool():
    chunks = [
        ("c1", "DOC-A", 0.9),
        ("c2", "DOC-B", 0.8),
        ("c3", "DOC-A", 0.7),
        ("c4", "DOC-C", 0.3),
    ]
    docs = aggregate_documents(chunks, method="max_pool", top_n=3)
    assert docs[0] == ("DOC-A", 0.9)          # mejor chunk de A
    assert [d for d, _ in docs] == ["DOC-A", "DOC-B", "DOC-C"]


def test_aggregate_sum_can_reorder():
    chunks = [
        ("c1", "DOC-A", 0.6),
        ("c2", "DOC-B", 0.55),
        ("c3", "DOC-B", 0.5),
    ]
    docs = dict(aggregate_documents(chunks, method="sum", top_n=3))
    assert math.isclose(docs["DOC-B"], 1.05)  # 0.55 + 0.5 supera a A (0.6)
    assert docs["DOC-B"] > docs["DOC-A"]


def test_fragmentos_no_se_repiten_en_una_consulta():
    """El corpus tiene documentos casi identicos (mismo informe en varios idiomas
    o ediciones). Sin deduplicar, hasta 3 de los 10 huecos se gastaban en texto
    repetido, que no puede aportar relevancia nueva al NDCG@10."""
    from src.retrieval.pipeline import Retriever

    texto = "Un fragmento identico que aparece en dos documentos distintos."
    metadata = [
        {"chunk_id": "c1", "doc_id": "D1", "texto": texto, "fuente": "a.pdf"},
        {"chunk_id": "c2", "doc_id": "D2", "texto": texto, "fuente": "b.pdf"},
        {"chunk_id": "c3", "doc_id": "D3", "texto": "Un fragmento distinto.", "fuente": "c.pdf"},
    ]

    class FakeStore:
        def __init__(self, meta): self.metadata = meta
        def search(self, q, top_k): return [[1.0, 0.9, 0.8]], [[0, 1, 2]]

    class FakeEncoder:
        def encode(self, texts, is_query=False): return [[0.0]]

    cfg = {
        "retrieval": {"top_k_faiss": 10, "fusion": "rrf", "rrf_k": 60,
                      "final_fragments": 10, "final_documents": 3},
        "rerank": {"enabled": False},
        "chunking": {"output_max_words": 250},
        "aggregation": {"method": "max_pool", "k_chunks": 10},
        "graph": {},
    }
    r = Retriever({"e": FakeStore(metadata)}, {"e": FakeEncoder()}, cfg)
    res = r.retrieve("q001", "consulta")
    textos = [f.text for f in res.fragments]
    assert len(textos) == len(set(textos)), "hay fragmentos repetidos"
    assert len(textos) == 2, "deberia quedar 1 de los duplicados + el distinto"


# --------------------------------------------------------------------------- #
#  Pesos por ranking (correccion R8: el grafo no debe mandar solo)
# --------------------------------------------------------------------------- #
def test_rrf_sin_pesos_es_la_ecuacion_7():
    """Con pesos a 1 el resultado debe ser identico al RRF del enunciado."""
    from src.retrieval.fusion import rrf
    a = [("c1", 0.9), ("c2", 0.8)]
    b = [("c3", 0.7), ("c1", 0.6)]
    assert rrf([a, b]) == rrf([a, b], weights=[1.0, 1.0])


def test_ranking_atenuado_no_desplaza_al_de_peso_pleno():
    """El primero de un ranking atenuado no puede superar al primero de uno pleno."""
    from src.retrieval.fusion import rrf
    denso = [("bueno", 0.9)]
    grafo = [("popular", 5.0)]
    fusionado = dict(rrf([denso, grafo], weights=[1.0, 0.3]))
    assert fusionado["bueno"] > fusionado["popular"]
    # con peso pleno quedarian empatados, que es justo el problema medido
    empate = dict(rrf([denso, grafo], weights=[1.0, 1.0]))
    assert empate["bueno"] == empate["popular"]


def test_ranking_atenuado_si_rompe_empates():
    """Atenuado no es lo mismo que ignorado: sigue aportando evidencia."""
    from src.retrieval.fusion import rrf
    denso = [("a", 0.9), ("b", 0.9)]
    grafo = [("b", 1.0)]
    fusionado = dict(rrf([denso, grafo], weights=[1.0, 0.3]))
    assert fusionado["b"] > fusionado["a"]


def test_pesos_mal_dimensionados_fallan():
    from src.retrieval.fusion import rrf
    import pytest
    with pytest.raises(ValueError):
        rrf([[("c1", 1.0)], [("c2", 1.0)]], weights=[1.0])
