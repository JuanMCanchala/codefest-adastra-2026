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
