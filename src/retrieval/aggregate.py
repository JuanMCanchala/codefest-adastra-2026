"""Agregacion de fragmentos al nivel de documento (Seccion 8.6) para F1@3.

Dado el ranking de chunks (con su doc_id y score), agrupa por documento, calcula
una puntuacion agregada por doc y devuelve los top-N documentos.

OJO (spec 10.2.1): a nivel documento el emparejamiento del ground truth es por
`fuente`, no por doc_id. Aqui agregamos por doc_id interno; el mapeo doc_id ->
fuente se resuelve al escribir resultados (cada doc_id conoce su fuente).
"""
from __future__ import annotations

from collections import defaultdict


ScoredChunk = tuple[str, str, float]   # (chunk_id, doc_id, score)


def aggregate_documents(
    scored_chunks: list[ScoredChunk],
    method: str = "max_pool",
    top_n: int = 3,
    k_chunks: int = 30,
) -> list[tuple[str, float]]:
    """Agrupa chunks por doc_id y devuelve [(doc_id, score)] top_n.

    method:
      - max_pool:      score del mejor chunk del doc (robusto, default)
      - sum:           suma de scores de sus chunks recuperados
      - weighted_mean: media ponderada por posicion (chunks mejor rankeados pesan mas)
    """
    top = scored_chunks[:k_chunks]
    by_doc: dict[str, list[float]] = defaultdict(list)
    for _chunk_id, doc_id, score in top:
        by_doc[doc_id].append(score)

    doc_scores: dict[str, float] = {}
    for doc_id, scores in by_doc.items():
        if method == "max_pool":
            doc_scores[doc_id] = max(scores)
        elif method == "sum":
            doc_scores[doc_id] = sum(scores)
        elif method == "weighted_mean":
            weights = [1.0 / (i + 1) for i in range(len(scores))]
            doc_scores[doc_id] = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        else:
            raise ValueError(f"metodo de agregacion desconocido: {method!r}")

    ranked = sorted(doc_scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_n]
