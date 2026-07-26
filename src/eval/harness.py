"""Arnes de evaluacion interno.

Sin ground truth publico, construimos nuestro propio set (queries sinteticas con
relevancia conocida) y medimos NDCG@10 + F1@3 EXACTAMENTE como el reto, para
optimizar hiperparametros (encoder, chunk, fusion, rerank, agregacion).

Formato del eval set (eval_interno/eval.jsonl), una linea por consulta:
  {
    "query_id": "q001",
    "text": "consulta en lenguaje natural",
    "idioma": "es",
    "fenomeno": 1,
    "relevantes": { "fuente/archivo_a.pdf": 3, "fuente/archivo_b.html": 1 }
  }
donde `relevantes` mapea `fuente` (clave real de emparejamiento, spec 10.2.1) a
un grado de relevancia (3=muy relevante ... 0). El nivel documento usa el
conjunto de fuentes con grado > 0.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

from .metrics import ndcg_at_k, f1_at_3


def evaluate(
    eval_set: list[dict],
    retrieve_fn,
    docid_to_fuente: dict[str, str],
    all_chunk_fuentes: Iterable[str],
    k: int = 10,
) -> dict:
    """Corre el retriever sobre el eval set y devuelve metricas agregadas.

    retrieve_fn(query_id, text) -> QueryResult
    docid_to_fuente:   mapea el doc_id interno del equipo al `fuente` original.
    all_chunk_fuentes: fuente de CADA chunk del indice (para el ranking ideal
                       de NDCG: cada fuente relevante aporta tantos fragmentos
                       relevantes como chunks tenga -> IDCG correcto, NDCG<=1).
    """
    fuente_counts = Counter(all_chunk_fuentes)
    ndcg_scores, f1_scores = [], []
    per_query = []

    for item in eval_set:
        rel = item["relevantes"]                       # {fuente: grado}
        relevant_fuentes = {f for f, g in rel.items() if g > 0}

        result = retrieve_fn(item["query_id"], item["text"])

        # --- NDCG@10 (fragmentos): grado por la fuente del doc del fragmento ---
        ranked_rel = []
        for frag in result.fragments:
            fuente = docid_to_fuente.get(frag.doc_id, "")
            ranked_rel.append(float(rel.get(fuente, 0)))
        # ranking ideal: todos los fragmentos relevantes del corpus para la query
        ideal_pool: list[float] = []
        for fuente, grado in rel.items():
            if grado > 0:
                ideal_pool.extend([float(grado)] * fuente_counts.get(fuente, 0))
        ndcg = ndcg_at_k(ranked_rel, ideal_pool, k=k)

        # --- F1@3 (documentos): emparejamiento por fuente ---
        retrieved_fuentes = [docid_to_fuente.get(d.doc_id, "") for d in result.documents]
        f1 = f1_at_3(retrieved_fuentes, relevant_fuentes)

        ndcg_scores.append(ndcg)
        f1_scores.append(f1)
        per_query.append({"query_id": item["query_id"], "ndcg@10": ndcg, "f1@3": f1})

    mean_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    return {
        "mean_ndcg@10": mean_ndcg,
        "mean_f1@3": mean_f1,
        "n_queries": len(eval_set),
        "per_query": per_query,
    }
