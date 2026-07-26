"""Metricas de evaluacion, implementadas exactamente como la Seccion 10 del reto.

Sirven para el arnes de evaluacion interno: sin ground truth publico, medimos
NDCG@10 y F1@3 contra nuestro set sintetico y optimizamos hiperparametros.

Referencias del PDF:
  - NDCG@10 (fragmentos):  ecuaciones (8), (9), (10)
  - F1@3 (documentos):     ecuaciones (11), (12), (13), (14)
  - Conteo de Borda:       Seccion 11.2
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


# --------------------------------------------------------------------------- #
#  NDCG@k  (nivel fragmento)
# --------------------------------------------------------------------------- #
def dcg_at_k(relevances: Sequence[float], k: int = 10) -> float:
    """DCG@k = sum_{i=1}^{k} rel_i / log2(i + 1).

    `relevances` viene en el orden del ranking entregado por el equipo.
    La posicion i es 1-indexada (el primer elemento es i=1 -> log2(2)=1).
    """
    total = 0.0
    for idx, rel in enumerate(relevances[:k]):
        position = idx + 1
        total += rel / math.log2(position + 1)
    return total


def ndcg_at_k(
    ranked_relevances: Sequence[float],
    ideal_relevances: Sequence[float] | None = None,
    k: int = 10,
) -> float:
    """NDCG@k = DCG@k / IDCG@k, en el rango [0, 1].

    `ranked_relevances`: relevancias en el orden entregado por el sistema.
    `ideal_relevances`:  universo de relevancias disponibles para esa consulta.
                         Si es None, se asume el propio ranking (util en tests).
                         El IDCG usa las mejores k relevancias en orden optimo.
    """
    dcg = dcg_at_k(ranked_relevances, k)
    universe = list(ideal_relevances) if ideal_relevances is not None else list(ranked_relevances)
    ideal_order = sorted(universe, reverse=True)
    idcg = dcg_at_k(ideal_order, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def mean_ndcg_at_k(per_query_relevances: Iterable[Sequence[float]], k: int = 10) -> float:
    """Promedio de NDCG@k sobre todas las consultas (ecuacion 10).

    Cada elemento es el ranking de relevancias de una consulta. Se asume que
    cada lista ya representa el universo relevante de esa consulta.
    """
    values = [ndcg_at_k(rels, rels, k) for rels in per_query_relevances]
    if not values:
        return 0.0
    return sum(values) / len(values)


# --------------------------------------------------------------------------- #
#  F1@3  (nivel documento) — metrica de conjunto, no considera el orden
# --------------------------------------------------------------------------- #
def f1_at_3(retrieved_docs: Iterable[str], relevant_docs: Iterable[str]) -> float:
    """F1@3 para una consulta (ecuaciones 11-13).

    IMPORTANTE (spec 10.2.1): el emparejamiento a nivel documento se hace por el
    campo `fuente` (archivo original de ADL), NO por el doc_id arbitrario del
    equipo. Aqui los identificadores que se pasen deben ser esa clave estable.

    - P@3 = |D_hat & D*| / 3
    - R@3 = |D_hat & D*| / min(|D*|, 3)
    - F1@3 = 2*P*R / (P+R)
    """
    retrieved = list(dict.fromkeys(retrieved_docs))  # unico, preserva orden
    relevant = set(relevant_docs)
    if not relevant:
        return 0.0

    hits = len(set(retrieved) & relevant)
    precision = hits / 3.0
    recall = hits / min(len(relevant), 3)
    if precision + recall == 0.0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def mean_f1_at_3(
    per_query_retrieved: Iterable[Iterable[str]],
    per_query_relevant: Iterable[Iterable[str]],
) -> float:
    """Promedio de F1@3 sobre todas las consultas (ecuacion 14)."""
    scores = [
        f1_at_3(ret, rel)
        for ret, rel in zip(per_query_retrieved, per_query_relevant)
    ]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


# --------------------------------------------------------------------------- #
#  Conteo de Borda  (leaderboard unificado, Seccion 11.2)
# --------------------------------------------------------------------------- #
def _positions_from_scores(scores: dict[str, float]) -> dict[str, int]:
    """Convierte {equipo: score} en {equipo: posicion} (1 = mejor).

    Empates comparten la mejor posicion del grupo (posicion competitiva).
    """
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    positions: dict[str, int] = {}
    last_score: float | None = None
    last_position = 0
    for idx, (team, score) in enumerate(ordered):
        position = idx + 1
        if last_score is not None and score == last_score:
            position = last_position          # empate: misma posicion
        positions[team] = position
        last_score = score
        last_position = position
    return positions


def borda_points(scores: dict[str, float], n_teams: int | None = None) -> dict[str, int]:
    """Puntos de Borda por tabla: B = N - p  (spec 11.2).

    El mejor (p=1) obtiene N-1; el peor (p=N) obtiene 0.
    """
    n = n_teams if n_teams is not None else len(scores)
    positions = _positions_from_scores(scores)
    return {team: n - pos for team, pos in positions.items()}


def borda_leaderboard(
    ndcg_scores: dict[str, float],
    f1_scores: dict[str, float],
) -> list[tuple[str, int]]:
    """Leaderboard unificado: B_i = B_NDCG + B_F1 (ecuacion 15).

    Devuelve [(equipo, borda_total)] ordenado de mayor a menor.
    Desempate por NDCG@10 (spec 11.2.1).
    """
    teams = set(ndcg_scores) | set(f1_scores)
    n = len(teams)
    b_ndcg = borda_points(ndcg_scores, n)
    b_f1 = borda_points(f1_scores, n)

    totals = {t: b_ndcg.get(t, 0) + b_f1.get(t, 0) for t in teams}
    ranked = sorted(
        totals.items(),
        key=lambda kv: (kv[1], ndcg_scores.get(kv[0], 0.0)),
        reverse=True,
    )
    return ranked
