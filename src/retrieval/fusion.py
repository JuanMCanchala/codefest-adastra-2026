"""Fusion de rankings de multiples indices/encoders (Seccion 8.4).

Implementa las tres estrategias del reto SIN modelos generativos:
  - CombSUM        (ecuacion 5)
  - CombMNZ        (ecuacion 6)
  - Reciprocal Rank Fusion / RRF  (ecuacion 7)   <- recomendada (robusta a escala)

Cada 'ranking' es una lista [(item_id, score), ...] ordenada de mayor a menor.
Los scores se asumen comparables dentro de un mismo indice (coseno).
"""
from __future__ import annotations

from collections import defaultdict


Ranking = list[tuple[str, float]]


def combsum(rankings: list[Ranking]) -> list[tuple[str, float]]:
    """s(c) = sum_j s_j(c). Un item ausente en un indice aporta 0."""
    acc: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for item, score in ranking:
            acc[item] += score
    return sorted(acc.items(), key=lambda kv: kv[1], reverse=True)


def combmnz(rankings: list[Ranking]) -> list[tuple[str, float]]:
    """s(c) = (sum_j s_j(c)) * |{j : s_j(c) > 0}|. Premia consistencia."""
    acc: dict[str, float] = defaultdict(float)
    hits: dict[str, int] = defaultdict(int)
    for ranking in rankings:
        for item, score in ranking:
            acc[item] += score
            if score > 0:
                hits[item] += 1
    fused = {item: total * hits[item] for item, total in acc.items()}
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


def rrf(rankings: list[Ranking], k0: int = 60) -> list[tuple[str, float]]:
    """s(c) = sum_j 1 / (k0 + r_j(c)), con r_j empezando en 1 (ecuacion 7).

    RRF ignora la magnitud del score y usa solo la posicion -> robusto a que
    distintos encoders tengan escalas de similitud diferentes.
    """
    acc: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank_idx, (item, _score) in enumerate(ranking):
            rank = rank_idx + 1
            acc[item] += 1.0 / (k0 + rank)
    return sorted(acc.items(), key=lambda kv: kv[1], reverse=True)


def fuse(rankings: list[Ranking], method: str = "rrf", rrf_k: int = 60) -> list[tuple[str, float]]:
    """Punto de entrada unico segun config.yaml (retrieval.fusion)."""
    if not rankings:
        return []
    if len(rankings) == 1:
        return list(rankings[0])
    if method == "combsum":
        return combsum(rankings)
    if method == "combmnz":
        return combmnz(rankings)
    if method == "rrf":
        return rrf(rankings, k0=rrf_k)
    raise ValueError(f"metodo de fusion desconocido: {method!r}")
