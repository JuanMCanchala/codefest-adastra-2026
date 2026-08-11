"""Fusion de rankings de multiples indices/encoders (Seccion 8.4).

Implementa las tres estrategias del reto SIN modelos generativos:
  - CombSUM        (ecuacion 5)
  - CombMNZ        (ecuacion 6)
  - Reciprocal Rank Fusion / RRF  (ecuacion 7)   <- recomendada (robusta a escala)

Cada 'ranking' es una lista [(item_id, score), ...] ordenada de mayor a menor.
Los scores se asumen comparables dentro de un mismo indice (coseno).

Pesos por ranking
-----------------
Las tres formulas admiten un peso por indice. Sin el, RRF trata todos los
rankings como iguales: el primer elemento de cada uno aporta 1/(k0+1) venga de
donde venga. Eso es lo correcto entre indices que miden lo mismo (dos encoders
densos), pero no cuando uno mide algo distinto. Medido sobre las 50 consultas
reales, el ranking del grafo -que ordena por coocurrencia de entidades, no por
relevancia semantica- expulsaba del top-10 los tres mejores fragmentos de la
consulta q006. Un peso menor lo convierte en un matiz que puede reordenar
empates sin desplazar por si solo un acierto del recuperador denso.

Sigue sin intervenir ningun modelo generativo: son posiciones y aritmetica.
"""
from __future__ import annotations

from collections import defaultdict


Ranking = list[tuple[str, float]]


def _pesos(rankings: list[Ranking], weights: list[float] | None) -> list[float]:
    if weights is None:
        return [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(f"{len(weights)} pesos para {len(rankings)} rankings")
    return weights


def combsum(rankings: list[Ranking], weights: list[float] | None = None) -> list[tuple[str, float]]:
    """s(c) = sum_j w_j * s_j(c). Un item ausente en un indice aporta 0."""
    ws = _pesos(rankings, weights)
    acc: dict[str, float] = defaultdict(float)
    for ranking, w in zip(rankings, ws):
        for item, score in ranking:
            acc[item] += w * score
    return sorted(acc.items(), key=lambda kv: kv[1], reverse=True)


def combmnz(rankings: list[Ranking], weights: list[float] | None = None) -> list[tuple[str, float]]:
    """s(c) = (sum_j w_j * s_j(c)) * |{j : s_j(c) > 0}|. Premia consistencia."""
    ws = _pesos(rankings, weights)
    acc: dict[str, float] = defaultdict(float)
    hits: dict[str, int] = defaultdict(int)
    for ranking, w in zip(rankings, ws):
        for item, score in ranking:
            acc[item] += w * score
            if score > 0:
                hits[item] += 1
    fused = {item: total * hits[item] for item, total in acc.items()}
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


def rrf(rankings: list[Ranking], k0: int = 60,
        weights: list[float] | None = None) -> list[tuple[str, float]]:
    """s(c) = sum_j w_j / (k0 + r_j(c)), con r_j empezando en 1 (ecuacion 7).

    RRF ignora la magnitud del score y usa solo la posicion -> robusto a que
    distintos encoders tengan escalas de similitud diferentes. Con w_j = 1 para
    todos, es exactamente la ecuacion 7 del enunciado.
    """
    ws = _pesos(rankings, weights)
    acc: dict[str, float] = defaultdict(float)
    for ranking, w in zip(rankings, ws):
        for rank_idx, (item, _score) in enumerate(ranking):
            rank = rank_idx + 1
            acc[item] += w / (k0 + rank)
    return sorted(acc.items(), key=lambda kv: kv[1], reverse=True)


def fuse(rankings: list[Ranking], method: str = "rrf", rrf_k: int = 60,
         weights: list[float] | None = None) -> list[tuple[str, float]]:
    """Punto de entrada unico segun config.yaml (retrieval.fusion)."""
    if not rankings:
        return []
    if len(rankings) == 1:
        return list(rankings[0])
    if method == "combsum":
        return combsum(rankings, weights=weights)
    if method == "combmnz":
        return combmnz(rankings, weights=weights)
    if method == "rrf":
        return rrf(rankings, k0=rrf_k, weights=weights)
    raise ValueError(f"metodo de fusion desconocido: {method!r}")
