"""Tests de las metricas contra los ejemplos y definiciones del reto."""
import math

from src.eval.metrics import (
    dcg_at_k,
    ndcg_at_k,
    f1_at_3,
    borda_points,
    borda_leaderboard,
)


def test_dcg_first_position_no_discount():
    # posicion 1 -> log2(2) = 1 -> sin descuento
    assert dcg_at_k([3.0]) == 3.0


def test_dcg_known_values():
    # rel=[3,2,3] -> 3/1 + 2/log2(3) + 3/log2(4) = 3 + 1.2618... + 1.5
    expected = 3 / 1 + 2 / math.log2(3) + 3 / math.log2(4)
    assert dcg_at_k([3, 2, 3]) == expected


def test_ndcg_perfect_ranking_is_one():
    rels = [3, 2, 1, 0]
    assert ndcg_at_k(rels, rels) == 1.0


def test_ndcg_reversed_is_less_than_one():
    ideal = [3, 2, 1, 0]
    worst = [0, 1, 2, 3]
    score = ndcg_at_k(worst, ideal)
    assert 0.0 < score < 1.0


def test_ndcg_all_zero_is_zero():
    assert ndcg_at_k([0, 0, 0], [0, 0, 0]) == 0.0


def test_f1_perfect():
    # 3 recuperados, 3 relevantes, todos aciertan -> P=1, R=1, F1=1
    assert f1_at_3(["A", "B", "C"], ["A", "B", "C"]) == 1.0


def test_f1_partial():
    # 1 acierto de 3; relevantes=3 -> P=1/3, R=1/3, F1=1/3
    score = f1_at_3(["A", "X", "Y"], ["A", "B", "C"])
    assert math.isclose(score, 1 / 3)


def test_f1_recall_capped_at_min_relevant_3():
    # solo 2 relevantes existen; recuperamos ambos -> R = 2/min(2,3) = 1
    # P = 2/3 -> F1 = 2*(2/3)*1/((2/3)+1) = (4/3)/(5/3) = 0.8
    score = f1_at_3(["A", "B", "Z"], ["A", "B"])
    assert math.isclose(score, 0.8)


def test_f1_no_relevant_is_zero():
    assert f1_at_3(["A", "B", "C"], []) == 0.0


def test_borda_points_basic():
    # 4 equipos: mejor obtiene N-1=3, peor 0
    scores = {"e1": 0.9, "e2": 0.8, "e3": 0.7, "e4": 0.5}
    pts = borda_points(scores)
    assert pts == {"e1": 3, "e2": 2, "e3": 1, "e4": 0}


def test_borda_leaderboard_matches_pdf_example():
    # Tabla 3 del PDF (4 equipos):
    #   NDCG:  e1=0.72(pos2), e2=0.81(pos1), e3=0.65(pos3), e4=0.50(pos4)
    #   F1:    e1=0.68(pos1), e2=0.55(pos3), e3=0.62(pos2), e4=0.41(pos4)
    #   Borda total: e1=5, e2=4, e3=3, e4=0  -> e1 lidera
    ndcg = {"e1": 0.72, "e2": 0.81, "e3": 0.65, "e4": 0.50}
    f1 = {"e1": 0.68, "e2": 0.55, "e3": 0.62, "e4": 0.41}
    board = borda_leaderboard(ndcg, f1)
    totals = dict(board)
    assert totals["e1"] == 5
    assert totals["e2"] == 4
    assert totals["e3"] == 3
    assert totals["e4"] == 0
    assert board[0][0] == "e1"  # el equipo 1 lidera el leaderboard unificado
