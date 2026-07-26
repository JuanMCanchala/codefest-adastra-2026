"""Test del GraphRetriever con un grafo y un NER simulados (sin descargar GLiNER).

Verifica la logica de fusion grafo->retrieval (Seccion 8.5): match de entidades,
recuperacion de chunks del nodo + vecinos, y puntuacion por evidencia.
"""
import networkx as nx

from src.graph.retrieve import GraphRetriever


class FakeNER:
    """NER simulado: devuelve como entidades las palabras del query que
    aparezcan en una lista fija (evita cargar el modelo real en los tests)."""

    def __init__(self, known):
        self.known = known

    def predict_entities(self, text, types, threshold=0.4):
        low = text.lower()
        return [{"text": e, "label": "entidad"} for e in self.known if e.lower() in low]


def _build_graph():
    g = nx.DiGraph()
    g.add_node("Estados Unidos", tipo="pais", chunks=["c1", "c2"])
    g.add_node("armas autonomas", tipo="tecnologia", chunks=["c2", "c3"])
    g.add_node("Convenio de Ginebra", tipo="evento", chunks=["c4"])
    g.add_edge("Estados Unidos", "armas autonomas", peso=3)
    g.add_edge("armas autonomas", "Convenio de Ginebra", peso=1)
    return g


def test_graph_retrieves_chunks_of_matched_entity():
    g = _build_graph()
    ner = FakeNER(["Estados Unidos", "armas autonomas", "Convenio de Ginebra"])
    gr = GraphRetriever(g, ner, ["pais", "tecnologia", "evento"])
    res = dict(gr.retrieve("que hace Estados Unidos"))
    # c1 y c2 son chunks directos de 'Estados Unidos'
    assert "c1" in res and "c2" in res
    # via vecino 'armas autonomas' tambien aparece c3 (peso reducido)
    assert "c3" in res
    assert res["c1"] >= res["c3"]        # chunk directo pesa mas que via vecino


def test_graph_no_match_returns_empty():
    g = _build_graph()
    ner = FakeNER(["Estados Unidos"])
    gr = GraphRetriever(g, ner, ["pais"])
    assert gr.retrieve("una consulta sin entidades conocidas") == []


def test_graph_substring_match():
    g = _build_graph()
    ner = FakeNER(["Ginebra"])       # subcadena de 'Convenio de Ginebra'
    gr = GraphRetriever(g, ner, ["evento"])
    res = dict(gr.retrieve("el tratado de Ginebra"))
    assert "c4" in res               # match por subcadena
