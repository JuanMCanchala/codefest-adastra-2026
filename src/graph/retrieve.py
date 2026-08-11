"""Recuperacion via grafo de conocimiento y fusion con la base vectorial (Sec 8.5).

Flujo (sin modelos generativos):
  1. Se extraen las entidades de la consulta con el MISMO NER del grafo.
  2. Se localizan los nodos del grafo que coinciden con esas entidades.
  3. Se recuperan los chunks vinculados a esos nodos y a sus vecinos de primer
     orden (entidades directamente relacionadas).
  4. A cada chunk se le asigna una puntuacion segun cuantas relaciones relevantes
     lo respaldan. El ranking resultante se fusiona con FAISS via RRF, tratando
     el grafo como un indice adicional.
"""
from __future__ import annotations

from collections import defaultdict


class GraphRetriever:
    """Recupera chunks a partir del grafo. Reutiliza el modelo NER del builder."""

    def __init__(self, graph, ner_model, entity_types: list[str]):
        # graph: networkx.DiGraph con nodos {tipo, chunks:[...]} y aristas con peso
        self.graph = graph
        self.model = ner_model
        self.types = entity_types
        # indice auxiliar: nombre normalizado -> nodo original
        self._norm = {self._normalize(n): n for n in graph.nodes}

    @staticmethod
    def _normalize(name: str) -> str:
        return name.strip().lower()

    # Longitud minima para aceptar una coincidencia parcial. Por debajo, las
    # siglas cortas casan dentro de palabras sin ninguna relacion: medido sobre
    # las consultas reales, 'ia' coincide con 'colomb-ia' y 'cas' con cualquier
    # nodo que contenga esa secuencia.
    _MIN_PARCIAL = 4

    def _match_query_entities(self, query: str, threshold: float = 0.4) -> list[str]:
        """Nodos del grafo que coinciden con entidades de la consulta.

        Primero igualdad normalizada. Si no la hay, se admite coincidencia
        parcial —'orbita baja' contra 'orbita baja terrestre'— pero exigiendo que
        encaje en frontera de palabra y que la entidad tenga cuerpo suficiente,
        y eligiendo el nodo mas parecido en longitud en vez del primero que
        aparezca, que dependia del orden de insercion del grafo.
        """
        ents = self.model.predict_entities(query, self.types, threshold=threshold)
        matched: list[str] = []
        for ent in ents:
            key = self._normalize(ent["text"])
            if key in self._norm:
                matched.append(self._norm[key])
                continue
            if len(key) < self._MIN_PARCIAL:
                continue
            candidatos = [(abs(len(n) - len(key)), original)
                          for n, original in self._norm.items()
                          if self._encaja(key, n)]
            if candidatos:
                matched.append(min(candidatos)[1])
        return list(dict.fromkeys(matched))

    @staticmethod
    def _encaja(key: str, nombre: str) -> bool:
        """Contencion en frontera de palabra, en cualquiera de los dos sentidos."""
        corto, largo = (key, nombre) if len(key) <= len(nombre) else (nombre, key)
        if corto not in largo:
            return False
        i = largo.index(corto)
        antes_ok = i == 0 or not largo[i - 1].isalnum()
        j = i + len(corto)
        despues_ok = j == len(largo) or not largo[j].isalnum()
        return antes_ok and despues_ok

    def _chunks_of(self, node: str) -> list[str]:
        data = self.graph.nodes.get(node, {})
        chunks = data.get("chunks", [])
        if isinstance(chunks, str):           # tras cargar graphml, es cadena
            return [c for c in chunks.split(",") if c]
        return list(chunks)

    def retrieve(self, query: str, threshold: float = 0.4,
                 top_k: int = 100) -> list[tuple[str, float]]:
        """Devuelve [(chunk_id, score)] ordenado. score = nº de entidades/relaciones
        relevantes que respaldan el chunk (evidencia acumulada).

        El corte en top_k no es cosmetico. La fusion RRF solo mira posiciones, de
        modo que el primer elemento de cada ranking pesa lo mismo venga de donde
        venga. Los rankings denso y disperso llegan topados a top_k_faiss; sin un
        tope equivalente, una consulta con entidades muy conectadas ('colombia',
        'artificial intelligence') aportaba 10.174 candidatos y ponia una señal de
        popularidad a competir de igual a igual con la de relevancia semantica.
        """
        seeds = self._match_query_entities(query, threshold=threshold)
        if not seeds:
            return []

        chunk_score: dict[str, float] = defaultdict(float)
        for node in seeds:
            # chunks del propio nodo (peso pleno)
            for cid in self._chunks_of(node):
                chunk_score[cid] += 1.0
            # vecinos de primer orden (peso ponderado por la fuerza de la relacion)
            neighbors = set(self.graph.successors(node)) | set(self.graph.predecessors(node))
            for nb in neighbors:
                w = self._edge_weight(node, nb)
                for cid in self._chunks_of(nb):
                    chunk_score[cid] += 0.5 * w

        ordenado = sorted(chunk_score.items(), key=lambda kv: kv[1], reverse=True)
        return ordenado[:top_k]

    def _edge_weight(self, a: str, b: str) -> float:
        if self.graph.has_edge(a, b):
            return float(self.graph[a][b].get("peso", 1)) / 10.0 + 0.1
        if self.graph.has_edge(b, a):
            return float(self.graph[b][a].get("peso", 1)) / 10.0 + 0.1
        return 0.1
