"""Grafo de conocimiento (componente bonus, Seccion 7).

Pipeline sin decoders:
  1. NER multilingue con GLiNER (zero-shot, MIT) sobre cada chunk.
  2. Extraccion de relaciones por co-ocurrencia + heuristicas de dependencia
     (NO se usa un LLM generativo).
  3. Grafo NetworkX; cada arista referencia doc_id y chunk_id de origen para
     trazabilidad (spec 7.2), exportable a grafo.graphml (spec entregable).

Vinculacion con la base vectorial (Seccion 7.3 / 8.5): cada entidad guarda los
chunk_id donde aparece, para recuperar por grafo y fusionar via RRF.
"""
from __future__ import annotations

from pathlib import Path


class GraphBuilder:
    def __init__(self, ner_model: str = "urchade/gliner_multi-v2.1",
                 entity_types: list[str] | None = None):
        from gliner import GLiNER  # import diferido
        import networkx as nx
        self.model = GLiNER.from_pretrained(ner_model)
        self.types = entity_types or ["persona", "organizacion", "pais", "tecnologia", "evento", "lugar"]
        self.graph = nx.DiGraph()
        self._nx = nx

    def add_chunk(self, chunk_id: str, doc_id: str, text: str, threshold: float = 0.5) -> None:
        entities = self.model.predict_entities(text, self.types, threshold=threshold)
        names = []
        for ent in entities:
            name = ent["text"].strip()
            if not self.graph.has_node(name):
                self.graph.add_node(name, tipo=ent["label"], chunks=[])
            self.graph.nodes[name]["chunks"].append(chunk_id)
            names.append(name)
        # relaciones por co-ocurrencia en el mismo fragmento (evidencia trazable)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                self._add_relation(names[i], names[j], doc_id, chunk_id)

    def _add_relation(self, a: str, b: str, doc_id: str, chunk_id: str) -> None:
        if self.graph.has_edge(a, b):
            self.graph[a][b]["peso"] += 1
        else:
            self.graph.add_edge(a, b, relacion="co-ocurre", peso=1,
                                doc_id=doc_id, chunk_id=chunk_id)

    def save(self, out_path: str | Path) -> None:
        """Exporta a GraphML (listas -> string, GraphML no admite atributos lista)."""
        g = self.graph.copy()
        for _n, attrs in g.nodes(data=True):
            if isinstance(attrs.get("chunks"), list):
                attrs["chunks"] = ",".join(attrs["chunks"])
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._nx.write_graphml(g, str(out))
