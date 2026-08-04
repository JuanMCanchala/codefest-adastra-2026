"""Pipeline de recuperacion end-to-end (Seccion 8).

Consulta -> encoder(s) -> FAISS -> fusion RRF -> [rerank cross-encoder] ->
  * top-10 fragmentos (<=250 palabras, oraciones completas)
  * top-3 documentos (agregacion chunk->doc)

Opera exclusivamente sobre vectores, scores de similitud y metadata. Sin
modelos generativos (spec 8.3).
"""
from __future__ import annotations

from .fusion import fuse
from .aggregate import aggregate_documents
from ..chunking.chunker import split_for_output
from ..schema import QueryResult, DocResult, FragmentResult


class Retriever:
    """Orquesta multiples VectorStore (uno por encoder) + encoders + reranker."""

    def __init__(self, stores: dict, encoders: dict, cfg: dict, reranker=None,
                 graph_retriever=None, sparse_indexes: dict | None = None):
        # stores:         {encoder_name: VectorStore}
        # encoders:       {encoder_name: Encoder}
        # sparse_indexes: {encoder_name: SparseIndex} (opcional, senal lexical)
        self.stores = stores
        self.encoders = encoders
        self.cfg = cfg
        self.reranker = reranker
        self.graph_retriever = graph_retriever   # GraphRetriever opcional (bonus)
        self.sparse_indexes = sparse_indexes or {}
        self._chunk_to_idx: dict[str, int] | None = None   # chunk_id -> id interno FAISS
        self._centroid_cache: dict[str, object] = {}

    def _search_one(self, name: str, query: str) -> list[tuple[str, float]]:
        """Ranking [(chunk_id, score)] de un encoder para la consulta."""
        enc = self.encoders[name]
        store = self.stores[name]
        qvec = enc.encode([query], is_query=True)
        scores, idxs = store.search(qvec, top_k=self.cfg["retrieval"]["top_k_faiss"])
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            meta = store.metadata[idx]
            out.append((meta["chunk_id"], float(score)))
        return out

    def _doc_centroid(self, doc_id: str, scored_chunks: list) -> "object | None":
        """Vector representativo de un documento: media de los vectores de sus
        fragmentos recuperados, tomados del propio indice FAISS."""
        import numpy as np

        if doc_id in self._centroid_cache:
            return self._centroid_cache[doc_id]
        store = next(iter(self.stores.values()), None)
        if store is None:
            return None
        if self._chunk_to_idx is None:
            self._chunk_to_idx = {m["chunk_id"]: i for i, m in enumerate(store.metadata)}

        vecs = []
        for cid, did, _score in scored_chunks:
            if did != doc_id:
                continue
            idx = self._chunk_to_idx.get(cid)
            if idx is None:
                continue
            try:
                vecs.append(store.index.reconstruct(int(idx)))
            except Exception:
                continue
            if len(vecs) >= 5:          # bastan unos pocos para caracterizarlo
                break
        if not vecs:
            self._centroid_cache[doc_id] = None
            return None
        c = np.mean(np.asarray(vecs, dtype=np.float32), axis=0)
        norm = float(np.linalg.norm(c))
        c = c / norm if norm else c
        self._centroid_cache[doc_id] = c
        return c

    def _select_diverse_docs(self, ranked_docs, scored_chunks, n_docs: int) -> list[str]:
        """Elige los n_docs mejores evitando documentos casi identicos.

        El corpus contiene el mismo informe en varios idiomas y ediciones (por
        ejemplo el Global Counterspace Report de SWF en español 2025, español
        2026 y portugues 2026). Sin este filtro, 14 de las 50 consultas gastaban
        sus tres huecos en versiones del mismo documento, lo que limita el F1@3 a
        un unico acierto posible. Los nombres de archivo no bastan para
        detectarlo; se comparan los vectores, que al ser cross-lingual acercan
        las traducciones de un mismo texto.
        """
        import numpy as np

        umbral = self.cfg.get("retrieval", {}).get("doc_dedup_threshold", 0.92)
        elegidos: list[str] = []
        centroides: list = []
        descartados: list[str] = []

        for doc_id, _score in ranked_docs:
            if len(elegidos) >= n_docs:
                break
            c = self._doc_centroid(doc_id, scored_chunks)
            if c is not None and centroides:
                if max(float(np.dot(c, o)) for o in centroides) >= umbral:
                    descartados.append(doc_id)
                    continue
            elegidos.append(doc_id)
            if c is not None:
                centroides.append(c)

        # si el filtro dejo huecos, completar con los descartados por orden
        for doc_id in descartados:
            if len(elegidos) >= n_docs:
                break
            elegidos.append(doc_id)
        return elegidos[:n_docs]

    def retrieve(self, query_id: str, query: str) -> QueryResult:
        rcfg = self.cfg["retrieval"]

        # 1) ranking por encoder + fusion RRF (Seccion 8.4)
        rankings = [self._search_one(name, query) for name in self.encoders]

        # 1a) senal lexical (dispersa) del mismo encoder: recupera coincidencias
        # exactas de siglas y nombres propios que el vector denso diluye.
        for name, sparse in self.sparse_indexes.items():
            enc = self.encoders.get(name)
            if sparse is None or enc is None or not hasattr(enc, "encode_sparse"):
                continue
            q_weights = enc.encode_sparse([query])[0]
            sparse_ranking = sparse.search(q_weights, top_k=rcfg["top_k_faiss"])
            if sparse_ranking:
                rankings.append(sparse_ranking)

        # 1b) grafo de conocimiento como indice adicional (Seccion 8.5, bonus)
        gcfg = self.cfg.get("graph", {})
        if self.graph_retriever and gcfg.get("fuse_into_retrieval"):
            graph_ranking = self.graph_retriever.retrieve(query)
            if graph_ranking:
                rankings.append(graph_ranking)

        fused = fuse(rankings, method=rcfg["fusion"], rrf_k=rcfg["rrf_k"])

        # indice chunk_id -> metadata (del primer store; los chunk_id son globales)
        meta_by_id = {}
        for store in self.stores.values():
            for m in store.metadata:
                meta_by_id.setdefault(m["chunk_id"], m)

        # 2) rerank cross-encoder opcional (toggle, zona gris - ver rerank.py)
        if self.reranker and self.cfg["rerank"]["enabled"]:
            top = fused[: self.cfg["rerank"]["top_k_candidates"]]
            candidates = [(cid, meta_by_id[cid]["texto"]) for cid, _ in top if cid in meta_by_id]
            fused = self.reranker.rerank(query, candidates)

        # 3) fragmentos: top-10 respetando <=250 palabras y oraciones completas.
        # Se descartan textos repetidos: el corpus contiene documentos casi
        # identicos (por ejemplo, el mismo informe en varios idiomas o ediciones),
        # asi que sin deduplicar hasta 3 de los 10 huecos se gastaban en texto
        # duplicado, que no puede aportar relevancia nueva al NDCG.
        # Los fragmentos demasiado cortos ('3 (2016).', 'counter space weapons.')
        # son notas al pie o restos de encabezado: ocupan un hueco del top-10 sin
        # poder aportar relevancia. Se posponen y solo se usan si no hay
        # suficientes fragmentos sustanciales para completar los 10 exigidos.
        min_words = rcfg.get("min_fragment_words", 20)
        n_target = rcfg["final_fragments"]
        fragments: list[FragmentResult] = []
        reserva: list[tuple[str, str, str]] = []
        seen_texts: set[str] = set()

        for chunk_id, _score in fused:
            if chunk_id not in meta_by_id:
                continue
            meta = meta_by_id[chunk_id]
            for sub in split_for_output(meta["texto"], lang=meta.get("idioma", "es"),
                                        max_words=self.cfg["chunking"]["output_max_words"]):
                key = " ".join(sub.split()).lower()
                if not key or key in seen_texts:
                    continue
                seen_texts.add(key)
                if len(sub.split()) < min_words:
                    reserva.append((chunk_id, meta["doc_id"], sub))
                    continue
                fragments.append(FragmentResult(
                    rank=len(fragments) + 1, chunk_id=chunk_id,
                    doc_id=meta["doc_id"], text=sub,
                ))
                if len(fragments) >= n_target:
                    break
            if len(fragments) >= n_target:
                break

        for chunk_id, doc_id, sub in reserva:      # completar si faltan
            if len(fragments) >= n_target:
                break
            fragments.append(FragmentResult(
                rank=len(fragments) + 1, chunk_id=chunk_id, doc_id=doc_id, text=sub,
            ))

        # 4) documentos: agregacion chunk->doc (Seccion 8.6) top-3
        scored_chunks = [
            (cid, meta_by_id[cid]["doc_id"], score)
            for cid, score in fused if cid in meta_by_id
        ]
        acfg = self.cfg["aggregation"]
        n_docs = rcfg["final_documents"]
        # Se piden mas candidatos de los necesarios para poder descartar
        # duplicados sin quedarse corto.
        ranked_docs = aggregate_documents(
            scored_chunks, method=acfg["method"],
            top_n=n_docs * 6, k_chunks=acfg["k_chunks"],
        )
        elegidos = self._select_diverse_docs(ranked_docs, scored_chunks, n_docs)
        documents = [DocResult(rank=i + 1, doc_id=doc_id)
                     for i, doc_id in enumerate(elegidos)]

        return QueryResult(query_id=query_id, documents=documents, fragments=fragments)
