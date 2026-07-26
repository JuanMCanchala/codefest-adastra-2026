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

    def __init__(self, stores: dict, encoders: dict, cfg: dict, reranker=None):
        # stores:   {encoder_name: VectorStore}
        # encoders: {encoder_name: Encoder}
        self.stores = stores
        self.encoders = encoders
        self.cfg = cfg
        self.reranker = reranker

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

    def retrieve(self, query_id: str, query: str) -> QueryResult:
        rcfg = self.cfg["retrieval"]

        # 1) ranking por encoder + fusion RRF (Seccion 8.4)
        rankings = [self._search_one(name, query) for name in self.encoders]
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

        # 3) fragmentos: top-10 respetando <=250 palabras y oraciones completas
        fragments: list[FragmentResult] = []
        for chunk_id, _score in fused:
            if chunk_id not in meta_by_id:
                continue
            meta = meta_by_id[chunk_id]
            for sub in split_for_output(meta["texto"], lang=meta.get("idioma", "es"),
                                        max_words=self.cfg["chunking"]["output_max_words"]):
                fragments.append(FragmentResult(
                    rank=len(fragments) + 1,
                    chunk_id=chunk_id,            # chunk_id original (trazabilidad)
                    doc_id=meta["doc_id"],
                    text=sub,
                ))
                if len(fragments) >= rcfg["final_fragments"]:
                    break
            if len(fragments) >= rcfg["final_fragments"]:
                break

        # 4) documentos: agregacion chunk->doc (Seccion 8.6) top-3
        scored_chunks = [
            (cid, meta_by_id[cid]["doc_id"], score)
            for cid, score in fused if cid in meta_by_id
        ]
        acfg = self.cfg["aggregation"]
        doc_scores = aggregate_documents(
            scored_chunks, method=acfg["method"],
            top_n=rcfg["final_documents"], k_chunks=acfg["k_chunks"],
        )
        documents = [DocResult(rank=i + 1, doc_id=doc_id)
                     for i, (doc_id, _s) in enumerate(doc_scores)]

        return QueryResult(query_id=query_id, documents=documents, fragments=fragments)
