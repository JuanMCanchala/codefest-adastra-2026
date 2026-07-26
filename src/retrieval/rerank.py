"""Reranking con cross-encoder (TOGGLE - decision estrategica).

IMPORTANTE (zona gris del reglamento): la Seccion 8.3 prohibe modelos
GENERATIVOS (decoders: GPT, LLaMA, Gemini, Claude). Un cross-encoder como
BAAI/bge-reranker-v2-m3 es arquitectura ENCODER (familia BERT) que produce un
score de relevancia par (consulta, fragmento); NO genera texto. Argumentamos
que es admisible y se debe:
  1. mantener este paso desactivable via config (rerank.enabled),
  2. consultar al jurado antes de la entrega,
  3. justificar la arquitectura encoder en el informe tecnico.

Si el jurado lo veta -> rerank.enabled=false y quedamos con fusion RRF pura.
"""
from __future__ import annotations


class CrossEncoderReranker:
    def __init__(self, model_id: str = "BAAI/bge-reranker-v2-m3", device: str = "cuda", use_fp16: bool = True):
        from FlagEmbedding import FlagReranker  # import diferido
        self.model = FlagReranker(model_id, use_fp16=use_fp16, device=device)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],   # [(item_id, text), ...]
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """Devuelve [(item_id, score)] reordenado por relevancia del cross-encoder."""
        if not candidates:
            return []
        pairs = [[query, text] for _id, text in candidates]
        scores = self.model.compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = [scores]
        ranked = sorted(
            ((cid, float(s)) for (cid, _t), s in zip(candidates, scores)),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return ranked[:top_k] if top_k else ranked
