"""Indice disperso (lexical) a partir de los pesos que produce BGE-M3.

BGE-M3 genera, EN LA MISMA PASADA que el vector denso, un conjunto de pesos
lexicos por token (`lexical_weights`). Esa señal captura coincidencia exacta de
terminos —siglas, nombres propios, tecnicismos: LEO, Kessler, IADC, NDCG— que es
justo donde la recuperacion densa es debil. Fusionar ambas señales con RRF es la
mejora mejor documentada en recuperacion multilingue.

Implementacion: indice invertido `token -> [(idx_chunk, peso)]`. Una consulta
solo toca los chunks que comparten alguno de sus tokens, en lugar de compararse
contra el corpus entero. La puntuacion es el producto punto de los pesos sobre
los tokens compartidos, equivalente a `compute_lexical_matching_score` de
FlagEmbedding pero sin recorrer todos los pares.

No interviene ningun modelo generativo: son pesos numericos y aritmetica.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


class SparseIndex:
    """Indice invertido de pesos lexicos, con persistencia en JSON."""

    def __init__(self) -> None:
        # token -> [(indice_interno_del_chunk, peso)]
        self.inverted: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self.chunk_ids: list[str] = []

    def add(self, lexical_weights: list[dict], chunk_ids: list[str]) -> None:
        """Indexa los pesos lexicos de un lote de fragmentos.

        `lexical_weights[i]` es un dict {token: peso} del fragmento `chunk_ids[i]`.
        """
        assert len(lexical_weights) == len(chunk_ids), "pesos y chunk_ids desalineados"
        for weights, cid in zip(lexical_weights, chunk_ids):
            idx = len(self.chunk_ids)
            self.chunk_ids.append(cid)
            for token, weight in weights.items():
                w = float(weight)
                if w > 0:
                    self.inverted[str(token)].append((idx, w))

    def search(self, query_weights: dict, top_k: int = 100) -> list[tuple[str, float]]:
        """Devuelve [(chunk_id, score)] ordenado por producto punto lexico."""
        scores: dict[int, float] = defaultdict(float)
        for token, q_weight in query_weights.items():
            qw = float(q_weight)
            if qw <= 0:
                continue
            for idx, d_weight in self.inverted.get(str(token), ()):
                scores[idx] += qw * d_weight

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(self.chunk_ids[idx], score) for idx, score in ranked]

    @property
    def n_chunks(self) -> int:
        return len(self.chunk_ids)

    @property
    def n_tokens(self) -> int:
        return len(self.inverted)

    def save(self, out_dir: str | Path) -> None:
        """Persiste junto al indice FAISS del mismo encoder."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {"chunk_ids": self.chunk_ids, "inverted": {t: v for t, v in self.inverted.items()}}
        with (out / "sparse_index.json").open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, ensure_ascii=False)

    @classmethod
    def load(cls, in_dir: str | Path) -> "SparseIndex | None":
        """Carga el indice disperso si existe; None si el encoder no lo genero."""
        path = Path(in_dir) / "sparse_index.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        index = cls()
        index.chunk_ids = payload["chunk_ids"]
        index.inverted = defaultdict(list)
        for token, entries in payload["inverted"].items():
            index.inverted[token] = [(int(i), float(w)) for i, w in entries]
        return index
