"""Indice disperso (lexical) a partir de los pesos que produce BGE-M3.

BGE-M3 genera, EN LA MISMA PASADA que el vector denso, un conjunto de pesos
lexicos por token (`lexical_weights`). Esa señal captura coincidencia exacta de
terminos —siglas, nombres propios, tecnicismos: NBQR, RPO, GAO, GAOR, GDO— que
es justo donde la recuperacion densa es debil. Fusionar ambas señales con RRF es
la mejora mejor documentada en recuperacion multilingue.

Implementacion: indice invertido `token -> (indices, pesos)` respaldado por
arreglos numpy. Una consulta solo toca los chunks que comparten alguno de sus
tokens, y la puntuacion es un producto punto vectorizado.

Por que numpy y no listas de tuplas: con 85.000 fragmentos el indice tiene ~8,5
millones de entradas. Guardadas como tuplas de Python ocupaban 3,3 GB en memoria
(diez veces su tamaño en disco) y, sumadas al encoder y al reranker, agotaban la
RAM y el proceso moria sin traza. En arreglos int32/float32 son ~70 MB.

No interviene ningun modelo generativo: son pesos numericos y aritmetica.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


class SparseIndex:
    """Indice invertido de pesos lexicos respaldado por arreglos numpy."""

    def __init__(self) -> None:
        # token -> (indices de chunk int32, pesos float32)
        self.inverted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.chunk_ids: list[str] = []

    # ------------------------------------------------------------------ build
    def add(self, lexical_weights: list[dict], chunk_ids: list[str]) -> None:
        """Indexa los pesos lexicos de un lote de fragmentos."""
        assert len(lexical_weights) == len(chunk_ids), "pesos y chunk_ids desalineados"
        acc: dict[str, list[tuple[int, float]]] = defaultdict(list)
        # conservar lo ya indexado
        for token, (idxs, ws) in self.inverted.items():
            acc[token] = list(zip(idxs.tolist(), ws.tolist()))

        for weights, cid in zip(lexical_weights, chunk_ids):
            idx = len(self.chunk_ids)
            self.chunk_ids.append(cid)
            for token, weight in weights.items():
                w = float(weight)
                if w > 0:
                    acc[str(token)].append((idx, w))

        self.inverted = {
            token: (np.fromiter((i for i, _ in pairs), dtype=np.int32, count=len(pairs)),
                    np.fromiter((w for _, w in pairs), dtype=np.float32, count=len(pairs)))
            for token, pairs in acc.items()
        }

    # ----------------------------------------------------------------- search
    def search(self, query_weights: dict, top_k: int = 100) -> list[tuple[str, float]]:
        """Devuelve [(chunk_id, score)] ordenado por producto punto lexico."""
        if not self.chunk_ids:
            return []
        scores = np.zeros(len(self.chunk_ids), dtype=np.float32)
        tocado = False
        for token, q_weight in query_weights.items():
            qw = float(q_weight)
            if qw <= 0:
                continue
            entry = self.inverted.get(str(token))
            if entry is None:
                continue
            idxs, ws = entry
            np.add.at(scores, idxs, qw * ws)
            tocado = True
        if not tocado:
            return []

        k = min(top_k, len(scores))
        mejores = np.argpartition(-scores, k - 1)[:k]
        mejores = mejores[np.argsort(-scores[mejores])]
        return [(self.chunk_ids[i], float(scores[i])) for i in mejores if scores[i] > 0]

    @property
    def n_chunks(self) -> int:
        return len(self.chunk_ids)

    @property
    def n_tokens(self) -> int:
        return len(self.inverted)

    # ------------------------------------------------------------ persistence
    def save(self, out_dir: str | Path) -> None:
        """Persiste junto al indice FAISS del mismo encoder, en formato npz.

        Se guardan los arreglos concatenados mas los desplazamientos de cada
        token, de modo que cargar es leer bloques contiguos en vez de reconstruir
        millones de objetos de Python.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tokens = list(self.inverted.keys())
        offsets = np.zeros(len(tokens) + 1, dtype=np.int64)
        for i, token in enumerate(tokens):
            offsets[i + 1] = offsets[i] + len(self.inverted[token][0])
        if tokens:
            indices = np.concatenate([self.inverted[t][0] for t in tokens])
            weights = np.concatenate([self.inverted[t][1] for t in tokens])
        else:
            indices = np.array([], dtype=np.int32)
            weights = np.array([], dtype=np.float32)

        np.savez(out / "sparse_index.npz", indices=indices, weights=weights, offsets=offsets)
        meta = {"tokens": tokens, "chunk_ids": self.chunk_ids}
        with (out / "sparse_meta.json").open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(meta, fh, ensure_ascii=False)

    @classmethod
    def load(cls, in_dir: str | Path) -> "SparseIndex | None":
        """Carga el indice disperso si existe; None si el encoder no lo genero.

        Admite el formato antiguo (sparse_index.json) para no invalidar indices
        ya construidos.
        """
        path = Path(in_dir)
        npz, meta_path = path / "sparse_index.npz", path / "sparse_meta.json"
        if npz.exists() and meta_path.exists():
            data = np.load(npz)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            index = cls()
            index.chunk_ids = meta["chunk_ids"]
            indices, weights, offsets = data["indices"], data["weights"], data["offsets"]
            index.inverted = {
                token: (indices[offsets[i]:offsets[i + 1]], weights[offsets[i]:offsets[i + 1]])
                for i, token in enumerate(meta["tokens"])
            }
            return index

        legacy = path / "sparse_index.json"
        if not legacy.exists():
            return None
        payload = json.loads(legacy.read_text(encoding="utf-8"))
        index = cls()
        index.chunk_ids = payload["chunk_ids"]
        index.inverted = {
            token: (np.fromiter((int(i) for i, _ in entries), dtype=np.int32, count=len(entries)),
                    np.fromiter((float(w) for _, w in entries), dtype=np.float32, count=len(entries)))
            for token, entries in payload["inverted"].items()
        }
        return index
