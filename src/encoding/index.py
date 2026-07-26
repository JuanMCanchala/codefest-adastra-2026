"""Indice vectorial FAISS (Seccion 5).

IndexFlatIP + vectores normalizados == similitud coseno exacta (ecuacion 4),
suficiente para el volumen del reto y sin perdida de exactitud.

FAISS solo guarda vectores + ids internos enteros; la metadata (Tabla 1) vive
en metadata.jsonl, donde la linea i corresponde al id interno i de FAISS
(spec: 'el orden de las lineas debe coincidir con los identificadores internos').
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..utils.io import write_jsonl, read_jsonl


class VectorStore:
    """Envuelve un indice FAISS + su metadata alineada por id interno."""

    def __init__(self, dim: int, index_type: str = "IndexFlatIP"):
        import faiss  # import diferido
        self.dim = dim
        self.index_type = index_type
        if index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(dim)
        elif index_type == "IndexHNSWFlat":
            self.index = faiss.IndexHNSWFlat(dim, 32)
        else:
            raise ValueError(f"index_type no soportado: {index_type}")
        self.metadata: list[dict] = []

    def add(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        """Anade vectores (n, d) ya normalizados y su metadata alineada."""
        assert vectors.shape[0] == len(metadata), "vectores y metadata desalineados"
        assert vectors.shape[1] == self.dim, "dimension incorrecta"
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self.metadata.extend(metadata)

    def search(self, query_vecs: np.ndarray, top_k: int = 100):
        """Devuelve (scores, indices) del top_k por producto interno (=coseno)."""
        q = np.ascontiguousarray(query_vecs, dtype=np.float32)
        scores, idxs = self.index.search(q, top_k)
        return scores, idxs

    def save(self, out_dir: str | Path) -> None:
        """Persiste index.faiss + metadata.jsonl en la carpeta del encoder."""
        import faiss
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out / "index.faiss"))
        write_jsonl(out / "metadata.jsonl", self.metadata)

    @classmethod
    def load(cls, in_dir: str | Path) -> "VectorStore":
        import faiss
        in_path = Path(in_dir)
        index = faiss.read_index(str(in_path / "index.faiss"))
        store = cls.__new__(cls)
        store.index = index
        store.dim = index.d
        store.index_type = type(index).__name__
        store.metadata = read_jsonl(in_path / "metadata.jsonl")
        return store
