"""Encoders multilingues (Seccion 4). Solo arquitecturas ENCODER (familia BERT);
prohibido cualquier decoder/generativo (spec 4.2 y 8.3).

Todos los modelos son de licencia permisiva (MIT/Apache) y multilingues nativos
ES/EN/PT, lo cual es imprescindible para la recuperacion cross-lingual del reto.

Los imports pesados son diferidos: importar este modulo no carga torch ni los
modelos, solo al instanciar/encode.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Encoder(ABC):
    """Interfaz comun. `encode` devuelve una matriz (n, d) float32 normalizada
    (para usar IndexFlatIP = coseno)."""

    name: str
    dim: int

    @abstractmethod
    def encode(self, texts: list[str], is_query: bool = False, batch_size: int = 32) -> np.ndarray:
        ...


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


_WARNED_CPU_BUILD = False


def resolve_device(device: str = "auto") -> str:
    """'auto' -> 'cuda' si hay GPU disponible, si no 'cpu'. Deja pasar valores
    explicitos ('cuda'/'cpu'). Permite el mismo codigo en smoke CPU y GPU real.

    Avisa si torch es un build de CPU habiendo GPU en la maquina: instalar
    paquetes que dependen de torch (docling, gliner...) puede sobrescribir en
    silencio la version CUDA por la de CPU de PyPI, y todo pasa a ir ~10x mas
    lento sin ningun error visible.
    """
    global _WARNED_CPU_BUILD
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    if not _WARNED_CPU_BUILD and torch.version.cuda is None:
        import shutil
        if shutil.which("nvidia-smi"):
            print(f"[AVISO] torch {torch.__version__} es un build de CPU pero hay GPU NVIDIA. "
                  f"Reinstalar con: pip install torch=={torch.__version__.split('+')[0]} "
                  f"--index-url https://download.pytorch.org/whl/cu126")
            _WARNED_CPU_BUILD = True
    return "cpu"


class BGEM3Encoder(Encoder):
    """BAAI/bge-m3 (MIT). Multilingue, denso + disperso (lexical) + ColBERT.

    En este reto usamos el vector denso para FAISS; el score disperso puede
    fusionarse aparte via RRF (Seccion 8.4) para ganar robustez lexica.
    """

    def __init__(self, model_id: str = "BAAI/bge-m3", device: str = "auto", use_fp16: bool = True):
        from FlagEmbedding import BGEM3FlagModel  # import diferido
        device = resolve_device(device)
        self.name = "bge-m3"
        self.model = BGEM3FlagModel(model_id, use_fp16=use_fp16 and device == "cuda", devices=device)
        self.dim = 1024

    def encode(self, texts: list[str], is_query: bool = False, batch_size: int = 32) -> np.ndarray:
        out = self.model.encode(texts, batch_size=batch_size, return_dense=True,
                                return_sparse=False, return_colbert_vecs=False)
        dense = np.asarray(out["dense_vecs"], dtype=np.float32)
        return _l2_normalize(dense)

    def encode_sparse(self, texts: list[str], batch_size: int = 32):
        """Pesos lexicos (lexical_weights) para un segundo ranking fusionable."""
        out = self.model.encode(texts, batch_size=batch_size, return_dense=False,
                                return_sparse=True, return_colbert_vecs=False)
        return out["lexical_weights"]


class STEncoder(Encoder):
    """Encoder generico basado en sentence-transformers. Cubre E5 (con prefijos
    'query:'/'passage:') y cualquier otro modelo encoder de HuggingFace."""

    def __init__(
        self,
        model_id: str,
        name: str | None = None,
        device: str = "auto",
        query_prefix: str = "",
        passage_prefix: str = "",
    ):
        from sentence_transformers import SentenceTransformer  # import diferido
        self.name = name or model_id.split("/")[-1]
        self.model = SentenceTransformer(model_id, device=resolve_device(device))
        self.dim = self.model.get_sentence_embedding_dimension()
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    def encode(self, texts: list[str], is_query: bool = False, batch_size: int = 32) -> np.ndarray:
        prefix = self.query_prefix if is_query else self.passage_prefix
        payload = [prefix + t for t in texts] if prefix else texts
        vecs = self.model.encode(payload, batch_size=batch_size, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False)
        return vecs.astype(np.float32)


def build_encoder(cfg: dict, device: str = "auto") -> Encoder:
    """Fabrica un encoder desde una entrada de config.yaml (encoders[])."""
    model_id = cfg["model_id"]
    if "bge-m3" in model_id:
        return BGEM3Encoder(model_id, device=device)
    return STEncoder(
        model_id,
        name=cfg.get("name"),
        device=device,
        query_prefix=cfg.get("query_prefix", ""),
        passage_prefix=cfg.get("passage_prefix", ""),
    )
