"""compare_configs.py - Mide el aporte de multi-encoder y del reranker.

Construye el indice una vez (todos los encoders del config) y evalua 4 variantes
de recuperacion, rankeandolas con las metricas del reto + Conteo de Borda:

  A) bge-m3                (baseline denso)
  B) bge-m3 + e5           (fusion RRF de dos encoders)
  C) bge-m3 + reranker     (cross-encoder sobre baseline)
  D) bge-m3 + e5 + reranker (todo)

    python scripts/compare_configs.py --config config.multi.yaml
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_index import build                     # noqa: E402
from src.encoding.index import VectorStore                # noqa: E402
from src.encoding.encoders import build_encoder, resolve_device  # noqa: E402
from src.retrieval.pipeline import Retriever              # noqa: E402
from src.retrieval.rerank import CrossEncoderReranker     # noqa: E402
from src.eval.harness import evaluate                     # noqa: E402
from src.eval.metrics import borda_leaderboard            # noqa: E402
from src.utils.io import read_jsonl                       # noqa: E402


def make_retriever(cfg, stores, encoders, names, reranker, use_rerank):
    """Retriever con un subconjunto de encoders y rerank on/off."""
    sub_stores = {n: stores[n] for n in names}
    sub_encoders = {n: encoders[n] for n in names}
    variant_cfg = copy.deepcopy(cfg)
    variant_cfg["rerank"]["enabled"] = use_rerank
    return Retriever(sub_stores, sub_encoders, variant_cfg,
                     reranker=reranker if use_rerank else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.multi.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = resolve_device("auto")
    print(f"[compare] device = {device}")

    print("=== Construyendo indice (bge-m3 + e5) ===")
    build(cfg)

    base = Path(cfg["paths"]["entrega"]) / "base_vectorial"
    stores, encoders = {}, {}
    docid_to_fuente, chunk_fuentes = {}, {}
    for enc_cfg in cfg["encoders"]:
        name = enc_cfg["name"]
        store = VectorStore.load(base / f"encoder_{name}")
        stores[name] = store
        encoders[name] = build_encoder(enc_cfg, device=device)
        for m in store.metadata:
            docid_to_fuente.setdefault(m["doc_id"], m["fuente"])
            chunk_fuentes.setdefault(m["chunk_id"], m["fuente"])
    chunk_fuentes = list(chunk_fuentes.values())

    print("=== Cargando reranker (bge-reranker-v2-m3) ===")
    reranker = CrossEncoderReranker(cfg["rerank"]["model_id"], device=device)

    eval_set = read_jsonl("eval_interno/eval.jsonl")
    variants = {
        "A_bge":            (["bge-m3"], False),
        "B_bge+e5":         (["bge-m3", "e5-large"], False),
        "C_bge+rerank":     (["bge-m3"], True),
        "D_bge+e5+rerank":  (["bge-m3", "e5-large"], True),
    }

    ndcg, f1 = {}, {}
    print("\n=== RESULTADOS POR VARIANTE ===")
    for label, (names, use_rr) in variants.items():
        r = make_retriever(cfg, stores, encoders, names, reranker, use_rr)
        m = evaluate(eval_set, r.retrieve, docid_to_fuente, chunk_fuentes, k=10)
        ndcg[label], f1[label] = m["mean_ndcg@10"], m["mean_f1@3"]
        print(f"  {label:20s} NDCG@10={m['mean_ndcg@10']:.4f}  F1@3={m['mean_f1@3']:.4f}")

    print("\n=== LEADERBOARD (Conteo de Borda) ===")
    for rank, (label, b) in enumerate(borda_leaderboard(ndcg, f1), 1):
        print(f"  {rank}. {label:20s} Borda={b}  NDCG={ndcg[label]:.4f}  F1={f1[label]:.4f}")


if __name__ == "__main__":
    main()
