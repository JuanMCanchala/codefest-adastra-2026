"""eval_corpus.py - Evalua el indice YA construido sobre el corpus real.

A diferencia de compare_configs.py, no reindexa: carga la base vectorial
persistida y compara variantes que no requieren re-encoding (agregacion
chunk->documento y reranking), rankeandolas con Conteo de Borda.

    python scripts/eval_corpus.py --config config.corpus.yaml
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.encoding.index import VectorStore                # noqa: E402
from src.encoding.sparse import SparseIndex               # noqa: E402
from src.encoding.encoders import build_encoder, resolve_device  # noqa: E402
from src.retrieval.pipeline import Retriever              # noqa: E402
from src.eval.harness import evaluate                     # noqa: E402
from src.eval.metrics import borda_leaderboard            # noqa: E402
from src.utils.io import read_jsonl                       # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.corpus.yaml")
    ap.add_argument("--eval", default="eval_interno/eval_corpus.jsonl")
    ap.add_argument("--with-rerank", action="store_true",
                    help="incluye variantes con cross-encoder (lento en CPU)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = resolve_device("auto")
    print(f"[eval] device = {device}")

    base = Path(cfg["paths"]["entrega"]) / "base_vectorial"
    stores, encoders, sparse_indexes = {}, {}, {}
    docid_to_fuente, chunk_fuentes = {}, {}
    for enc_cfg in cfg["encoders"]:
        name = enc_cfg["name"]
        store = VectorStore.load(base / f"encoder_{name}")
        stores[name] = store
        encoders[name] = build_encoder(enc_cfg, device=device)
        sp = SparseIndex.load(base / f"encoder_{name}")
        if sp is not None:
            sparse_indexes[name] = sp
            print(f"[eval] indice disperso de {name}: {sp.n_chunks} chunks, {sp.n_tokens} tokens")
        for m in store.metadata:
            docid_to_fuente.setdefault(m["doc_id"], m["fuente"])
            chunk_fuentes.setdefault(m["chunk_id"], m["fuente"])
    chunk_fuentes = list(chunk_fuentes.values())

    n_chunks = sum(s.index.ntotal for s in stores.values())
    print(f"[eval] indice: {n_chunks} vectores, {len(docid_to_fuente)} documentos")

    eval_set = read_jsonl(args.eval)
    print(f"[eval] {len(eval_set)} consultas\n")

    reranker = None
    if args.with_rerank:
        from src.retrieval.rerank import CrossEncoderReranker
        print("[eval] cargando cross-encoder...")
        reranker = CrossEncoderReranker(cfg["rerank"]["model_id"], device=device)

    # (agregacion, rerank, usar_disperso). El eje 'disperso' es la pregunta
    # central: mide el aporte de la senal lexical de BGE-M3 sobre el denso solo.
    variants = [
        ("max_pool", False, False),   # A: denso solo (linea base actual)
        ("max_pool", False, True),    # B: hibrido denso+disperso
        ("sum", False, True),
        ("weighted_mean", False, True),
    ]
    if args.with_rerank:
        variants += [
            ("max_pool", True, False),   # C: denso + rerank
            ("max_pool", True, True),    # D: hibrido + rerank (todo)
        ]

    ndcg, f1 = {}, {}
    print("=== VARIANTES ===")
    for agg, use_rr, use_sparse in variants:
        vcfg = copy.deepcopy(cfg)
        vcfg["aggregation"]["method"] = agg
        vcfg["rerank"]["enabled"] = use_rr
        r = Retriever(stores, encoders, vcfg,
                      reranker=reranker if use_rr else None,
                      sparse_indexes=sparse_indexes if use_sparse else None)
        m = evaluate(eval_set, r.retrieve, docid_to_fuente, chunk_fuentes, k=10)
        label = f"{'hibrido' if use_sparse else 'denso'}-{agg}{'+rr' if use_rr else ''}"
        ndcg[label], f1[label] = m["mean_ndcg@10"], m["mean_f1@3"]
        print(f"  {label:26s} NDCG@10={m['mean_ndcg@10']:.4f}  F1@3={m['mean_f1@3']:.4f}")
        if agg == "max_pool" and not use_rr:
            for pq in m["per_query"]:
                print(f"      {pq['query_id']}  ndcg={pq['ndcg@10']:.3f}  f1={pq['f1@3']:.3f}")

    print("\n=== LEADERBOARD (Conteo de Borda) ===")
    for rank, (label, b) in enumerate(borda_leaderboard(ndcg, f1), 1):
        print(f"  {rank}. {label:22s} Borda={b}  NDCG={ndcg[label]:.4f}  F1={f1[label]:.4f}")


if __name__ == "__main__":
    main()
