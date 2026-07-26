"""Smoke test del pipeline ML COMPLETO sobre el corpus proxy.

Construye el indice (modelo pequeno), corre el retriever y mide NDCG@10 + F1@3
sobre el eval semilla. Prueba que todo el plumbing funciona extremo a extremo.

    python scripts/smoke_pipeline.py

NOTA: el proxy tiene pocos chunks, asi que puede devolver <10 fragmentos; el
objetivo es validar el flujo y las metricas, no cumplir el esquema de 50 consultas.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_index import build                     # noqa: E402
from src.encoding.index import VectorStore                # noqa: E402
from src.encoding.encoders import build_encoder, resolve_device  # noqa: E402
from src.retrieval.pipeline import Retriever              # noqa: E402
from src.eval.harness import evaluate                     # noqa: E402
from src.utils.io import read_jsonl                       # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.smoke.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = resolve_device("auto")  # cuda si hay GPU, si no cpu
    print(f"[smoke] device = {device}")

    print("=== 1) Construyendo indice sobre el proxy ===")
    build(cfg)

    print("\n=== 2) Cargando retriever ===")
    base = Path(cfg["paths"]["entrega"]) / "base_vectorial"
    stores, encoders = {}, {}
    docid_to_fuente: dict[str, str] = {}
    chunk_fuentes: dict[str, str] = {}   # chunk_id -> fuente (unico por chunk)
    for enc_cfg in cfg["encoders"]:
        name = enc_cfg["name"]
        store = VectorStore.load(base / f"encoder_{name}")
        stores[name] = store
        encoders[name] = build_encoder(enc_cfg, device=device)
        for m in store.metadata:
            docid_to_fuente.setdefault(m["doc_id"], m["fuente"])
            chunk_fuentes.setdefault(m["chunk_id"], m["fuente"])
    retriever = Retriever(stores=stores, encoders=encoders, cfg=cfg)

    print("\n=== 3) Consulta de ejemplo ===")
    demo = retriever.retrieve("q001", "dilemas eticos de las armas autonomas")
    print(f"  docs top-3: {[d.doc_id for d in demo.documents]}")
    print(f"  fragmentos: {len(demo.fragments)}")
    if demo.fragments:
        print(f"  frag#1 ({demo.fragments[0].doc_id}): {demo.fragments[0].text[:90]}...")

    print("\n=== 4) Metricas sobre eval semilla ===")
    eval_set = read_jsonl("eval_interno/eval.jsonl")
    metrics = evaluate(eval_set, retriever.retrieve, docid_to_fuente,
                       list(chunk_fuentes.values()), k=10)
    print(f"  NDCG@10 medio: {metrics['mean_ndcg@10']:.4f}")
    print(f"  F1@3 medio:    {metrics['mean_f1@3']:.4f}")
    print(f"  ({metrics['n_queries']} consultas)")
    for pq in metrics["per_query"]:
        print(f"    {pq['query_id']}: ndcg={pq['ndcg@10']:.3f} f1={pq['f1@3']:.3f}")

    print("\n[smoke_pipeline] OK - plumbing ML validado")


if __name__ == "__main__":
    main()
