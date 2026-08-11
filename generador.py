"""generador.py - Entregable 4 del reto.

Usa el indice FAISS, lee el archivo de consultas y genera resultados.jsonl.
Objetivo: REPRODUCIR los resultados (spec: si no se reproducen, se excluye).

Uso:
    python generador.py --config config.yaml \
        --queries data/queries.jsonl \
        --out entrega/resultados.jsonl

Determinismo: semillas fijas + orden de consultas q001..q050. Los encoders y el
indice FAISS producen los mismos vectores/rankings dada la misma entrada.
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import yaml

from src.utils.io import read_jsonl, write_jsonl
from src.schema import validate_resultados


def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def load_pipeline(cfg: dict):
    """Carga stores + encoders (+ reranker) segun config. Import diferido de ML."""
    from src.encoding.index import VectorStore
    from src.encoding.encoders import build_encoder, resolve_device
    from src.retrieval.pipeline import Retriever

    from src.encoding.sparse import SparseIndex

    device = resolve_device("auto")
    base = Path(cfg["paths"]["entrega"]) / "base_vectorial"
    stores, encoders, sparse_indexes = {}, {}, {}
    for enc_cfg in cfg["encoders"]:
        name = enc_cfg["name"]
        stores[name] = VectorStore.load(base / f"encoder_{name}")
        encoders[name] = build_encoder(enc_cfg, device=device)
        # Senal lexical del mismo encoder: decisiva en este corpus, donde las
        # consultas estan llenas de siglas (NBQR, RPO, GAO, GAOR, GDO) y nombres
        # propios que el vector denso diluye.
        sparse = SparseIndex.load(base / f"encoder_{name}")
        if sparse is not None:
            sparse_indexes[name] = sparse
            print(f"[generador] indice disperso {name}: {sparse.n_chunks} chunks")

    reranker = None
    if cfg["rerank"]["enabled"]:
        from src.retrieval.rerank import CrossEncoderReranker
        reranker = CrossEncoderReranker(cfg["rerank"]["model_id"], device=device)

    # Grafo de conocimiento (bonus, Seccion 8.5): se carga desde grafo.graphml y
    # se le adjunta el NER para extraer entidades de la consulta en recuperacion.
    graph_retriever = None
    gcfg = cfg.get("graph", {})
    if gcfg.get("enabled") and gcfg.get("fuse_into_retrieval"):
        graphml = base / "grafo" / "grafo.graphml"
        if graphml.exists():
            import networkx as nx
            from gliner import GLiNER
            from src.graph.retrieve import GraphRetriever
            graph = nx.read_graphml(str(graphml))
            ner = GLiNER.from_pretrained(gcfg["ner_model"])
            graph_retriever = GraphRetriever(graph, ner, gcfg["entity_types"])
            print(f"[generador] grafo integrado a la recuperacion: "
                  f"{graph.number_of_nodes():,} entidades, "
                  f"{graph.number_of_edges():,} relaciones")
        else:
            # El bono solo cuenta si el grafo participa en la recuperacion, asi
            # que perderlo por un archivo ausente no puede pasar desapercibido.
            print(f"[ADVERTENCIA] graph.enabled=true pero no existe {graphml}: "
                  f"se generaran resultados SIN el grafo")

    return Retriever(stores=stores, encoders=encoders, cfg=cfg,
                     reranker=reranker, graph_retriever=graph_retriever,
                     sparse_indexes=sparse_indexes)


def main() -> None:
    ap = argparse.ArgumentParser(description="Genera resultados.jsonl del reto")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--queries", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pausa", type=float, default=0.0,
                    help="segundos de espera entre consultas. Tres modelos "
                         "(encoder denso, cross-encoder y NER) comparten la GPU "
                         "sin descanso; en portatiles esto baja el pico termico "
                         "a cambio de unos minutos mas")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_determinism(cfg.get("seed", 42))

    queries_path = args.queries or cfg["paths"]["queries"]
    out_path = args.out or cfg["paths"]["resultados"]

    queries = read_jsonl(queries_path)  # [{query_id, text}, ...]
    retriever = load_pipeline(cfg)

    results = []
    for i, q in enumerate(queries, 1):
        qr = retriever.retrieve(q["query_id"], q["text"])
        results.append(qr.to_json())
        print(f"[generador] {i}/{len(queries)}  {q['query_id']}", flush=True)
        # El encoder denso y el cross-encoder comparten una GPU de 8 GB; liberar
        # la cache entre consultas evita que la memoria se acumule.
        if i % 5 == 0:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        if args.pausa:
            import time
            time.sleep(args.pausa)

    # Validacion estricta antes de escribir (spec 9.3.2)
    errors = validate_resultados(results, expected_lines=len(queries))
    if errors:
        print(f"[ADVERTENCIA] {len(errors)} problemas de esquema:")
        for e in errors[:20]:
            print("  -", e)

    n = write_jsonl(out_path, results)
    print(f"[OK] {n} lineas escritas en {out_path}")


if __name__ == "__main__":
    main()
