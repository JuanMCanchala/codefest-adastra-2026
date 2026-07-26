"""sweep.py - Barrido de hiperparametros (el meta-move).

Sin ground truth publico, probamos configuraciones y las rankeamos con las
metricas exactas del reto (NDCG@10 + F1@3) fusionadas por Conteo de Borda.
La mejor configuracion es la que va a la entrega.

Varia:
  - chunking.index_max_tokens  (requiere reconstruir el indice: re-encoding)
  - aggregation.method         (barato: solo cambia la agregacion chunk->doc)

    python scripts/sweep.py --config config.bge.yaml

Ampliar la grilla (encoders, fusion, rerank on/off) cuando este el corpus real.
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
from src.eval.harness import evaluate                     # noqa: E402
from src.eval.metrics import borda_leaderboard            # noqa: E402
from src.utils.io import read_jsonl                       # noqa: E402

GRID_TOKENS = [64, 96, 128]
GRID_AGG = ["max_pool", "sum", "weighted_mean"]


def load_retriever(cfg, device):
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
    return Retriever(stores, encoders, cfg), docid_to_fuente, list(chunk_fuentes.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.bge.yaml")
    args = ap.parse_args()
    base_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = resolve_device("auto")
    eval_set = read_jsonl("eval_interno/eval.jsonl")

    ndcg_scores, f1_scores, rows = {}, {}, []
    for tok in GRID_TOKENS:
        cfg = copy.deepcopy(base_cfg)
        cfg["chunking"]["index_max_tokens"] = tok
        build(cfg)                                   # reconstruye indice para este chunk size
        retriever, d2f, chunk_fuentes = load_retriever(cfg, device)
        for agg in GRID_AGG:
            retriever.cfg["aggregation"]["method"] = agg
            m = evaluate(eval_set, retriever.retrieve, d2f, chunk_fuentes, k=10)
            label = f"tok{tok}-{agg}"
            ndcg_scores[label] = m["mean_ndcg@10"]
            f1_scores[label] = m["mean_f1@3"]
            rows.append((label, m["mean_ndcg@10"], m["mean_f1@3"]))
            print(f"  {label:22s} NDCG@10={m['mean_ndcg@10']:.4f}  F1@3={m['mean_f1@3']:.4f}")

    print("\n=== LEADERBOARD (Conteo de Borda, como el reto) ===")
    board = borda_leaderboard(ndcg_scores, f1_scores)
    for rank, (label, borda) in enumerate(board, 1):
        print(f"  {rank:2d}. {label:22s} Borda={borda:2d}  "
              f"NDCG={ndcg_scores[label]:.4f}  F1={f1_scores[label]:.4f}")
    print(f"\n[sweep] MEJOR CONFIG: {board[0][0]}")


if __name__ == "__main__":
    main()
