"""build_index.py - Construye la base vectorial (Fase de indexacion, Seccion 6).

Flujo: documentos -> extraccion -> limpieza -> chunking -> encoding -> FAISS
       + metadata.jsonl  (+ grafo.graphml opcional).

Uso:
    python scripts/build_index.py --config config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extraction.base import extract, detect_format  # noqa: E402
from src.cleaning.normalize import clean_document        # noqa: E402
from src.chunking.chunker import chunk_document          # noqa: E402
from src.schema import ChunkMeta                          # noqa: E402


def iter_corpus(raw_dir: Path):
    """Genera (path, doc_id, fuente, fenomeno). El fenomeno se infiere de la
    subcarpeta (fenomeno1/2/3) o de un manifiesto; ajustar al layout real de ADL."""
    for i, path in enumerate(sorted(raw_dir.rglob("*"))):
        if not path.is_file():
            continue
        try:
            detect_format(path)
        except ValueError:
            continue
        fenomeno = 0
        for n in (1, 2, 3):
            if f"fenomeno{n}" in str(path).lower() or f"phenomenon{n}" in str(path).lower():
                fenomeno = n
        doc_id = f"DOC-{i:04d}"
        # CLAVE (F1@3): forward slashes siempre, para emparejar `fuente` de forma
        # estable entre Windows/Linux y contra el ground truth.
        fuente = path.relative_to(raw_dir).as_posix()
        yield path, doc_id, fuente, fenomeno


def build(cfg: dict) -> None:
    from src.encoding.encoders import build_encoder
    from src.encoding.index import VectorStore

    raw_dir = Path(cfg["paths"]["corpus_raw"])
    chunk_cfg = cfg["chunking"]

    # 1) extraer + chunkear todo el corpus
    drop_bp = cfg.get("cleaning", {}).get("drop_boilerplate", True)
    all_chunks: list[ChunkMeta] = []
    for path, doc_id, fuente, fenomeno in iter_corpus(raw_dir):
        doc = extract(path, doc_id=doc_id, fuente=fuente, fenomeno=fenomeno)
        doc.text = clean_document(doc.text, drop_boilerplate=drop_bp)  # Seccion 2.2
        raw_chunks = chunk_document(
            doc.text, lang=doc.idioma or "es",
            index_max_tokens=chunk_cfg["index_max_tokens"],
            overlap_sentences=chunk_cfg["overlap_sentences"],
        )
        for rc in raw_chunks:
            all_chunks.append(ChunkMeta(
                doc_id=doc_id,
                chunk_id=f"{doc_id}-chunk-{rc.posicion:04d}",
                fuente=fuente, formato=doc.formato, fenomeno=fenomeno,
                posicion=rc.posicion, num_tokens=rc.num_tokens, texto=rc.text,
                idioma=doc.idioma,
            ))
    print(f"[build] {len(all_chunks)} chunks de {raw_dir}")

    texts = [c.texto for c in all_chunks]
    metadata = [c.to_json() for c in all_chunks]

    # 2) un indice FAISS por encoder (spec 4.4 / 5)
    base = Path(cfg["paths"]["entrega"]) / "base_vectorial"
    for enc_cfg in cfg["encoders"]:
        name = enc_cfg["name"]
        print(f"[build] encoding con {name} ({enc_cfg['model_id']}) ...")
        encoder = build_encoder(enc_cfg)
        vectors = encoder.encode(texts, is_query=False, batch_size=32)
        store = VectorStore(dim=encoder.dim, index_type=cfg["faiss"]["index_type"])
        store.add(vectors, metadata)
        store.save(base / f"encoder_{name}")
        print(f"[build] guardado {name}: {store.index.ntotal} vectores")

    # 3) grafo de conocimiento (bonus)
    if cfg.get("graph", {}).get("enabled"):
        from src.graph.build import GraphBuilder
        gb = GraphBuilder(cfg["graph"]["ner_model"], cfg["graph"]["entity_types"])
        for c in all_chunks:
            gb.add_chunk(c.chunk_id, c.doc_id, c.texto)
        gb.save(base / "grafo" / "grafo.graphml")
        print(f"[build] grafo: {gb.graph.number_of_nodes()} nodos, {gb.graph.number_of_edges()} aristas")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    build(cfg)


if __name__ == "__main__":
    main()
