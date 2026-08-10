"""append_docs.py - Anade al indice los documentos que aun no estan en el.

Reconstruir la base vectorial completa cuesta horas de GPU y no aporta nada
cuando solo faltan unos pocos documentos (por ejemplo los 73 .pbf, cuya
extraccion fallaba en silencio). FAISS IndexFlatIP y el indice disperso admiten
anexado, y como los identificadores internos se asignan por orden de insercion,
anexar al final no altera ninguno de los ya existentes: la metadata, el indice
disperso y el grafo siguen alineados.

    python scripts/append_docs.py --config config.adl.yaml
    python scripts/append_docs.py --config config.adl.yaml --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_index import iter_corpus                      # noqa: E402
from src.chunking.chunker import chunk_document, filter_min_chars   # noqa: E402
from src.schema import ChunkMeta                          # noqa: E402


def chunks_faltantes(cfg: dict, ya_indexados: set[str]) -> list[ChunkMeta]:
    """Fragmenta los documentos del corpus que no aparecen en la metadata."""
    raw_dir = Path(cfg["paths"]["corpus_raw"])
    chunk_cfg = cfg["chunking"]
    cache_dir = Path(cfg["paths"].get("processed", "data/processed")) / "text"
    min_chars = cfg.get("cleaning", {}).get("min_chunk_chars", 0)
    max_per_doc = chunk_cfg.get("max_chunks_per_doc", 0)

    nuevos: list[ChunkMeta] = []
    for path, doc_id, fuente, fenomeno in iter_corpus(raw_dir, cfg["paths"].get("inventario")):
        if doc_id in ya_indexados:
            continue
        cached = cache_dir / f"{doc_id}.txt"
        if not cached.exists() or cached.stat().st_size == 0:
            print(f"[append] sin texto en cache, omitido: {doc_id}  {fuente[:60]}")
            continue
        texto = cached.read_text(encoding="utf-8", errors="ignore")
        raw_chunks = chunk_document(
            texto, lang="es",
            index_max_tokens=chunk_cfg["index_max_tokens"],
            overlap_sentences=chunk_cfg["overlap_sentences"],
        )
        raw_chunks = filter_min_chars(raw_chunks, min_chars)
        if max_per_doc and len(raw_chunks) > max_per_doc:
            raw_chunks = raw_chunks[:max_per_doc]
        for rc in raw_chunks:
            nuevos.append(ChunkMeta(
                doc_id=doc_id,
                chunk_id=f"{doc_id}-chunk-{rc.posicion:04d}",
                # FAQ fila 21: "utilicen la extension real del archivo de origen,
                # escrita en minusculas".
                fuente=fuente, formato=path.suffix.lstrip(".").lower(),
                fenomeno=fenomeno, posicion=rc.posicion,
                num_tokens=rc.num_tokens, texto=rc.text,
            ))
    return nuevos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.adl.yaml")
    ap.add_argument("--batch", type=int, default=16,
                    help="lote de encoding; bajarlo reduce el pico termico")
    ap.add_argument("--dry-run", action="store_true",
                    help="solo informa cuantos fragmentos se anadirian")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base = Path(cfg["paths"]["entrega"]) / "base_vectorial"

    from src.encoding.index import VectorStore
    from src.encoding.sparse import SparseIndex
    from src.encoding.encoders import build_encoder

    for enc_cfg in cfg["encoders"]:
        name = enc_cfg["name"]
        store = VectorStore.load(base / f"encoder_{name}")
        ya = {m["doc_id"] for m in store.metadata}
        print(f"[append] {name}: {store.index.ntotal:,} vectores de {len(ya):,} documentos")

        nuevos = chunks_faltantes(cfg, ya)
        # Los identificadores internos de FAISS continuan donde acaba el indice
        # actual, asi que el chunk_id de los nuevos fragmentos arranca en ntotal.
        for i, c in enumerate(nuevos, start=store.index.ntotal):
            c.chunk_uid = c.chunk_id
            c.chunk_id = str(i)
        docs_nuevos = {c.doc_id for c in nuevos}
        print(f"[append] a anadir: {len(nuevos):,} fragmentos de {len(docs_nuevos)} documentos")
        if args.dry_run or not nuevos:
            continue

        textos = [c.texto for c in nuevos]
        encoder = build_encoder(enc_cfg)
        vectors = encoder.encode(textos, is_query=False, batch_size=args.batch)
        store.add(vectors, [c.to_json() for c in nuevos])
        store.save(base / f"encoder_{name}")
        print(f"[append] {name}: indice denso ahora con {store.index.ntotal:,} vectores")

        if enc_cfg.get("use_sparse") and hasattr(encoder, "encode_sparse"):
            sparse = SparseIndex.load(base / f"encoder_{name}")
            if sparse is None:
                print("[append] no hay indice disperso previo; se omite")
                continue
            weights = encoder.encode_sparse(textos, batch_size=args.batch)
            sparse.add(weights, [c.chunk_id for c in nuevos])
            sparse.save(base / f"encoder_{name}")
            print(f"[append] {name}: indice disperso ahora con {sparse.n_chunks:,} chunks, "
                  f"{sparse.n_tokens:,} tokens unicos")

        # Invariante del formato de entrega: la linea i de metadata.jsonl es el id
        # interno i de FAISS, y el indice disperso comparte ese mismo orden.
        assert store.index.ntotal == len(store.metadata), "metadata desalineada de FAISS"


if __name__ == "__main__":
    main()
