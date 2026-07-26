"""Smoke test de la ingesta SIN modelos ML: extraccion + chunking + metadata.

Recorre data/proxy, extrae texto (md/json/...), chunkea con completitud
linguistica y valida la metadata de la Tabla 1. No carga encoders ni FAISS.

    python scripts/smoke_ingest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extraction.base import extract, detect_format          # noqa: E402
from src.chunking.chunker import chunk_document                 # noqa: E402
from src.schema import ChunkMeta, validate_chunk_meta, word_count  # noqa: E402

PROXY = Path("data/proxy")


def main() -> None:
    total_chunks = 0
    errors = 0
    for i, path in enumerate(sorted(PROXY.rglob("*"))):
        if not path.is_file():
            continue
        try:
            detect_format(path)
        except ValueError:
            continue
        fenomeno = next((n for n in (1, 2, 3) if f"fenomeno{n}" in str(path).lower()), 0)
        doc_id = f"DOC-{i:04d}"
        fuente = path.relative_to(PROXY).as_posix()
        doc = extract(path, doc_id=doc_id, fuente=fuente, fenomeno=fenomeno)
        chunks = chunk_document(doc.text, index_max_tokens=80, overlap_sentences=1)
        for rc in chunks:
            meta = ChunkMeta(
                doc_id=doc_id, chunk_id=f"{doc_id}-chunk-{rc.posicion:04d}",
                fuente=fuente, formato=doc.formato, fenomeno=fenomeno,
                posicion=rc.posicion, num_tokens=rc.num_tokens, texto=rc.text,
            )
            errs = validate_chunk_meta(meta.to_json())
            errors += len(errs)
            max_w = max((word_count(rc.text) for rc in chunks), default=0)
        print(f"  {fuente:45s} fmt={doc.formato:5s} fen={fenomeno} "
              f"chars={len(doc.text):5d} chunks={len(chunks)} max_words={max_w}")
        total_chunks += len(chunks)

    print(f"\n[smoke] total chunks={total_chunks}  errores_metadata={errors}")
    print("[smoke] OK" if errors == 0 else "[smoke] REVISAR metadata")


if __name__ == "__main__":
    main()
