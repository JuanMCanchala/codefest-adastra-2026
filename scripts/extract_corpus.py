"""extract_corpus.py - Extraccion + limpieza en paralelo, con cache en disco.

La extraccion del corpus de ADL (1826 documentos, 760 PDFs) tarda ~2.7 h en un
solo proceso y es puramente CPU. Ademas habria que repetirla cada vez que se
cambia el tamaño de chunk, que es justo el hiperparametro que queremos barrer.

Este script la hace UNA vez, en paralelo, y guarda el texto ya limpio:

    data/processed/text/<DOC_ID>.txt     texto extraido y normalizado
    data/processed/docs.jsonl            metadata por documento

Despues, build_index.py lee de la cache y reindexar cuesta minutos en vez de horas.
Es reanudable: los documentos ya cacheados se saltan.

    python scripts/extract_corpus.py --config config.adl.yaml --workers 8
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_index import iter_corpus                      # noqa: E402
from src.extraction.base import extract                  # noqa: E402
from src.cleaning.normalize import clean_document        # noqa: E402
from src.utils.io import write_jsonl                     # noqa: E402


def _process_one(args: tuple) -> dict:
    """Extrae y limpia un documento. Se ejecuta en un proceso aparte."""
    path_str, doc_id, fuente, fenomeno, pdf_backend, drop_bp, out_dir = args
    out_file = Path(out_dir) / f"{doc_id}.txt"
    if out_file.exists() and out_file.stat().st_size > 0:
        return {"doc_id": doc_id, "fuente": fuente, "fenomeno": fenomeno,
                "formato": Path(path_str).suffix.lstrip(".").lower(),
                "n_chars": out_file.stat().st_size, "cached": True}
    try:
        doc = extract(Path(path_str), doc_id=doc_id, fuente=fuente,
                      fenomeno=fenomeno, pdf_backend=pdf_backend)
        texto = clean_document(doc.text, drop_boilerplate=drop_bp)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(texto, encoding="utf-8")
        return {"doc_id": doc_id, "fuente": fuente, "fenomeno": fenomeno,
                "formato": doc.formato, "n_chars": len(texto), "cached": False}
    except Exception as exc:
        return {"doc_id": doc_id, "fuente": fuente, "fenomeno": fenomeno,
                "formato": Path(path_str).suffix.lstrip(".").lower(),
                "n_chars": 0, "error": type(exc).__name__}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.adl.yaml")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="procesar solo N documentos")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    raw_dir = Path(cfg["paths"]["corpus_raw"])
    out_dir = Path(cfg["paths"]["processed"]) / "text"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_backend = cfg.get("extraction", {}).get("pdf_backend", "pymupdf")
    drop_bp = cfg.get("cleaning", {}).get("drop_boilerplate", True)

    items = list(iter_corpus(raw_dir, cfg["paths"].get("inventario")))
    if args.limit:
        items = items[: args.limit]
    tasks = [(str(p), d, f, fen, pdf_backend, drop_bp, str(out_dir))
             for p, d, f, fen in items]
    print(f"[extract] {len(tasks)} documentos, {args.workers} procesos")

    results: list[dict] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_one, t): t[1] for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 100 == 0 or i == len(tasks):
                done = time.time() - t0
                rate = i / max(done, 1e-9)
                eta = (len(tasks) - i) / max(rate, 1e-9)
                print(f"  {i}/{len(tasks)}  {rate:.1f} doc/s  ETA {eta/60:.0f} min")

    errors = [r for r in results if r.get("error")]
    empties = [r for r in results if not r.get("error") and r["n_chars"] < 50]
    total_chars = sum(r["n_chars"] for r in results)
    write_jsonl(Path(cfg["paths"]["processed"]) / "docs.jsonl", results)

    print(f"\n[extract] listo en {(time.time()-t0)/60:.1f} min")
    print(f"  documentos: {len(results)} | errores: {len(errors)} | casi vacios: {len(empties)}")
    print(f"  texto total: {total_chars/1e6:.1f} M caracteres")
    if errors:
        from collections import Counter
        print("  tipos de error:", dict(Counter(r["error"] for r in errors)))


if __name__ == "__main__":
    main()
