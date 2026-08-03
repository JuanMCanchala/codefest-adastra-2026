"""ocr_scanned.py - Recupera el texto de los PDFs escaneados del corpus.

Del corpus de ADL, 51 PDFs no tienen capa de texto (son imagen pura). 47 de
ellos son informes de Alertas Tempranas, el observatorio colombiano cuyos
documentos son los mas relevantes para las consultas q033-q050 (control
territorial, GAO/GAOR/GDO, mineria ilegal). Sin OCR, el sistema no los ve.

Rasteriza cada pagina con PyMuPDF y la pasa por PaddleOCR (modelo clasico de
deteccion+reconocimiento, NO generativo: no entra en la zona gris del
reglamento). El texto resultante se escribe en la misma cache que usa
build_index.py, asi que basta reindexar despues.

    python scripts/ocr_scanned.py --config config.adl.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# PaddlePaddle 3.x en Windows falla al ejecutar la deteccion con oneDNN
# (NotImplementedError: ConvertPirAttribute2RuntimeAttribute). Desactivarlo antes
# de importar paddle evita el fallo; el coste en velocidad es asumible para 51 PDFs.
os.environ.setdefault("FLAGS_use_mkldnn", "0")

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cleaning.normalize import clean_document   # noqa: E402


def _texts_from_result(res) -> list[str]:
    """Extrae las cadenas reconocidas, tolerando los formatos de PaddleOCR v2 y v3.

    v3: [{'rec_texts': [...], 'rec_scores': [...], ...}]
    v2: [[[box, (texto, confianza)], ...]]
    """
    out: list[str] = []
    for page in res or []:
        if isinstance(page, dict):                       # v3
            out.extend(str(t) for t in page.get("rec_texts", []) if t)
        elif isinstance(page, (list, tuple)):            # v2
            for line in page or []:
                try:
                    out.append(str(line[1][0]))
                except (IndexError, TypeError):
                    continue
        elif isinstance(page, str):
            out.append(page)
    return out


def ocr_pdf(ocr, pdf_path: Path, dpi: int = 200, max_pages: int = 60) -> str:
    """Rasteriza y reconoce el texto de un PDF escaneado."""
    import fitz

    doc = fitz.open(str(pdf_path))
    parts: list[str] = []
    tmpdir = Path(tempfile.mkdtemp(prefix="ocr_"))
    try:
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            img = tmpdir / f"p{i:03d}.png"
            page.get_pixmap(dpi=dpi).save(str(img))
            try:
                res = ocr.predict(str(img)) if hasattr(ocr, "predict") else ocr.ocr(str(img))
            except Exception:
                continue
            parts.extend(_texts_from_result(res))
            img.unlink(missing_ok=True)
    finally:
        doc.close()
    return "\n".join(parts)


_OCR = None   # una instancia por proceso trabajador (cargar el modelo es caro)


def _worker(task: tuple) -> dict:
    """Procesa un PDF escaneado. Reutiliza el modelo dentro del mismo proceso."""
    global _OCR
    pdf_str, doc_id, cache_dir, dpi = task
    if _OCR is None:
        from paddleocr import PaddleOCR
        _OCR = PaddleOCR(lang="es", enable_mkldnn=False,
                         use_doc_orientation_classify=False,
                         use_doc_unwarping=False,
                         use_textline_orientation=False)
    try:
        texto = clean_document(ocr_pdf(_OCR, Path(pdf_str), dpi=dpi))
    except Exception as exc:
        return {"doc_id": doc_id, "n_chars": 0, "error": type(exc).__name__}
    if len(texto) >= 50:
        (Path(cache_dir) / f"{doc_id}.txt").write_text(texto, encoding="utf-8")
    return {"doc_id": doc_id, "n_chars": len(texto)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.adl.yaml")
    ap.add_argument("--dpi", type=int, default=150,
                    help="150 basta para texto escaneado y es ~2x mas rapido que 200")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    corpus = Path(cfg["paths"]["corpus_raw"])
    processed = Path(cfg["paths"]["processed"])
    cache = processed / "text"

    docs = [json.loads(l) for l in (processed / "docs.jsonl").open(encoding="utf-8")]
    pending = [d for d in docs
               if d.get("formato") == "pdf" and not d.get("error") and d["n_chars"] < 50]
    if args.limit:
        pending = pending[: args.limit]
    print(f"[ocr] {len(pending)} PDFs escaneados por procesar")
    if not pending:
        return

    from concurrent.futures import ProcessPoolExecutor, as_completed
    tasks = [(str(corpus / d["fuente"]), d["doc_id"], str(cache), args.dpi)
             for d in pending]
    print(f"[ocr] {args.workers} procesos, {args.dpi} DPI")

    t0, recovered, failed = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, t): t[1] for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r.get("error"):
                failed += 1
            elif r["n_chars"] >= 50:
                recovered += 1
            rate = i / max(time.time() - t0, 1e-9)
            print(f"  {i}/{len(tasks)}  {r['doc_id']:16s} {r['n_chars']:7d} chars"
                  f"  ETA {(len(tasks)-i)/max(rate,1e-9)/60:.0f} min")

    print(f"\n[ocr] recuperados {recovered}/{len(pending)} | fallos {failed}"
          f" | {(time.time()-t0)/60:.1f} min")
    print("[ocr] reindexar con: python scripts/build_index.py --config config.adl.yaml")


if __name__ == "__main__":
    main()
