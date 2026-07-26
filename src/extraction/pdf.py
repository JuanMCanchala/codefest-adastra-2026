"""Extraccion de PDF. Backend de vanguardia (Docling) con fallback PyMuPDF.

Docling (IBM, MIT) reconstruye layout, tablas y orden de lectura mucho mejor que
un volcado de texto plano -> mas fidelidad = mejor emparejamiento por `text`.
"""
from __future__ import annotations

from pathlib import Path


def extract_pdf(path: str | Path, backend: str = "docling") -> str:
    if backend == "docling":
        try:
            return _extract_docling(path)
        except Exception:  # pragma: no cover - fallback en runtime
            return _extract_pymupdf(path)
    return _extract_pymupdf(path)


def _extract_docling(path: str | Path, do_ocr: bool = False) -> str:
    """Docling con OCR desactivado por defecto: los PDFs de texto no lo necesitan
    y asi se evita descargar modelos OCR (rapidocr). Mantiene deteccion de tablas.
    Para PDFs escaneados, llamar con do_ocr=True."""
    from docling.document_converter import DocumentConverter, PdfFormatOption  # diferido
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    opts = PdfPipelineOptions()
    opts.do_ocr = do_ocr
    opts.do_table_structure = True
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    result = converter.convert(str(path))
    return result.document.export_to_markdown()


def _extract_pymupdf(path: str | Path) -> str:
    import fitz  # PyMuPDF, import diferido
    doc = fitz.open(str(path))
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n\n".join(pages)
