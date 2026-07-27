"""Backend OCR experimental: baidu/Unlimited-OCR (3B, MIT).

EXPERIMENTAL - rama feat/unlimited-ocr. NO se usa por defecto. Ver la nota de
decision en docs/NOTA_UNLIMITED_OCR.md antes de activarlo.

Que aporta: parsea documentos largos en una sola pasada ("one-shot long-horizon",
contexto ~32K) en vez de trocear por pagina, multilingue, SOTA abierto en
OmniDocBench v1.5 (93.2%). Util para PDFs ESCANEADOS, imagenes con texto e
infografias, donde PaddleOCR se queda corto.

Que NO aporta: para PDFs digitales limpios, docling/pymupdf extraen el texto
LITERAL, que es lo que conviene porque la evaluacion del reto compara el campo
`text` del fragmento. Un modelo generativo puede reformular y perder literalidad.

Restriccion del reto: es un DECODER (generativo). La prohibicion de la Seccion 8.3
aplica a la RECUPERACION y la 4.2 a los EMBEDDINGS; la Seccion 2.1 si contempla
OCR en preprocesamiento. Aun asi, CONSULTAR AL JURADO antes de usarlo.

Uso (requiere descargar ~6GB de pesos):
    pip install transformers torch
    from src.extraction.unlimited_ocr import UnlimitedOCR
    ocr = UnlimitedOCR()
    texto = ocr.extract_image("figura.png")
    texto = ocr.extract_pdf("informe_escaneado.pdf")
"""
from __future__ import annotations

from pathlib import Path

MODEL_ID = "baidu/Unlimited-OCR"


class UnlimitedOCR:
    """Envoltorio del modelo. Los imports e inicializacion son diferidos: importar
    este modulo no descarga nada ni carga torch."""

    def __init__(self, model_id: str = MODEL_ID, device: str = "auto", dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        from ..encoding.encoders import resolve_device

        self.device = resolve_device(device)
        torch_dtype = getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch_dtype,
        )
        self.model = self.model.eval().to(self.device)

    def extract_image(self, path: str | Path, max_length: int = 8192, **kwargs) -> str:
        """OCR de una sola imagen (figura, infografia, pagina escaneada)."""
        return self.model.infer(
            self.tokenizer,
            image_file=str(path),
            max_length=max_length,
            **kwargs,
        )

    def extract_pages(self, paths: list[str | Path], max_length: int = 32768, **kwargs) -> str:
        """Parseo one-shot de varias paginas: mantiene el hilo del documento en
        lugar de tratarlas como imagenes independientes."""
        return self.model.infer_multi(
            self.tokenizer,
            image_files=[str(p) for p in paths],
            max_length=max_length,
            **kwargs,
        )

    def extract_pdf(self, path: str | Path, dpi: int = 144, max_pages: int | None = None,
                    **kwargs) -> str:
        """Rasteriza el PDF y lo pasa por el parser one-shot.

        Solo tiene sentido para PDFs ESCANEADOS. Para PDFs digitales usar
        src/extraction/pdf.py (docling/pymupdf), que preserva el texto literal.
        """
        import tempfile
        import fitz  # PyMuPDF para rasterizar

        doc = fitz.open(str(path))
        pages = doc if max_pages is None else list(doc)[:max_pages]
        images: list[str] = []
        tmpdir = tempfile.mkdtemp(prefix="unlimited_ocr_")
        for i, page in enumerate(pages):
            pix = page.get_pixmap(dpi=dpi)
            out = Path(tmpdir) / f"page_{i:04d}.png"
            pix.save(str(out))
            images.append(str(out))
        doc.close()
        return self.extract_pages(images, **kwargs)


def is_available() -> bool:
    """True si transformers y torch estan instalados (no comprueba los pesos)."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False
