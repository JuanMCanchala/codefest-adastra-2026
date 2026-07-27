# Nota de decisión — Unlimited-OCR (candidato para extracción)

**Estado:** PARQUEADO en la rama `feat/unlimited-ocr`. No está en `main` ni activo por defecto.
**Fecha:** julio 2026
**Origen:** publicación de LinkedIn que reportaba el modelo; verificado contra las fuentes oficiales.

## Qué es (verificado, no del post)

| Dato | Valor |
|---|---|
| Modelo | `baidu/Unlimited-OCR` |
| Tamaño | 3B parámetros |
| Licencia | **MIT** (compatible con el reto) |
| Publicado | 22-jun-2026 |
| Linaje | continue-trained desde un checkpoint de **DeepSeek-OCR** (encoder visual congelado, decoder entrenado, ~2M documentos) |
| Benchmark | **93.2%** en OmniDocBench v1.5 (+6.2 sobre DeepSeek-OCR); 93.9% en v1.6 — SOTA entre modelos abiertos |
| Ejecución | 100% local: Transformers, SGLang, GGUF (llama.cpp/Ollama) |
| API | `AutoModel` + `trust_remote_code=True`, `model.infer()` / `model.infer_multi()`, `bfloat16` |

**Su diferencial real:** parseo *one-shot long-horizon* (contexto ~32K) — lee el documento largo completo en una pasada en vez de trocearlo página por página, preservando la continuidad del texto.

Fuentes: [HF](https://huggingface.co/baidu/Unlimited-OCR) · [GitHub](https://github.com/baidu/Unlimited-OCR)

## Por qué NO está en `main`

### 1. Riesgo de fidelidad textual (el motivo principal)
La evaluación del reto compara el **contenido del campo `text`** de cada fragmento (Sec. 10.2.1).
Unlimited-OCR es un modelo **generativo**: puede normalizar, reformular o alucinar texto en vez de
transcribirlo literalmente. Para PDFs digitales, `docling`/`pymupdf` extraen el texto **exacto**, que es
justo lo que maximiza el emparejamiento. Cambiar a un VLM sin medir sería arriesgar puntaje.

### 2. Zona gris del reglamento
Es un **decoder**. Las prohibiciones explícitas del reto son:
- Sec. 4.2 — decoders prohibidos para generar **embeddings**.
- Sec. 8.3 — decoders prohibidos en la **recuperación** (reranking, expansión de consulta, síntesis).

La Sec. 2.1 **sí contempla OCR** en preprocesamiento, así que su uso en extracción *probablemente* es
admisible — pero es la misma zona gris que el reranker cross-encoder: **hay que preguntar al jurado**.

### 3. Coste operativo
3B en bf16 ≈ 6 GB de VRAM (tenemos 8 GB en la RTX 4060). Viable porque la extracción es una fase
separada del indexado, pero ajustado; la variante GGUF cuantizada alivia.

## Cuándo SÍ lo activamos

Tiene sentido evaluarlo **solo si el corpus real de ADL contiene**:
- PDFs **escaneados** (sin capa de texto), o
- **imágenes/infografías** con texto relevante (donde PaddleOCR es flojo), o
- PDFs con **layout complejo multilingüe** que docling parsee mal.

Para PDFs digitales limpios —probablemente la mayoría del corpus— **docling ya gana** sin VLM ni zona gris.

## Cómo decidir (A/B con datos, no por intuición)

1. Inspeccionar el corpus real: ¿cuántos documentos son escaneados o imagen?
2. Sobre una muestra de esos documentos, extraer con `docling`/`paddleocr` **y** con `unlimited-ocr`.
3. Comparar **fidelidad literal** (¿transcribe o reformula?) y cobertura de texto.
4. Medir el efecto real en **NDCG@10 / F1@3** con el arnés interno (`scripts/compare_configs.py`).
5. Activar solo si gana en el barrido **y** el jurado aprueba el uso de un decoder en preprocesamiento.

## Qué hay implementado en esta rama

- `src/extraction/unlimited_ocr.py` — adaptador `UnlimitedOCR` con `extract_image()`,
  `extract_pages()` (one-shot multi-página) y `extract_pdf()` (rasteriza y parsea).
  Imports diferidos: importar el módulo **no** descarga pesos ni carga torch.
- **No** está cableado en `build_index.py` ni en `config.yaml`. Integrarlo es añadir una rama en
  `src/extraction/base.py` para `formato == "image"` y un flag `extraction.ocr_backend`.

## Para retomar

```bash
git checkout feat/unlimited-ocr
pip install transformers torch          # los pesos (~6GB) se bajan al instanciar
python -c "from src.extraction.unlimited_ocr import UnlimitedOCR; print(UnlimitedOCR().extract_image('figura.png'))"
```
