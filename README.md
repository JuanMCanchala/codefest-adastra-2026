# CODEFEST AD ASTRA 2026 — Etapa 1: Base de Conocimiento Vectorial

Sistema de **recuperación densa multilingüe** (RAG sin generación) para el reto de
la Fuerza Aeroespacial Colombiana + Universidad de los Andes. Indexa un corpus
multiformato (ES/EN/PT) sobre 3 fenómenos y responde 50 consultas devolviendo
**top-3 documentos + top-10 fragmentos**. Evaluación: `NDCG@10` (fragmentos) +
`F1@3` (documentos), fusionados con Conteo de Borda.

## Documentación

| Documento | Para qué |
|---|---|
| **[docs/SISTEMA.md](docs/SISTEMA.md)** | Referencia completa: arquitectura, cada módulo, cada decisión, runbook y **preguntas frecuentes con sus respuestas** |
| **[docs/BITACORA.md](docs/BITACORA.md)** | Trazabilidad: qué cambió, por qué, resultados medidos, bugs corregidos, consultas al jurado |
| **[docs/informe_tecnico.pdf](docs/informe_tecnico.pdf)** | Entregable 3 oficial (4 págs) |
| [docs/NOTA_UNLIMITED_OCR.md](docs/NOTA_UNLIMITED_OCR.md) | Candidato parqueado y por qué |

## Las 5 claves que deciden el puntaje

1. **El emparejamiento es por contenido, no por id** (spec 10.2.1): fragmentos se
   juzgan por el campo `text`; documentos por `fuente` (archivo original de ADL),
   NO por `doc_id`. → la **calidad de extracción** y preservar `fuente` es todo.
2. **Cross-lingual**: consulta en español debe recuperar docs en inglés/portugués.
   → encoder multilingüe fuerte (**BGE-M3**).
3. **Prohibidos decoders**, pero los **cross-encoders sí** son encoders (zona gris,
   toggle + consultar jurado).
4. **No hay ground truth público** → construimos **eval interno** y optimizamos.
5. **Completitud lingüística + 250 palabras** → chunking de **2 niveles**.

## Arquitectura

```
extracción → limpieza → chunking(2 niveles) → encoders(BGE-M3 + E5)
   → FAISS IndexFlatIP → fusión RRF → [rerank cross-encoder] → agregación
   → resultados.jsonl        (+ grafo de conocimiento, bonus)
```

## Estructura del repo

```
config.yaml            # todos los hiperparámetros (barrido reproducible)
generador.py           # ENTREGABLE 4: reproduce resultados.jsonl
scripts/
  build_index.py       # documentos → FAISS + metadata + grafo
  smoke_ingest.py       # prueba extracción+chunking SIN modelos
  make_eval_set.py     # genera eval interno (LLM offline, solo dev)
src/
  extraction/          # pdf(docling+pymupdf), html(trafilatura), json/csv/ocr/pbf
  cleaning/            # normalización: quita TOC/boilerplate, des-hifena (Sec 2.2)
  chunking/            # segmentación oraciones + chunker 2 niveles
  encoding/            # encoders (BGE-M3, E5) + índice FAISS
  retrieval/           # fusión RRF/CombSUM/MNZ, rerank, agregación, pipeline
  graph/               # NER (GLiNER) → grafo.graphml + fusión al retrieval (bonus)
  eval/                # métricas exactas (NDCG@10, F1@3, Borda) + arnés
  schema.py            # metadata Tabla 1 + esquema resultados + validador estricto
tests/                 # 32 tests del core verificable
entrega/               # estructura de entrega final (se genera)
```

## Entorno

Validado en **Python 3.13** (`.venv`). Todo funciona en 3.13, incluido **docling**
(la sospecha de que faltaban wheels era infundada). GPU: RTX 4060 8GB.

Instalado y verificado: `faiss-cpu 1.14.3`, `torch 2.6.0+cu124` (CUDA OK),
`sentence-transformers`, `FlagEmbedding 1.4.0` (BGE-M3), `pymupdf`, `trafilatura`,
`docling`.

```bash
python -m venv .venv && .venv\Scripts\activate
# torch CUDA (RTX 4060):
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Pendiente de instalar cuando haya imágenes reales: `paddleocr` (OCR) y `gliner` (grafo).

## Runbook

```bash
python -m pytest                     # 32 tests (core verificable)
python scripts/smoke_ingest.py       # ingesta sobre data/proxy (sin modelos)
python scripts/build_index.py        # construye base_vectorial/ (necesita ML)
python generador.py                  # genera entrega/resultados.jsonl
```

## Entregables (spec 1.4)

1. `base_vectorial/encoder_<nombre>/{index.faiss, metadata.jsonl}` (obligatorio)
2. `resultados.jsonl` (50 líneas q001–q050, esquema estricto)
3. `informe_tecnico.pdf` (≤8 páginas)
4. `generador.py` (reproduce resultados)
5. `base_vectorial/grafo/grafo.graphml` (bonus)

## Decisiones abiertas / riesgos

- **Reranker cross-encoder**: `rerank.enabled` en config. Consultar al jurado; si
  lo vetan → `false` (fusión RRF pura).
- **`fuente` exacto**: `build_index.py` usa la ruta relativa del corpus; ajustar
  al layout/manifiesto real que entregue ADL para que el F1@3 empareje.
- **Determinismo**: `generador.py` fija semillas; pinnear versiones antes de entregar.
- **Estado**: core verificable ✅ (tests). Capa ML: skeletons listos, faltan
  correr con el venv 3.11 y afinar con el corpus real.
