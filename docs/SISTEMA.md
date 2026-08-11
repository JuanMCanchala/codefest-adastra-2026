# SISTEMA — Documentación completa

> Documento de referencia interno. Explica **qué hace cada pieza, por qué está así y cómo defenderlo**.
> Para el entregable oficial ver `informe_tecnico.pdf`. Para el historial de cambios ver `BITACORA.md`.

---

## 1. El reto en una página

**Objetivo:** construir una base de conocimiento vectorial que, ante 50 consultas en lenguaje
natural, devuelva los **3 documentos** y los **10 fragmentos** más relevantes de un corpus
multiformato y multilingüe (ES/EN/PT) sobre tres fenómenos:

| Fenómeno | Tema |
|---|---|
| 1 | IA e innovación en entornos militares / Defensa Nacional |
| 2 | Seguridad espacial y órbita baja terrestre (LEO), *space debris* |
| 3 | Dinámicas territoriales en América Latina y el Caribe |

**Cómo se puntúa:**
- `NDCG@10` sobre los **fragmentos** (mide calidad del orden del ranking)
- `F1@3` sobre los **documentos** (mide acierto del conjunto, sin importar orden)
- Ambas tablas se fusionan en un leaderboard único con **Conteo de Borda**

**Los 5 entregables:**
1. `base_vectorial/encoder_<nombre>/{index.faiss, metadata.jsonl}`
2. `resultados.jsonl` (50 líneas, q001–q050)
3. `informe_tecnico.pdf` (≤8 páginas)
4. `generador.py` (debe **reproducir** los resultados o se excluye la entrega)
5. `base_vectorial/grafo/grafo.graphml` (bonus)

---

## 2. Las reglas que realmente deciden el puntaje

Estas cinco son las que orientaron todo el diseño. Si te preguntan "¿por qué priorizaron X?",
la respuesta está casi siempre aquí.

### 2.1 El emparejamiento es por CONTENIDO (fragmentos) y por `doc_id` (documentos)
> Sección 10.2.1: *"La relevancia de cada fragmento se juzga sobre su contenido textual (campo
> `text`). El `chunk_id` **no** es la clave de emparejamiento… a nivel documento el emparejamiento
> se realiza a través del campo `fuente`, no del `doc_id` arbitrario asignado por el equipo."*

⚠️ **La segunda mitad de esa frase es una errata del handbook.** ADL lo corrigió en el archivo de
Preguntas Frecuentes (filas 19, 52 y 58): *"Eso es un error del handbook en su primera versión. El
emparejamiento es con el **`doc_id` suministrado**"*, el del inventario `Indice_Datos_Codefest.xlsx`.
Nosotros ya usábamos ese identificador antes de la aclaración, así que no hubo que cambiar nada.

**Consecuencia:** para los 10 fragmentos manda el **texto** — la calidad de la extracción y la
limpieza pone el techo del NDCG@10. Para los 3 documentos manda el `doc_id` oficial. Por eso
invertimos más esfuerzo en extracción/limpieza que en afinar el recuperador.

### 2.1b El `chunk_id` debe ser el del índice FAISS
> Preguntas Frecuentes, fila 42: *"deberían usar como `chunk_id` el mismo obtenido del índice FAISS."*

Es el identificador interno del vector, es decir el número de línea del registro en
`metadata.jsonl` empezando en 0. El identificador legible (`F2-SWF-012-chunk-0007`) se conserva en
el campo adicional `chunk_uid` para poder rastrear a mano cualquier resultado.

### 2.1c El grafo solo puntúa si se integra a la recuperación
> Preguntas Frecuentes, fila 42: *"Es bono y para que sea válido lo deben integrar a la
> recuperación, el solo construirlo no es válido."*

De ahí `graph.enabled: true` **y** `graph.fuse_into_retrieval: true`: el grafo aporta un ranking
más a la fusión RRF. Construirlo y exportarlo sin conectarlo no habría sumado nada.

### 2.2 Prohibición de modelos generativos
> Sección 8.3: *"En ninguna etapa del proceso de recuperación se permite el uso de modelos de
> lenguaje generativos (arquitecturas decoder…). La recuperación debe operar exclusivamente sobre
> vectores, puntuaciones de similitud y metadata."*
> Sección 4.2: *"Los equipos deben utilizar modelos encoder disponibles públicamente en
> HuggingFace bajo licencias de uso libre."*

**Consecuencia:** todo el sistema usa encoders (familia BERT). El único punto discutible es el
reranker cross-encoder (ver §7.3), aislado tras un interruptor.

### 2.3 Completitud lingüística (requisito obligatorio)
> Sección 3.3: *"Ningún fragmento puede contener oraciones o frases incompletas. Los cortes entre
> chunks consecutivos deben realizarse únicamente en límites oracionales."*

### 2.4 Límite de 250 palabras por fragmento de salida
> Sección 9.2. Además, la 9.2.1 **permite** concatenar fragmentos adyacentes del mismo documento
> para enriquecer el contexto, siempre que no se superen las 250 palabras. *(Palanca aún no
> explotada — ver §11.)*

### 2.5 Esquema estricto de salida
> Sección 9.3.2: exactamente **3 documentos** y **10 fragmentos** por consulta; los objetos que
> incumplan *"serán penalizados o descartados durante la evaluación automática"*.

---

## 3. Arquitectura

```
  ┌─ INDEXACIÓN (offline, scripts/build_index.py) ──────────────────────┐
  │                                                                     │
  │  Documentos ADL                                                     │
  │       │                                                             │
  │       ▼                                                             │
  │  Extracción        src/extraction/   (docling+pymupdf, trafilatura, │
  │       │                               pandas, OCR, pyosmium)        │
  │       ▼                                                             │
  │  Limpieza          src/cleaning/     (TOC, boilerplate, guiones)    │
  │       │                                                             │
  │       ▼                                                             │
  │  Chunking N1       src/chunking/     (oraciones completas + overlap)│
  │       │                                                             │
  │       ├──────────────► Encoder denso ──► index.faiss                │
  │       ├──────────────► Encoder disperso ► sparse_index.json         │
  │       ├──────────────► Metadata ───────► metadata.jsonl             │
  │       └──────────────► NER + relaciones ► grafo.graphml   (bonus)   │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─ RECUPERACIÓN (online, generador.py) ───────────────────────────────┐
  │                                                                     │
  │  Consulta                                                           │
  │     ├──► encoder denso ──► FAISS ────────┐                          │
  │     ├──► encoder disperso ► índice inv. ─┤                          │
  │     └──► NER ────────────► grafo ────────┤                          │
  │                                          ▼                          │
  │                                   Fusión RRF (k₀=60)                │
  │                                          │                          │
  │                                          ▼                          │
  │                            [Reranking cross-encoder]  ← toggle      │
  │                                          │                          │
  │                       ┌──────────────────┴───────────────┐          │
  │                       ▼                                  ▼          │
  │              Chunking N2 (≤250 pal)          Agregación chunk→doc   │
  │                       │                                  │          │
  │                  10 fragmentos                     3 documentos     │
  │                       └──────────────┬───────────────────┘          │
  │                                      ▼                              │
  │                          Validador estricto → resultados.jsonl      │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Mapa de módulos

| Archivo | Responsabilidad |
|---|---|
| **`src/schema.py`** | Metadata Tabla 1 (`ChunkMeta`), objetos de salida, **validador estricto** |
| **`src/extraction/base.py`** | Dispatch por formato; construye `Document` con `fuente` preservado |
| `src/extraction/pdf.py` | Docling (layout/tablas) con caída automática a PyMuPDF |
| `src/extraction/html_ext.py` | trafilatura (quita boilerplate web) |
| `src/extraction/misc.py` | MD/TXT, JSON (campos de texto), CSV/XLSX (fila→`col: valor`), OCR, PBF |
| **`src/cleaning/normalize.py`** | NFC, control chars, líneas de índice, boilerplate repetido, des-hifenación |
| **`src/chunking/sentences.py`** | Segmentación oracional multilingüe (pysbd + respaldo regex) |
| **`src/chunking/chunker.py`** | Nivel 1 (chunks de índice) y Nivel 2 (sub-fragmentos ≤250 palabras) |
| **`src/encoding/encoders.py`** | BGE-M3 y encoders sentence-transformers; `resolve_device` (auto GPU/CPU) |
| **`src/encoding/index.py`** | `VectorStore`: FAISS + metadata alineada, save/load |
| **`src/encoding/sparse.py`** | `SparseIndex`: índice invertido de pesos lexicales de BGE-M3 |
| **`src/retrieval/fusion.py`** | CombSUM, CombMNZ, **RRF** (ecuaciones 5–7) |
| **`src/retrieval/aggregate.py`** | Agregación chunk→documento (max_pool / sum / weighted_mean) |
| **`src/retrieval/rerank.py`** | Cross-encoder opcional (toggle) — ver §7.3 |
| **`src/retrieval/pipeline.py`** | `Retriever`: orquesta denso + disperso + grafo → fusión → salida |
| **`src/graph/build.py`** | NER (GLiNER) + relaciones por co-ocurrencia → `grafo.graphml` |
| **`src/graph/retrieve.py`** | `GraphRetriever`: entidades de la consulta → chunks (nodo + vecinos) |
| **`src/eval/metrics.py`** | NDCG@10, F1@3, Conteo de Borda — **idénticas a la spec** |
| **`src/eval/harness.py`** | Arnés de evaluación: corre el retriever sobre el eval set y mide |
| `entrega/generador.py` | **Entregable 4**: índice + consultas → `resultados.jsonl` (determinista). `--pausa` acota el pico térmico |
| `scripts/compare_resultados.py` | A/B entre dos `resultados.jsonl` de configuraciones distintas |
| `scripts/build_index.py` | Pipeline de indexación completo |
| `scripts/extract_corpus.py` | Extracción + limpieza en paralelo, con caché en disco (reanudable) |
| `scripts/append_docs.py` | **Indexación incremental**: anexa al índice los documentos que falten, sin reconstruir la base (los ids internos se asignan por orden de inserción, así que anexar al final no altera ninguno) |
| `scripts/migrate_chunk_ids.py` | Migración única de `chunk_id` al identificador interno de FAISS; reescribe metadata y disperso y **verifica** que sigan alineados |
| `scripts/ocr_scanned.py` | OCR (EasyOCR/GPU) de los 51 PDF escaneados hacia la caché de texto |
| `scripts/build_graph.py` | Construye `grafo.graphml` con NER por lotes; reanudable y con control térmico (`--batch`, `--pausa`, `--hilos`) |
| `scripts/check_results.py` | Cinco comprobaciones de calidad sobre `resultados.jsonl` |
| `scripts/fetch_corpus.py` | Descarga corpus de prueba real (arXiv API + institucionales) |
| `scripts/make_eval_corpus.py` | Genera eval set con relevancia desde el manifiesto |
| `scripts/eval_corpus.py` | Compara variantes sobre índice ya construido |
| `scripts/sweep.py` | Barrido de hiperparámetros rankeado por Borda |
| `scripts/compare_configs.py` | Compara multi-encoder y reranking |

---

## 5. Decisiones de diseño y su justificación

| Decisión | Alternativas descartadas | Por qué |
|---|---|---|
| **Chunking por oraciones con solapamiento** | Tamaño fijo de tokens; párrafo puro | El tamaño fijo **viola** el requisito 3.3 (parte oraciones). El párrafo puro produce fragmentos de tamaño muy desigual |
| **Dos niveles de chunking** | Un solo tamaño | El tamaño óptimo para *codificar* (ventana del encoder) ≠ tamaño máximo de *presentación* (250 palabras) |
| **BGE-M3** | E5, MiniLM, modelos monolingües | Cross-lingual nativo ES/EN/PT, 8192 tokens, MIT, denso+disperso+ColBERT en una pasada. Validado: 0.990 vs 0.966 de MiniLM |
| **IndexFlatIP normalizado** | IVFFlat, HNSW | Coseno **exacto**; a este volumen la búsqueda exhaustiva sobra. Los aproximados sacrifican exactitud por velocidad, que **no se puntúa** |
| **Fusión RRF (k₀=60)** | CombSUM, CombMNZ | Opera sobre **posiciones**, robusto a que distintos encoders tengan escalas de score distintas. Es el estándar de producción |
| **Agregación max_pool** | sum, weighted_mean | `sum` favorece documentos largos con muchos fragmentos mediocres; medido peor en F1@3 en **dos** corpus |
| **Docling + respaldo PyMuPDF** | Solo PyMuPDF | +13% de contenido y recupera tablas; el respaldo garantiza que ningún PDF tumbe el pipeline |
| **Grafo por co-ocurrencia** | Extracción de relaciones con LLM | Un LLM sería un **decoder** (prohibido). La co-ocurrencia es verificable y trazable a la evidencia |

---

## 6. Configuración (`config.yaml`) — qué hace cada parámetro

```yaml
seed: 42                      # determinismo (random, numpy, torch)

extraction:
  pdf_backend: docling        # docling (fidelidad) | pymupdf (velocidad)
  ocr_enabled: true           # OCR sobre imágenes con texto

cleaning:
  drop_boilerplate: true      # quita encabezados/pies repetidos
  min_chunk_chars: 40         # descarta fragmentos basura

chunking:
  index_max_tokens: 384       # tamaño del chunk que se codifica (<512 del encoder)
  overlap_sentences: 1        # oraciones repetidas entre chunks vecinos
  output_max_words: 250       # LÍMITE DURO de la spec
  respect_sentences: true     # requisito obligatorio 3.3

encoders:                     # lista: cada uno genera su propio índice FAISS
  - name: bge-m3
    use_dense: true           # vector semántico
    use_sparse: true          # pesos lexicales (siglas, nombres propios)
    use_colbert: false        # multi-vector (aún no explotado)

faiss:
  index_type: IndexFlatIP     # coseno exacto con vectores normalizados

retrieval:
  top_k_faiss: 100            # candidatos por índice antes de fusionar
  fusion: rrf                 # rrf | combsum | combmnz
  rrf_k: 60                   # constante de suavizado
  final_fragments: 10         # EXACTO por spec
  final_documents: 3          # EXACTO por spec

rerank:
  enabled: true               # TOGGLE — ver §7.3
  top_k_candidates: 50        # cuántos candidatos reordena

aggregation:
  method: max_pool            # ganador medido

graph:
  enabled: true               # bonus
  fuse_into_retrieval: true   # grafo como índice extra en la fusión
```

---

## 7. Preguntas que te pueden hacer (y cómo responderlas)

### 7.1 "¿Por qué esa estrategia de chunking?"
Porque el reglamento **obliga** a completitud lingüística (3.3) y **limita** los fragmentos de
salida a 250 palabras (9.2), y esas dos cosas actúan en momentos distintos. Agrupamos oraciones
completas hasta el límite del encoder para indexar, y al construir la respuesta re-dividimos
respetando otra vez los límites oracionales. El solapamiento de una oración evita que una idea a
caballo entre dos fragmentos quede representada a medias en ambos.

### 7.2 "¿Por qué BGE-M3 y no otro encoder?"
Por el requisito **cross-lingual**: las consultas se reparten entre ES/EN/PT y una consulta en un
idioma debe recuperar documentos en los otros dos. BGE-M3 es multilingüe nativo, licencia MIT,
acepta 8192 tokens y produce tres representaciones (densa, dispersa, multi-vector) en una sola
pasada. Lo medimos contra un multilingüe pequeño: 0.990 vs 0.966 de NDCG@10, y la diferencia se
concentró en la consulta en portugués (1.000 vs 0.877).

### 7.3 "¿El reranker no viola la prohibición de modelos generativos?"
La Sección 8.3 prohíbe **modelos generativos (arquitecturas decoder)** y menciona el reranking
*"mediante un LLM"*. `bge-reranker-v2-m3` es **XLM-RoBERTa: un encoder**, familia BERT; recibe el
par (consulta, fragmento) y devuelve **un número**, no genera texto. Además la Sección 4.2 nos
indica usar modelos encoder de HuggingFace con licencia libre, que es lo que es.

**Sé honesto con el contraargumento:** la frase *"la recuperación debe operar exclusivamente sobre
vectores, puntuaciones de similitud y metadata"* juega en contra, porque un cross-encoder relee el
**texto** en vez de operar sobre los vectores almacenados. Por eso: (a) está tras un interruptor
`rerank.enabled`, (b) **se consultó al jurado**, (c) si lo vetan, el sistema funciona con fusión
RRF pura sin cambiar una línea.

### 7.4 "¿Cómo garantizan que no usan decoders?"
Todos los modelos del pipeline son encoders: BGE-M3 (XLM-RoBERTa), E5 (XLM-R), GLiNER (DeBERTa),
bge-reranker (XLM-R). Las relaciones del grafo salen de **co-ocurrencia**, no de un LLM. La fusión
y la agregación son aritmética sobre puntuaciones. *(Nota: usamos un LLM offline solo para redactar
consultas de nuestro eval interno de desarrollo; no forma parte del pipeline entregado ni influye
en `resultados.jsonl`.)*

### 7.5 "¿Cómo aseguran las 250 palabras y las oraciones completas?"
Tres capas: (1) el chunker corta solo en límites oracionales; (2) `split_for_output()` re-divide
respetando oraciones; (3) un **validador estricto** verifica cada objeto antes de escribir y
reporta cualquier violación. Hay pruebas automatizadas para las tres.

### 7.6 "¿Cómo es reproducible?"
`generador.py` fija semillas de `random`, NumPy y PyTorch, activa algoritmos deterministas y
escribe JSON Lines con separadores fijos y UTF-8 explícito. Toda la configuración vive en un solo
`config.yaml`, así que cada resultado es rastreable a una configuración concreta.

### 7.7 "¿Cómo optimizaron sin conocer el ground truth?"
Construimos un **arnés de evaluación interno** con métricas idénticas a la spec (verificadas
reproduciendo el ejemplo de Borda de la Tabla 3 del PDF) y un corpus de prueba de 50 documentos
públicos reales. Para los juicios de relevancia usamos una señal **externa e independiente**: el
propio buscador de arXiv (qué consulta devolvió cada paper y en qué posición). Cada configuración
se rankea con **Conteo de Borda**, el mismo criterio del leaderboard oficial.

### 7.8 "¿Qué aporta el grafo?"
Relaciones **explícitas** entre entidades, que los embeddings solo capturan de forma implícita. En
recuperación extraemos las entidades de la consulta con el mismo NER, traemos los chunks del nodo
y de sus vecinos de primer orden, y fusionamos ese ranking como un índice más (Sección 8.5). Cada
arista referencia el `doc_id` y `chunk_id` que la respalda: toda relación es **trazable** a su
evidencia textual.

### 7.9 "¿Qué pasa si un documento del corpus está corrupto?"
Se omite, se reporta en consola y el resto del corpus se indexa igual. Un documento defectuoso no
puede tumbar la entrega completa.

### 7.10 "¿Sus métricas son buenas?"
Cuidado aquí: nuestros números (NDCG@10 ≈ 0.42) vienen de **juicios débiles e incompletos** —cada
consulta tiene ~5 documentos etiquetados de 50, así que cuando el sistema recupera documentos
relevantes *no etiquetados* cuentan como error. **Sirven para comparar configuraciones entre sí,
que es su propósito, no como estimación del puntaje real.**

---

## 8. Runbook

```bash
# Entorno (Python 3.13, .venv)
python -m venv .venv && .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # GPU
pip install -r requirements.txt

# Verificación rápida (sin modelos, ~0.2 s)
python -m pytest                      # 48 tests

# Ingesta sin modelos
python scripts/smoke_ingest.py

# Corpus de prueba real (50 PDFs, ~110 MB)
python scripts/fetch_corpus.py --out data/corpus --per-topic 5
python scripts/make_eval_corpus.py

# Indexar y evaluar
python scripts/build_index.py --config config.corpus.yaml
python scripts/eval_corpus.py  --config config.corpus.yaml --with-rerank

# Generar el entregable (generador.py vive en entrega/ por la estructura de la
# Seccion 1.4, pero se ejecuta desde la raiz, donde estan config y src/)
python entrega/generador.py --config config.adl.yaml --pausa 2

# Informe técnico (PDF)
cd docs && xelatex informe_tecnico.tex
```

---

## 9. Resultados medidos

**Corpus de prueba:** 50 documentos públicos reales (111 MB) → **2 601 fragmentos**, 0 fallos,
0 errores de esquema. Fuentes: arXiv, ESA Space Environment Report, UNOOSA/IADC, CEPAL.

| Configuración | NDCG@10 | F1@3 | Borda |
|---|---|---|---|
| max_pool + reranking | **0.4246** | **0.5000** | **8** |
| media ponderada | 0.3625 | 0.5000 | 6 |
| max_pool | 0.3625 | 0.5000 | 6 |
| media ponderada + reranking | 0.4246 | 0.4697 | 5 |
| suma | 0.3625 | 0.4697 | 3 |

*(La comparación denso vs híbrido denso+disperso está en curso — ver `BITACORA.md`.)*

**En el corpus sintético pequeño** (para contraste): BGE-M3 0.990 vs MiniLM 0.966; el reranking
allí **no aportaba** porque las métricas saturaban. Lección: los corpus de juguete engañan.

---

## 10. Estado de los entregables

| # | Entregable | Estado |
|---|---|---|
| 1 | Base vectorial (`index.faiss` + `metadata.jsonl`) | ✅ Generada y validada (alineación 1:1) |
| 2 | `resultados.jsonl` | ✅ Pipeline validado, 0 errores de esquema |
| 3 | `informe_tecnico.pdf` | ✅ 4 páginas (límite 8) |
| 4 | `generador.py` | ✅ Determinista |
| 5 | `grafo.graphml` (bonus) | ✅ 54 nodos / 125 aristas en prueba |

---

## 11. Pendientes y palancas sin explotar

| Pendiente | Impacto esperado | Nota |
|---|---|---|
| **Híbrido denso+disperso** | Alto (+26–31% NDCG en benchmarks) | **En curso** |
| Barrido chunk size × overlap × top_k en corpus real | Alto | No hay óptimo universal: hay que medirlo |
| Eval set ampliado con *pooling* | Habilitador | Con 11 consultas, ±3% es ruido |
| ColBERT multi-vector como tercera señal | Medio | BGE-M3 ya lo produce gratis |
| Diversificación MMR del top-10 | Medio | Operación vectorial (permitida) |
| Empaquetado a 250 palabras (Sec. 9.2.1) | Incierto | Permitido explícitamente, sin medir |
| **Corpus real de ADL** | Bloqueante | Ajustar mapeo `fuente`/fenómeno a su layout |
| CI de tests | — | Bloqueado: falta scope `workflow` en el token |
| Respuesta del jurado sobre reranker/OCR | — | Correo enviado |

---

## 12. Glosario

| Término | Significado |
|---|---|
| **Chunk / fragmento** | Porción de texto indexada individualmente |
| **Embedding denso** | Vector numérico que representa el significado del texto |
| **Señal dispersa (lexical)** | Pesos por token; captura coincidencia **exacta** de términos |
| **Cross-encoder** | Modelo que puntúa el par (consulta, fragmento) leyéndolos juntos |
| **Bi-encoder** | Codifica consulta y documento por separado (permite indexar) |
| **RRF** | *Reciprocal Rank Fusion*: fusiona rankings usando posiciones, no scores |
| **NDCG@10** | Calidad del orden del ranking, penaliza errores arriba |
| **F1@3** | Acierto del conjunto de 3 documentos (no importa el orden) |
| **Conteo de Borda** | Método de votación por posiciones; fusiona ambas tablas |
| **Pooling (evaluación)** | Juzgar la unión de lo que recuperan varios sistemas |
| **Supervisión débil** | Etiquetas aproximadas obtenidas de una señal existente |
