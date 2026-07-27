# BITÁCORA — Trazabilidad de decisiones y cambios

> Registro cronológico: **qué** se cambió, **por qué**, **dónde** quedó y **qué resultado** dio.
> Para la referencia del sistema ver `SISTEMA.md`; para el entregable ver `informe_tecnico.pdf`.

Repositorio: `JuanMCanchala/codefest-adastra-2026` · rama principal `main`

---

## Resumen ejecutivo de la evolución

| Fase | Qué se logró | Resultado medible |
|---|---|---|
| 1. Scaffold | Core verificable sin dependencias pesadas | 32 tests |
| 2. Capa ML | BGE-M3 en GPU + FAISS end-to-end | NDCG 0.990 (proxy) |
| 3. Entregable | `resultados.jsonl` válido y reproducible | 0 errores de esquema |
| 4. Grafo bonus | NER + fusión al retrieval | 54 nodos / 125 aristas |
| 5. Corpus real | 50 documentos públicos, métricas con señal | NDCG 0.4246 · F1 0.50 |
| 6. Informe | Entregable 3 en PDF | 4 páginas (límite 8) |
| 7. Híbrido | Señal lexical de BGE-M3 | *en curso* |

---

## Decisiones estratégicas (las que definieron el rumbo)

### D1 — Priorizar extracción sobre afinado del recuperador
**Por qué:** la Sección 10.2.1 dice que la relevancia se juzga sobre el **contenido** del fragmento
y sobre el campo `fuente`, no sobre identificadores. Eso significa que la fidelidad del texto pone
el techo de las métricas.
**Consecuencia:** se invirtió más esfuerzo en extracción/limpieza que en tuning del ranking.

### D2 — BGE-M3 como encoder principal
**Por qué:** el reto es cross-lingual (consultas ES/EN/PT contra documentos en los tres idiomas).
**Evidencia:** 0.990 vs 0.966 de NDCG@10 contra un multilingüe pequeño; la diferencia se concentró
en la consulta en portugués (1.000 vs 0.877).

### D3 — Reranker cross-encoder tras un interruptor
**Por qué:** es la mayor ganancia individual medida (+17%), pero vive en zona gris del reglamento.
**Cómo se gestionó:** implementado como toggle (`rerank.enabled`), documentado en el informe, y
**consultado formalmente al jurado** por correo junto con la duda del OCR generativo.

### D4 — Construir un arnés de evaluación propio
**Por qué:** sin ground truth público, ajustar hiperparámetros "a ojo" equivale a no ajustarlos.
**Resultado:** detectó dos defectos que ninguna inspección visual habría encontrado (ver C2 y C3).

### D5 — Validar con un corpus real, no sintético
**Por qué:** el proxy sintético saturaba todas las métricas en ~0.95.
**Resultado:** cambió una conclusión de diseño (ver R2). Fue la decisión metodológica más rentable.

---

## Correcciones importantes (bugs que habrían costado puntos)

### C1 — `fuente` con separador de Windows
- **Síntoma:** F1@3 = 0.000 pese a que la recuperación era correcta.
- **Causa:** en Windows la ruta se guardaba como `fenomeno1\archivo.md`; el emparejamiento espera `/`.
- **Impacto:** habría anulado **toda** la métrica de documentos (la mitad del puntaje).
- **Fix:** `path.relative_to(raw_dir).as_posix()` en `scripts/build_index.py`.

### C2 — NDCG > 1 (matemáticamente imposible)
- **Síntoma:** el arnés reportaba NDCG@10 = 1.4681.
- **Causa:** el ranking ideal (IDCG) contaba los grados de documentos *distintos*, pero varios
  fragmentos del mismo documento heredan su grado, así que el DCG obtenido superaba al ideal.
- **Fix:** el IDCG ahora replica cada grado tantas veces como fragmentos tenga esa fuente
  (`src/eval/harness.py`), acotado a [0,1]. Con test de regresión.

### C3 — `extraction.pdf_backend` del config se ignoraba
- **Síntoma:** el build sobre 50 PDFs cargaba modelos de layout de docling documento por documento.
- **Causa:** `base.extract()` nunca recibía el backend configurado; usaba siempre el default.
- **Impacto:** ~50× más lento de lo necesario en corpus grandes.
- **Fix:** parámetro `pdf_backend` propagado desde `config.yaml`.

### C4 — `.gitignore` inválido dejaba 111 MB sin ignorar
- **Causa:** gitignore **no admite comentarios al final de línea**; el patrón
  `data/corpus/   # comentario` no coincidía con nada.
- **Fix:** comentario movido a su propia línea. Detectado antes de commitear.

### C5 — Reranker incompatible con la versión de librerías
- **Síntoma:** `AttributeError: XLMRobertaTokenizer has no attribute prepare_for_model`.
- **Causa:** incompatibilidad entre `FlagEmbedding.FlagReranker` y la versión de `transformers`.
- **Fix:** se cambió a `CrossEncoder` de sentence-transformers — **mismo modelo**, cargador robusto.

### C7 — torch CUDA sobrescrito en silencio por el build de CPU
- **Síntoma:** `torch.cuda.is_available()` devolvía `False` pese a haber GPU libre. Todo el
  pipeline corría ~10× más lento (build de 2 601 fragmentos: ~20 min en vez de ~2).
- **Diagnóstico inicial equivocado:** se atribuyó a "procesos huérfanos" que degradaban el driver.
  Falso — esos procesos eran de otras aplicaciones del usuario y la GPU estaba sana (667 MiB de
  8 188 en uso).
- **Causa real:** instalamos `torch 2.6.0+cu124` y funcionaba. Al instalar después `docling` y
  `gliner`, pip resolvió su dependencia de torch y **lo reemplazó por `2.13.0+cpu`** (el wheel por
  defecto de PyPI en Windows). Sin error visible: simplemente dejó de haber GPU.
- **Fix:** `pip install torch==2.13.0+cu126 --index-url .../whl/cu126`. Se eligió **la misma
  versión** (2.13.0) para no romper las dependencias que la exigían. Nota: `pip install torch==2.13.0`
  **no** basta — pip considera el requisito satisfecho e ignora el sufijo `+cpu`; hay que
  especificar la versión local completa y forzar la reinstalación.
- **Prevención:** `resolve_device()` ahora **avisa por consola** si torch es un build de CPU
  habiendo una GPU NVIDIA en la máquina.
- **Lección:** instalar cualquier paquete que dependa de torch puede degradar el entorno en
  silencio. Verificar `torch.cuda.is_available()` después de cada instalación.

### C6 — Build frágil ante documentos corruptos
- **Fix:** cada documento se procesa en un `try/except`; si falla se omite y se reporta, y el resto
  del corpus se indexa igual.

---

## Resultados y hallazgos medidos

### R1 — El reranking es la palanca más rentable (corpus real)
| Configuración | NDCG@10 | F1@3 | Borda |
|---|---|---|---|
| **max_pool + reranking** | **0.4246** | **0.5000** | **8** |
| max_pool | 0.3625 | 0.5000 | 6 |
| suma | 0.3625 | 0.4697 | 3 |

**+17% relativo de NDCG@10** sin degradar F1@3.

### R2 — Los corpus de juguete engañan
Sobre el proxy sintético, el reranking **no aportaba nada** e incluso perjudicaba ligeramente
(0.9446 → 0.9357), porque con pocos documentos y consultas evidentes el recuperador denso ya
saturaba las métricas. **De haber confiado en ese conjunto habríamos descartado el mejor
componente del sistema.** Coincide con literatura externa: las mejoras del cross-encoder no son
universales (se observan caídas en FEVER, Climate-FEVER, ArguAna, SCI-DOCS).

### R3 — La agregación por suma perjudica el F1@3
Replicado en **ambos** corpus (0.4697 vs 0.5000). Favorece documentos largos con muchos fragmentos
mediocres sobre documentos concisos con un fragmento excelente. → se adoptó `max_pool`.

### R4 — La limpieza tiene impacto grande y barato
Sobre un PDF institucional real: falsas oraciones detectadas **279 → 59** en los primeros 3 000
caracteres, y desaparecieron los fragmentos compuestos únicamente por el índice del documento.

### R5 — Docling supera a PyMuPDF en fidelidad
+13% de contenido extraído y fue el único que recuperó una tabla como estructura tabular.
**Coste:** ~61 s/documento en CPU. → docling para la entrega, pymupdf para iterar rápido.

### R7 — El híbrido denso+disperso solo funciona CON reranker
| Configuración | NDCG@10 | F1@3 | Borda |
|---|---|---|---|
| **híbrido + reranking** | **0.4378** | **0.5303** | **10** |
| denso + reranking | 0.4246 | 0.5000 | 8 |
| denso solo | 0.3625 | 0.5000 | 7 |
| **híbrido solo** | **0.2817** | 0.4697 | 4 |

**Interpretación mecánica:** la señal dispersa amplía el conjunto de candidatos con documentos
léxicamente similares. Sin reranker, ese ruido contamina el orden final y el NDCG **cae un 22%**.
Con reranker, el cross-encoder aprovecha los candidatos relevantes que el denso no encontró y
descarta el resto. Es decir: **el disperso aporta *recall*, el reranker lo convierte en
*precisión*** — el patrón clásico de "primera etapa amplia + reordenador preciso".

**Implicación estratégica — la configuración óptima depende de la respuesta del jurado:**

| Si el jurado… | Configuración correcta | NDCG@10 |
|---|---|---|
| **permite** el reranker | híbrido + reranking | 0.4378 |
| **veta** el reranker | **denso solo** (NO híbrido) | 0.3625 |

Activar el disperso sin reranker sería un error grave (0.2817).

**Cautela estadística:** con 11 consultas, la mejora de 0.4246 → 0.4378 (+3%) está dentro del
ruido. La caída del híbrido solo (−22%) sí es inequívoca. Por eso se priorizó **ampliar el
conjunto de evaluación** antes de tomar más decisiones.

### R6 — Multi-encoder (BGE-M3 + E5) no aportó en el proxy
0.9446 (BGE-M3 solo) vs 0.9394 (con E5). **Pendiente de re-medir en corpus real** — el proxy ya
demostró ser mal juez (ver R2).

---

## Validación contra literatura externa

Se contrastaron las recomendaciones con publicaciones y sistemas ganadores de competencias:

| Recomendación | Evidencia externa |
|---|---|
| Híbrido denso+disperso | +15–30% recall; **+26–31% NDCG** sobre denso-solo en benchmarks mixtos |
| RRF con k₀=60 | Descrito como el método de fusión por defecto en producción |
| Barrer chunk size | *"Diferentes tareas RAG tienen distintos tamaños óptimos"* — no hay óptimo universal |
| Reranking en dos etapas | Supera consistentemente a búsqueda vectorial simple, **pero no universalmente** |
| Eval con pooling | Metodología estándar de TREC |

**Nota sobre el ganador de TREC TOT 2025** (NDCG 0.4106): usó primera etapa híbrida
(BM25 + BGE-M3 + LLM) y reranking listwise con Gemini. **Dos de sus tres piezas nos están
prohibidas** (Sec. 8.3). La parte aprovechable —y legal— es exactamente el híbrido denso+disperso.

---

## Historial de commits

| Commit | Fecha | Contenido |
|---|---|---|
| `fabd48a` | 26-jul | Scaffold completo: métricas, esquema, chunking, fusión, extracción, grafo (32 tests) |
| `df0a3e9` | 26-jul | Reranker robusto (CrossEncoder) + comparador multi-encoder |
| `1955bda` | 26-jul | Grafo de conocimiento (bonus): fusión grafo→retrieval (Sec. 8.5) |
| `bce8d4a` | 26-jul | Corpus real (50 docs) + eval set con relevancia + borrador del informe |
| `258d4d0` | 26-jul | Cablea `pdf_backend` + build resiliente + evaluador de corpus |
| `6bfc1b4` | 26-jul | Informe técnico completo (PDF, 4 págs) con resultados sobre corpus real |

**Rama `feat/unlimited-ocr`:** adaptador para `baidu/Unlimited-OCR` (3B, MIT) **parqueado**. No se
integró por riesgo de fidelidad literal (es generativo y la evaluación compara el campo `text`),
zona gris del reglamento y coste de VRAM. Ver `NOTA_UNLIMITED_OCR.md`.

---

## Consultas formales al jurado

Enviado por correo (26-jul), pendiente de respuesta:

1. **Reranking con cross-encoder** (`bge-reranker-v2-m3`): ¿admisible, siendo arquitectura encoder
   y no generando texto, dado que la Sec. 8.3 prohíbe reranking *"mediante un LLM"* pero también
   dice que la recuperación debe operar *"exclusivamente sobre vectores, puntuaciones y metadata"*?
2. **OCR con modelo generativo** en preprocesamiento: ¿admisible, dado que la Sec. 2.1 recomienda
   OCR pero la Sec. 4.2 restringe decoders en la *"construcción del índice"*?

**Plan de contingencia:** ambos componentes son desactivables por configuración. Si el jurado los
veta, `rerank.enabled: false` y el sistema opera con fusión RRF pura sin cambios de código.

---

## Limitaciones conocidas (ser honesto si preguntan)

1. **Los juicios de relevancia del corpus de prueba son incompletos.** ~5 documentos etiquetados
   por consulta de 50; cuando el sistema recupera documentos relevantes no etiquetados, cuentan
   como error. Las cifras sirven para **comparar configuraciones**, no como calidad absoluta.
2. **Solo 11 consultas de evaluación.** Diferencias menores a ~3% no son distinguibles del ruido.
3. **Aún no se barrió el tamaño de chunk en el corpus real** (solo en el proxy, que demostró ser
   mal juez).
4. **El corpus oficial de ADL aún no está disponible**; habrá que ajustar el mapeo de `fuente` y
   `fenomeno` a su estructura real.
5. **GPU intermitente:** `torch.cuda.is_available()` devuelve `False` pese a haber VRAM libre; las
   últimas corridas fueron en CPU (~20 min por build de 2 601 fragmentos).
