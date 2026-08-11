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

## El corpus oficial de ADL (llegó el 3-ago-2026)

**Qué trajo:** 1826 documentos (2,93 GB) de 20 observatorios, más dos regalos que no esperábamos:
el **PDF con las 50 consultas reales** de evaluación y un **inventario oficial** (`DOC_ID` único
por archivo).

**En qué difería de nuestras suposiciones:**

| Asumíamos | Realidad |
|---|---|
| Formato dominante PDF | **JSON (964, el 52%)**; PDF 759 |
| Consultas repartidas ES/EN/PT | **Las 50 en español** |
| `doc_id` a inventar por nosotros | **ADL lo entrega** |
| Rutas `fenomeno1/` | `F1_`, `F2_`, `F3_` |

**El inventario elimina el mayor riesgo del reto.** La evaluación a nivel documento empareja por
`fuente` (Sección 10.2.1); con el inventario usamos los identificadores del propio organizador en
vez de adivinarlos. 1826/1839 archivos emparejados (99%).

**Las consultas reales validan la apuesta del híbrido.** Están llenas de siglas y nombres propios
—`NBQR`, `RPO`, `GAO`, `GAOR`, `GDO`, `Chocó`, `Arauca`— justo donde el vector denso se diluye y
la señal lexical rinde.

**Resultado del pipeline sobre el corpus real:** 82 673 fragmentos de 1688 documentos, metadata
alineada 1:1, índice disperso con 59 673 tokens únicos, y `resultados.jsonl` de 50 líneas con
**cero errores de esquema**.

**Adaptaciones necesarias:** extractor de JSON reescrito (los esquemas son heterogéneos y el
anterior perdía observatorios enteros), extractor de PBF implementado, extracción paralela con
caché (2,7 h → 1,2 min) y los cinco defectos C8–C12 documentados abajo.

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

### C12 — El PDF con las 50 consultas se indexó como documento del corpus
- **Síntoma:** en la revisión cualitativa, `Extracto_Preguntas_50_v2.pdf` aparecía como documento
  nº 1 para q001.
- **Causa:** el paquete de ADL incluye, junto al corpus, el PDF con las consultas de evaluación y
  el inventario. Al descomprimir todo en la misma carpeta quedaron mezclados.
- **Impacto medido:** aparecía en el top-3 de **20 de las 50 consultas (40%)** —empareja de forma
  trivial porque contiene el texto literal de cada consulta— gastando uno de los **tres únicos**
  huecos de documento. En esas consultas el F1@3 habría caído hasta un tercio.
- **Fix:** el **inventario oficial define el corpus**. Todo archivo ausente del inventario se
  omite y se reporta en el log. Quedaron fuera 13 archivos: el PDF de preguntas, el inventario,
  un xlsx de organización y 10 catálogos `*_catalogo.json` / `*_registro.json` que solo contienen
  URLs, hashes y títulos (verificado uno a uno: sin contenido analizable).
- **Verificación:** contaminación 20/50 → **0/50**.

### C11 — `pysbd` hacía inviable el corpus real (8 horas de segmentación)
- **Síntoma:** el build quedaba sin avanzar durante horas sobre el corpus de ADL.
- **Diagnóstico inicial equivocado:** se midió el chunking del corpus completo y dio 20 s… pero la
  medición se ejecutó con el **Python del sistema, que no tiene pysbd instalado** y usaba el
  segmentador de respaldo. El venv sí lo tiene. **Lección: medir con el mismo intérprete que
  ejecuta el proceso.**
- **Cifras:** pysbd rinde **7 K chars/s**; el segmentador por reglas, **64 M chars/s**. Sobre los
  204 MB del corpus son ~8 horas frente a ~3 segundos, con prácticamente las mismas oraciones
  (1047 vs 1157).
- **Fix:** el segmentador por reglas pasa a ser el predeterminado. Pero el regex original habría
  cortado en `Dr.`, `Fig. 3` o iniciales, generando oraciones incompletas que la Sección 3.3
  **prohíbe**; se añadió una guarda de abreviaturas, iniciales y numeración, con tests. Un archivo
  que tardaba 75 s pasó a 0,10 s.

### C10 — Un documento de 48 MB bloqueaba la segmentación
- 4 archivos concentraban 103 MB de los 204 del corpus. Pasar una cadena de 48 MB al segmentador
  dejaba el proceso efectivamente colgado.
- **Fix:** segmentación por bloques de 100 KB cortando en saltos de línea (preserva párrafos y
  filas, no parte oraciones).

### C9 — Seis documentos generaban el 57% del índice
- Tres volcados bibliográficos de PubMed en CSV producían **78 000 fragmentos** entre los tres,
  sobre biomedicina (cromatografía, vasos retinianos) ajena a las 50 consultas.
- **Fix:** `chunking.max_chunks_per_doc` (3000). Solo afecta a 4 documentos de 1838 y queda
  registrado en el log del build.

### C8 — `extraction.ocr_enabled` se ignoraba
- El config desactivaba el OCR, pero `extract()` llamaba igual a PaddleOCR con las 8 imágenes del
  corpus, que se cuelga por un bug de oneDNN en Windows. El build quedaba bloqueado sin error.
- **Fix:** `extract()` respeta la opción; además `_extract_ocr` se migró a la API v3 de PaddleOCR
  (era código de la v2, nunca ejercitado hasta este corpus).

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

### R8 — El grafo con peso pleno EXPULSA los mejores documentos (corpus real, 50 consultas)

La primera vez que el recuperador por grafo corrió fuera de los tests fue sobre las 50 consultas
reales. Encontró dos defectos antes de generar nada:

1. **Sin tope de candidatos.** q005 devolvía **10 174 chunks**, mientras denso y disperso llegan
   topados a `top_k_faiss: 100`. Entidades muy conectadas (`colombia`, `artificial intelligence`)
   arrastraban medio corpus.
2. **Falsos positivos por subcadena.** La coincidencia parcial era `key in nombre` en crudo: `ia`
   casaba dentro de `colomb-ia`, y q013 emparejaba con un nodo `cas`. Ahora exige frontera de
   palabra y ≥4 caracteres, y elige el nodo más parecido en longitud en vez del primero según el
   orden de inserción del grafo. Tras el arreglo, el grafo aporta señal en **44 de 50** consultas.

**El A/B.** Tres corridas completas sobre el mismo índice, comparadas con `scripts/compare_resultados.py`:

| Indicador | Sin grafo | Grafo peso 1,0 | Grafo peso 0,3 |
|---|---|---|---|
| Consultas que cambian (vs sin grafo) | — | 27/50 | **2/50** |
| Solape de documentos | — | 90,0 % | 99,3 % |
| Coherencia temática | 84,0 % | 83,3 % | **84,0 %** |
| Documentos distintos en el top-3 | 94 | 93 | 93 |
| Consultas con un solo observatorio | 18 | 19 | **18** |

Con peso pleno el grafo mueve 27 consultas **y no mejora ni un indicador**. Pero el dato agregado
escondía lo importante, que solo apareció al mirar fragmento a fragmento:

| Consulta | Peso 1,0 | Peso 0,3 |
|---|---|---|
| **q006** (riesgos de IA militar sin doctrina) | `F1-CSET-125`, que aportaba los fragmentos **1, 2 y 3**, **desaparece del top-3 de documentos**; entran gobernanza de la CCW y tecnología de defensa indonesia | Conserva CSET-125 y CSET-104 en el top-3 |
| **q023** (vulnerabilidades satelitales) | Gana el fragmento del malware *Orbitshade* (r9)… pero expulsa del top-3 a `F2-SWF-124`, que sin grafo era el documento nº 1 **y es de donde sale ese fragmento** | Conserva SWF-124 en el nº 1 |

**Causa raíz.** RRF pondera **solo por posición**: el primer elemento de cada ranking aporta
1/(k₀+1) venga de donde venga. Eso es correcto entre índices que miden lo mismo, pero el grafo
ordena por **coocurrencia de entidades** y el denso por **relevancia semántica**. Dándoles el mismo
voto, una señal de popularidad desplaza aciertos.

**Corrección.** Se añadió peso por ranking a las tres fórmulas de fusión (`rrf`, `combsum`,
`combmnz`). Con todos los pesos a 1 el resultado es idéntico a la ecuación 7 del enunciado, y hay
un test que lo verifica. El grafo entra con `graph.fusion_weight: 0.3`: rompe empates y aporta
evidencia, pero no puede desplazar por sí solo un acierto del denso. Sigue siendo aritmética sobre
posiciones, sin nada generativo.

**Por qué 0,3 y no un valor intermedio.** Sin juicios de relevancia sobre las 50 consultas solo
disponemos de indicadores débiles —la coherencia temática ya demostró engañar (ver la limitación
al final)—. Afinar el peso contra ellos sería sobreajustar a una métrica que no mide lo que se
evalúa. 0,3 es el valor que preserva la línea base en todas las consultas inspeccionadas y mantiene
el grafo integrado, que es lo que el bono exige.

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

## Consultas formales al jurado — RESPONDIDAS

Enviado por correo (26-jul). La respuesta llegó en el archivo *Preguntas Frecuentes* que ADL
publicó a la comunidad; nuestra consulta es la **fila 31** de ese archivo.

1. **Reranking con cross-encoder** (`bge-reranker-v2-m3`) → **PERMITIDO**, textual:
   > *"Sí está permitido re-ranking con cross-encoders. La restricción aplica es para arquitecturas
   > decoders."*

   Confirmado tres veces de forma independiente (filas 31, 43 y 58, esta última a otro equipo).
   La zona gris queda cerrada a nuestro favor: `rerank.enabled: true` se mantiene.
2. **OCR con modelo generativo**: no respondida de forma directa, pero la fila 42 aclara que
   *"pueden usar encoders de visión para crear directamente el vector de las imágenes; también
   pueden usar VLMs solo para generar descripciones textuales"*. Es decir, el uso de generativos en
   **indexación** está admitido. Aun así mantenemos `Unlimited-OCR` parqueado: usamos EasyOCR
   (CRAFT + CRNN, no generativo), que ya recuperó los 51 PDF escaneados sin riesgo de alucinación
   sobre el campo `text`, que es el que se evalúa.

**Plan de contingencia:** ambos componentes siguen siendo desactivables por configuración.

---

## Cambios derivados del archivo de Preguntas Frecuentes (10-ago)

ADL publicó 68 preguntas de la comunidad con respuestas del Ing. Francisco Manrique. Cinco puntos
afectaban decisiones ya tomadas:

| # | Lo que dice la FAQ | Nuestro estado | Acción |
|---|---|---|---|
| 1 | Cross-encoder permitido (filas 31/43/58) | `rerank.enabled: true` | Ninguna: validado |
| 2 | El emparejamiento es por `doc_id` de ADL; lo del handbook fue **errata de versionado** (filas 19/52/58) | Ya usábamos el `DOC_ID` del inventario | Ninguna: acertamos |
| 3 | El grafo *"es bono y para que sea válido lo deben integrar a la recuperación, el solo construirlo no es válido"* (fila 42) | `enabled: false` | **C17** |
| 4 | *"Deberían usar como `chunk_id` el mismo obtenido del índice FAISS"* (fila 42) | `F1-AIINDEX-015-chunk-0000` | **C16** |
| 5 | Usar la **extensión real en minúsculas** en `formato`; la Tabla 1 solo daba ejemplos (filas 21/60) | Ya lo hacíamos | Ninguna |

Confirmaciones adicionales sin impacto: híbrido BM25/léxico permitido (filas 28/49); se exigen
siempre 10 fragmentos y 3 documentos, *"si es necesario baja tu umbral"* (fila 42); se admiten
archivos Python adicionales a `generador.py` (fila 46); la entrega es un enlace a repositorio
público por formulario, sin Docker (fila 37); el corpus son 1 826 documentos — F1 459, F2 479,
F3 888 (fila 45), cifra que coincide con nuestro inventario.

### C13 — Los 73 mapas `.pbf` nunca entraron al índice

**Síntoma:** los 73 archivos `.pbf` del corpus tenían caché de texto de 0 bytes. Ningún error.

**Diagnóstico:** la FAQ (fila 21) responde que `.pbf` es *"el formato de OpenStreetMap
(Protocolbuffer Binary Format)"* y sugiere `pyrosm`/`pyosmium`. **Para estos archivos concretos la
respuesta no aplica.** Se verificó sobre los bytes: empiezan por
`1a … 0a 10 "au_compilado_R02"`, es decir campo 3 (*layers*) y campo 1 (*name*) del esquema
**Mapbox Vector Tile**; un `.osm.pbf` empezaría por la longitud del *BlobHeader* seguida de la
cadena `OSMHeader`, que no aparece. Además viven en una pirámide de teselas
`Amazon_Underworld/tiles/{z}/{x}/{y}.pbf`. Decodificado como vector tile, un solo archivo entrega
251 elementos con atributos reales (municipio, país, población y presencia de grupos armados:
`au_eln`, `au_cv`, `au_pcc`…). El extractor era el correcto; simplemente nunca llegó a ejecutarse
con la librería instalada.

**Corrección:** se pobló la caché (73/73, 2,9 MB) y se dejó `pyosmium` como respaldo por si algún
`.pbf` sí fuera de OSM. Lección: *contrastar la respuesta del organizador contra el archivo real
antes de reescribir código por ella.*

### C14 — Un documento sin puntos se indexaba como un único fragmento

**Síntoma:** al fragmentar los 73 `.pbf` recuperados salían **77 fragmentos de 77 documentos**: uno
por documento, de hasta 291 KB.

**Causa (dos capas):**
1. El extractor aplanaba el tile a pares `clave: valor` sueltos, destruyendo la estructura de
   registro. El texto resultante no tenía **ni un solo punto**, así que el segmentador devolvía el
   documento entero como una sola "oración".
2. `chunk_document()` documentaba que *"si una sola oración supera `index_max_tokens`, se emite
   sola"*. Con una oración de 291 KB, BGE-M3 solo veía sus primeros 8 192 tokens y **el resto del
   documento quedaba fuera del índice sin ningún aviso**.

**Corrección:** el extractor emite ahora **un registro por elemento** (`capa X | fid: … | país: …`),
igual que se hace con CSV/XLSX, deduplicando por registro completo porque el mismo municipio se
repite en varios niveles de zoom. Y `chunk_document()` trocea por palabras cualquier oración que
exceda el presupuesto. Resultado: 3 871 fragmentos de 77 documentos (de 1 a ~50 por documento).

### C15 — Once documentos del inventario fuera del índice

Auditoría de cobertura tras C13: el índice tenía 1 815 de los 1 826 documentos del inventario.

| Documento | Causa | Resolución |
|---|---|---|
| `F1-AIINDEX-041` | Un `.csv` que en realidad está **separado por tabuladores**; `pd.read_csv` lo rechazaba con `ParserError` | Reintento tolerante (`sep=None`, `on_bad_lines="skip"`) → **2,25 M caracteres, 15 115 registros** recuperados |
| `F1-CENIA-008/012/016/022` | Páginas de menú cuyo único texto útil es el título (13–31 caracteres), por debajo de `min_chunk_chars: 40` | `filter_min_chars()` conserva el mejor fragmento cuando el filtro dejaría el documento **sin ninguno**: un documento ausente del índice no puede recuperarse jamás |
| `F2-SWF-065` | Imagen `.avif`, extensión no registrada; `detect_format` la daba por formato desconocido y se saltaba **en silencio** | `.avif/.webp/.gif/.bmp` añadidos al mapa de formatos |
| `F2-SWF-066/067/068/071` | Fotografías sin texto (un retrato y fotos de misión de la NASA) | OCR ejecutado: devuelve vacío **correctamente**. No hay nada que indexar |
| `F1-DEFENSA21-001` | El archivo son literalmente 2 bytes: `[]` | Sin contenido posible |

**Estado final: 1 821 de 1 826 documentos indexados.** Los 5 restantes no tienen texto extraíble
por ningún medio (4 fotografías + 1 archivo vacío). Se migró además `_extract_ocr()` de PaddleOCR a
EasyOCR, el motor ya medido en `ocr_scanned.py` (6,5 s frente a 115 s por imagen en esta máquina).

### C16 — `chunk_id` no era el identificador de FAISS

La FAQ (fila 42) pide usar como `chunk_id` *"el mismo obtenido del índice FAISS"*. Usábamos
identificadores descriptivos (`F1-AIINDEX-015-chunk-0000`).

**No hizo falta recodificar nada.** FAISS asigna sus identificadores internos por orden de
inserción, y la base ya cumplía el invariante de que la línea *i* de `metadata.jsonl` es el vector
*i*; bastó renumerar el campo (`scripts/migrate_chunk_ids.py`). El identificador descriptivo se
conserva en `chunk_uid`, campo adicional que la especificación permite, para no perder la
trazabilidad a ojo. El índice disperso comparte ese orden de inserción, así que se reescribió su
lista y **se verificó la alineación**: si los dos espacios de identificadores divergieran, la fusión
RRF mezclaría fragmentos que no se corresponden y la señal léxica dejaría de sumar en silencio.

### C17 — El grafo estaba construido pero desconectado

`graph.enabled: false` y `fuse_into_retrieval: false`. Según la FAQ, el bono **no habría puntuado**:
*"es bono y para que sea válido lo deben integrar a la recuperación, el solo construirlo no es
válido"*. Ambos pasan a `true`, de modo que el grafo entra como un ranking más en la fusión RRF
(Sec. 8.5). Se añadió además un aviso explícito en `generador.py`: si el grafo está activado pero
falta `grafo.graphml`, antes se generaban resultados sin él sin decir nada.

### Herramienta nueva: indexación incremental

`scripts/append_docs.py`. Reconstruir la base completa cuesta horas de GPU y no aporta nada cuando
solo faltan unos pocos documentos. Como FAISS `IndexFlatIP` y el índice disperso admiten anexado, y
los identificadores se asignan por orden de inserción, **anexar al final no altera ninguno de los
existentes**: metadata, índice disperso y grafo siguen alineados. Los 1 475 + 3 871 fragmentos
nuevos se añadieron en minutos en vez de horas. Total: **90 613 fragmentos** (antes 85 267).

---

## Limitaciones conocidas (ser honesto si preguntan)

1. **Los juicios de relevancia del corpus de prueba son incompletos.** ~5 documentos etiquetados
   por consulta de 50; cuando el sistema recupera documentos relevantes no etiquetados, cuentan
   como error. Las cifras sirven para **comparar configuraciones**, no como calidad absoluta.
2. **Solo 11 consultas de evaluación.** Diferencias menores a ~3% no son distinguibles del ruido.
3. **Aún no se barrió el tamaño de chunk en el corpus real** (solo en el proxy, que demostró ser
   mal juez).
4. **Las 50 consultas reales no traen juicios de relevancia.** No se puede calcular NDCG@10 ni F1@3
   sobre ellas: ninguna cifra de este documento afirma que una corrida sea *mejor* que otra en la
   métrica del reto. Solo se compara **cuánto cambia** y se inspeccionan fragmentos a mano.
5. **La "coherencia temática" es un indicador sesgado — no fiarse de él a solas.** Asume que un
   documento relevante para una consulta de F1 vive en la carpeta F1. Es falso: SIPRI y CEEEP están
   archivados bajo F3 y publican precisamente sobre IA militar, así que q001 (IA frente a amenazas
   NBQR), q003 (IA en operaciones militares) y q007 (sistemas autónomos y DIH) aparecen como
   "descuadradas" devolviendo documentos **correctos** —SIPRI sobre mando y control nuclear (NC3),
   CEEEP sobre IA y desinformación en conflictos—. Mide dónde archivó ADL el observatorio, no de
   qué trata el documento.
6. **Cinco documentos del inventario no están indexados** y nunca podrán estarlo: cuatro son
   fotografías sin texto y el quinto es un archivo de dos bytes (`[]`). Cobertura real: 1 821/1 826.
7. **Aporte del grafo, medido y acotado.** Con peso 0,3 cambia 2 de las 50 consultas (ver R8). Está
   integrado a la recuperación —requisito del bono— y calibrado con evidencia para no dañar el
   ranking, pero su contribución positiva demostrable es pequeña.
8. **Riesgo térmico en portátil.** Con los tres modelos (encoder denso, cross-encoder y NER) en una
   GPU de 8 GB, `generador.py` alcanzó 83 °C sin pausas. Existe `--pausa` para acotar el pico a
   costa de unos minutos, y `build_graph.py` tiene `--batch/--pausa/--hilos`.
