# Informe Técnico — Base de Conocimiento Vectorial
## CODEFEST AD ASTRA 2026 · Etapa 1

---

## 1. Resumen

Construimos un sistema de **recuperación densa multilingüe** que indexa un corpus
multiformato (ES/EN/PT) sobre los tres fenómenos del reto y responde consultas en lenguaje
natural devolviendo los 3 documentos y los 10 fragmentos más relevantes.

El diseño parte de una lectura precisa de las reglas de evaluación (Sección 10.2.1 de la
especificación): **la relevancia de un fragmento se juzga sobre su contenido textual (`text`)
y la de un documento sobre el campo `fuente`**, no sobre los identificadores internos que
asigna cada equipo. De ahí se desprende nuestra prioridad de ingeniería: la **fidelidad de la
extracción y la limpieza del texto** determinan el techo de las métricas, por encima de
cualquier ajuste fino del recuperador.

**Arquitectura:**

```
Extracción → Limpieza → Chunking (2 niveles) → Encoder(s) → FAISS
     → Fusión RRF → [Reranking] → Agregación chunk→documento → resultados.jsonl
                                                    (+ Grafo de conocimiento)
```

---

## 2. Preprocesamiento: extracción y limpieza

### 2.1 Extracción por formato

| Formato | Herramienta | Motivo |
|---|---|---|
| PDF | **Docling** (IBM, MIT), respaldo **PyMuPDF** | Docling reconstruye layout, tablas y orden de lectura; PyMuPDF garantiza robustez si Docling falla |
| HTML | **trafilatura** | Elimina *boilerplate* (menús, pies, anuncios) conservando el cuerpo |
| JSON | selección explícita de campos | Concatena `title`/`body_text`/`body_paragraphs` en orden; `url`/`date`/`authors` van a metadata |
| CSV / XLSX | pandas + openpyxl | Cada fila es una unidad, serializada como pares `columna: valor` para preservar el contexto semántico de cada celda |
| Imágenes | OCR multilingüe | Recupera texto de infografías y figuras |
| PBF | recorrido de capas | Atributos de los elementos como pares `clave: valor`, una sola versión por elemento para no duplicar por nivel de zoom |

La elección de Docling se validó empíricamente: sobre un PDF institucional real de 24 páginas
extrajo **13 % más contenido** que PyMuPDF y fue el único que recuperó una tabla como
estructura tabular en lugar de texto aplanado.

### 2.2 Limpieza y normalización

Esta etapa resultó ser **la de mayor impacto medido** de todo el preprocesamiento. Al analizar
la extracción de un PDF real detectamos tres patrones de ruido que degradaban el índice:

1. **Índices con puntos de relleno** (`Introducción . . . . . . . 3`). El segmentador de
   oraciones interpretaba cada punto como fin de frase.
2. **Encabezados y pies repetidos** en cada página, que generaban fragmentos sin valor
   informativo y diluían la señal semántica.
3. **Guiones de corte de línea** (`conges-\ntión`), que parten palabras y rompen el
   emparejamiento léxico y semántico.

Nuestra limpieza (`src/cleaning/normalize.py`) normaliza a NFC, elimina caracteres de control,
descarta líneas de índice, detecta *boilerplate* por repetición (líneas cortas que aparecen en
muchas páginas) y **re-une palabras cortadas por guion** distinguiéndolas de compuestos
legítimos (`space-debris` no se altera).

**Efecto medido:** en los primeros 3 000 caracteres del documento de prueba, las falsas
oraciones detectadas pasaron de **279 a 59**, y desaparecieron los fragmentos compuestos
únicamente por el índice del documento.

---

## 3. Estrategia de chunking

### 3.1 Chunking de dos niveles

La especificación impone dos restricciones que operan en momentos distintos:

- **Completitud lingüística** (Sección 3.3): ningún fragmento puede contener oraciones
  incompletas; los cortes se hacen sólo en límites oracionales.
- **Límite de 250 palabras por fragmento de salida** (Sección 9.2).

Estas restricciones no son la misma: el tamaño óptimo para *codificar* (limitado por la ventana
del encoder) no coincide con el tamaño máximo de *presentación*. Por eso separamos:

**Nivel 1 — chunk de índice.** Agrupa oraciones completas hasta un máximo configurable de
tokens (por defecto 256–384, dentro del límite del encoder), con solapamiento de una oración
entre fragmentos consecutivos. Es la unidad que se codifica y se almacena en FAISS.

**Nivel 2 — sub-fragmento de salida.** Al construir la respuesta, un chunk que exceda las 250
palabras se divide respetando de nuevo los límites oracionales. Todos los sub-fragmentos
conservan el `chunk_id` del chunk original, que cumple una función de **trazabilidad** hacia la
evidencia indexada (la evaluación empareja por `text`, no por identificador).

### 3.2 Justificación

- **Frente a tamaño fijo de tokens:** cortar por conteo parte oraciones y viola el requisito
  obligatorio de la Sección 3.3.
- **Frente a chunking por párrafo puro:** los párrafos de documentos técnicos varían de una
  línea a más de una página; produce fragmentos muy desiguales y algunos superan la ventana
  del encoder.
- **A favor del solapamiento:** una idea que cae en la frontera entre dos fragmentos quedaría
  representada a medias en ambos; el solapamiento de una oración mitiga ese efecto con un
  coste de almacenamiento bajo.

La segmentación oracional usa **pysbd** (multilingüe), con un segmentador de respaldo basado
en reglas para garantizar que el sistema opere aunque falte la dependencia.

### 3.3 Metadata por fragmento

Cada chunk almacena los ocho campos obligatorios de la Tabla 1 (`doc_id`, `chunk_id`, `fuente`,
`formato`, `fenomeno`, `posicion`, `num_tokens`, `texto`) más campos opcionales (`idioma`,
`titulo`, `fecha`). El campo `fuente` se normaliza siempre con separadores `/`: durante el
desarrollo detectamos que en Windows se guardaba con `\`, lo que **anulaba el emparejamiento
a nivel documento** y llevaba el F1@3 a cero pese a una recuperación correcta.

---

## 4. Codificación semántica: selección de encoder

### 4.1 Encoder principal: BAAI/bge-m3

| Criterio (Sección 4.3) | Cumplimiento |
|---|---|
| Soporte multilingüe | Nativo en ES/EN/PT, entrenado para recuperación cross-lingual |
| Dimensionalidad | 1024 — equilibrio entre expresividad y coste |
| Longitud de entrada | Hasta 8 192 tokens, muy por encima de los 512 habituales |
| Benchmarks | Estado del arte en MIRACL/MTEB multilingüe para recuperación densa |
| Licencia | **MIT** |
| Eficiencia | Ejecuta en una GPU de 8 GB en fp16 |

El criterio decisivo es el **cross-lingual**: el conjunto de evaluación distribuye las consultas
entre español, inglés y portugués, y una consulta en un idioma debe recuperar documentos en
los otros dos. Un encoder monolingüe o con alineación débil entre idiomas fallaría en dos
tercios de las consultas.

**Validación empírica.** Comparamos BGE-M3 contra un modelo multilingüe pequeño
(`paraphrase-multilingual-MiniLM-L12-v2`) sobre el mismo conjunto: NDCG@10 de **0,990 frente a
0,966**. La diferencia se concentró exactamente donde se predijo: la consulta en portugués pasó
de 0,877 a **1,000**.

### 4.2 Restricción de arquitectura

Todos los modelos empleados en indexación y recuperación son **encoders** (familia BERT). No se
utiliza ningún modelo generativo (decoder) en la construcción del índice ni en la recuperación,
conforme a las Secciones 4.2 y 8.3.

### 4.3 Múltiples encoders

El sistema admite varios encoders en paralelo, cada uno con su propio índice FAISS, fusionados
en recuperación (Sección 5.2). Evaluamos la combinación BGE-M3 + `multilingual-e5-large`.

---

## 5. Índice vectorial FAISS

### 5.1 Tipo de índice: IndexFlatIP

Elegimos **`IndexFlatIP` con vectores normalizados a norma unitaria**, que equivale
exactamente a similitud coseno (ecuación 4 de la especificación).

**Justificación:** para el volumen de este reto, la búsqueda exhaustiva es holgadamente viable y
ofrece resultados **exactos**. Los índices aproximados (`IVFFlat`, `HNSW`) sacrifican exactitud
para ganar velocidad, un intercambio que no compensa cuando la métrica evaluada es la calidad
del ranking y el tiempo de respuesta no se puntúa. Si el corpus creciera en un orden de
magnitud, la migración a `IndexHNSWFlat` es un cambio de una línea en la configuración.

### 5.2 Índice y almacén de metadata

FAISS almacena únicamente vectores e identificadores internos enteros. La metadata vive en
`metadata.jsonl`, donde **la línea *i* corresponde al identificador interno *i*** que FAISS
asigna al indexar, tal como exige el formato de entrega. La persistencia usa las funciones
nativas `write_index`/`read_index`, de modo que el índice se recarga sin reindexar el corpus.

---

## 6. Módulo de recuperación

### 6.1 Fusión de múltiples índices

Cuando hay más de un encoder, cada índice produce su propio ranking. Implementamos las tres
estrategias de la Sección 8.4 (CombSUM, CombMNZ y RRF) y adoptamos **Reciprocal Rank Fusion**
como estrategia por defecto:

$$s_{\text{RRF}}(c) = \sum_{j=1}^{m} \frac{1}{k_0 + r_j(c)}, \quad k_0 = 60$$

RRF opera sobre **posiciones** y no sobre puntuaciones, por lo que es robusto frente a las
diferentes escalas de similitud que producen encoders distintos —un problema real, ya que
CombSUM permitiría que el encoder con puntuaciones sistemáticamente más altas domine la fusión.

### 6.2 Agregación de fragmentos a documentos

Para el nivel documento (F1@3) agrupamos los chunks recuperados por `doc_id` y calculamos una
puntuación agregada. Están implementadas tres estrategias —máximo (*max pooling*), suma y media
ponderada por posición— seleccionables por configuración y comparadas empíricamente
(Sección 8).

### 6.3 Reranking con cross-encoder (opcional)

El sistema incluye un paso opcional de reordenamiento con `BAAI/bge-reranker-v2-m3`, un
**cross-encoder** que puntúa conjuntamente el par (consulta, fragmento).

**Consideración sobre el reglamento.** La Sección 8.3 prohíbe el uso de modelos **generativos**
(decoders) en la recuperación. Un cross-encoder es arquitectura **encoder** (familia BERT) y
produce una puntuación escalar de relevancia; no genera texto. Entendemos que su uso es
admisible, pero por prudencia el paso es **desactivable mediante configuración**
(`rerank.enabled`) y, ante una interpretación restrictiva del jurado, el sistema opera con
fusión RRF pura sin ninguna modificación de código.

### 6.4 Cumplimiento del formato de salida

Un validador estricto verifica cada objeto antes de escribir `resultados.jsonl`: exactamente 3
documentos, exactamente 10 fragmentos, ningún fragmento por encima de 250 palabras y orden
`q001`–`q050`. Esta verificación es automática y forma parte del pipeline de generación.

---

## 7. Grafo de conocimiento (componente bonus)

### 7.1 Construcción

1. **Reconocimiento de entidades:** `GLiNER` multilingüe (MIT), un modelo *zero-shot* que
   extrae tipos de entidad definidos en configuración (persona, organización, país, tecnología,
   evento, lugar) sin necesidad de reentrenamiento. Verificamos su comportamiento cross-lingual:
   sobre texto en español identificó correctamente `Estados Unidos` (país), `sistemas de armas
   autónomas` (tecnología) y `Convenio de Ginebra` (evento); sobre texto en inglés,
   `autonomous weapons systems` y `arms race`.
2. **Extracción de relaciones:** por co-ocurrencia dentro del mismo fragmento, con peso
   acumulado por número de coincidencias. **No se emplea ningún modelo generativo**; las
   relaciones surgen de evidencia textual verificable.
3. **Persistencia:** grafo dirigido en NetworkX, exportado a `grafo.graphml`. Cada nodo guarda
   los `chunk_id` donde aparece la entidad y cada arista referencia el `doc_id` y `chunk_id` que
   la respalda, garantizando la **trazabilidad** de toda relación hasta su evidencia textual.

### 7.2 Integración con la recuperación

El grafo actúa como un **índice adicional** dentro del esquema de fusión (Sección 8.5): se
extraen las entidades de la consulta con el mismo NER, se recuperan los chunks vinculados a esas
entidades y a sus **vecinos de primer orden**, se puntúan por evidencia acumulada (peso pleno
para el nodo directo, reducido para los vecinos) y el ranking resultante se fusiona con los
vectoriales mediante RRF.

Esto aporta una señal que los embeddings sólo capturan de forma implícita: **relaciones
explícitas entre entidades**, útil en consultas centradas en actores concretos.

---

## 8. Metodología de evaluación y resultados

### 8.1 El problema de optimizar sin ground truth

El conjunto de evaluación y sus juicios de relevancia no son públicos durante el reto. Ajustar
hiperparámetros «a ojo» equivale a no ajustarlos. Nuestra respuesta fue construir un
**arnés de evaluación interno**:

1. Un conjunto de consultas propio con relevancia conocida, equilibrado entre los tres fenómenos
   y los tres idiomas, replicando la distribución descrita en la Sección 10.1.
2. Una implementación de **NDCG@10 y F1@3 idéntica a la especificación** (ecuaciones 8–14),
   verificada con pruebas unitarias, incluida la reproducción exacta del ejemplo de Conteo de
   Borda de la Tabla 3.
3. Un barrido de configuraciones que rankea cada variante con **Conteo de Borda**, el mismo
   criterio del leaderboard oficial: así optimizamos el equilibrio entre ambas métricas y no una
   sola.

Este arnés detectó dos defectos que habrían costado puntos y que ninguna inspección visual habría
revelado: el problema del separador en `fuente` (Sección 3.3) y un error en el cálculo del
ranking ideal del NDCG que producía valores superiores a 1.

### 8.2 Corpus de validación

Para validar el sistema en condiciones realistas antes de disponer del corpus oficial,
construimos un conjunto de prueba con **50 documentos públicos auténticos** (111 MB) sobre los
tres fenómenos:

| Fenómeno | Documentos | Fuentes |
|---|---|---|
| 1 · IA en defensa | 15 | Artículos de arXiv sobre armas autónomas, IA militar y gobernanza de IA |
| 2 · Seguridad espacial y LEO | 21 | arXiv sobre desechos orbitales y colisiones + **ESA Space Environment Report** + informe IADC de **UNOOSA** |
| 3 · Dinámicas territoriales | 14 | arXiv sobre desigualdad, violencia y migración en América Latina + **Panorama Social de la CEPAL** |

Este corpus reproduce las dificultades del escenario real: documentos **largos** (el informe de
la CEPAL supera los 980 000 caracteres y produce 654 fragmentos por sí solo), **multilingües**
(español, inglés y portugués), con estructura académica e institucional real —resúmenes,
secciones, referencias, tablas, encabezados repetidos.

**Juicios de relevancia.** Al no poder anotar 50 documentos a mano, aprovechamos una señal
existente: el propio buscador de arXiv. Si la consulta `all:"space debris"` devolvió un
artículo, ese artículo es relevante para una consulta en lenguaje natural sobre desechos
espaciales, y su **posición en los resultados** da el grado de relevancia (posiciones 1–2 → 3,
3–4 → 2, resto → 1). Es supervisión débil, pero es una señal real e independiente de nuestro
sistema, lo que evita el sesgo circular de evaluarnos con juicios que nosotros mismos hubiéramos
producido. Las consultas se redactaron en lenguaje natural repartidas entre **español, inglés y
portugués**, deliberadamente en idiomas distintos al de los documentos para forzar el escenario
cross-lingual.

### 8.3 Resultados

[PENDIENTE_RESULTADOS]

---

## 9. Reproducibilidad

El script `generador.py` reconstruye `resultados.jsonl` a partir del índice persistido y del
archivo de consultas. El determinismo se asegura fijando las semillas de `random`, NumPy y
PyTorch, activando algoritmos deterministas y escribiendo el JSON Lines con separadores fijos y
codificación UTF-8 explícita. Toda la configuración —estrategia de chunking, encoders, tipo de
índice, fusión, agregación— reside en un único archivo `config.yaml`, de modo que cualquier
resultado reportado es rastreable a una configuración concreta.

El proyecto cuenta con **[N_TESTS] pruebas automatizadas** que cubren las métricas, el validador
de esquema, el chunker, las estrategias de fusión, la limpieza y el recuperador basado en grafo.

---

## 10. Conclusiones

La decisión de diseño más rentable de este trabajo no fue la elección del modelo, sino **leer la
regla de emparejamiento y orientar el esfuerzo a la fidelidad del texto**. La limpieza del
corpus produjo la mejora más grande y barata; el encoder correcto (BGE-M3) resolvió el requisito
cross-lingual; y el arnés de evaluación interno convirtió decisiones de intuición en decisiones
medidas.

Los componentes de mayor riesgo regulatorio —el reranking con cross-encoder— quedan aislados
tras interruptores de configuración, de modo que el sistema puede ajustarse a la interpretación
del jurado sin tocar una línea de código.
