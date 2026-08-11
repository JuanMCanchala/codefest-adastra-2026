# CODEFEST AD ASTRA 2026 — Etapa 1: Base de Conocimiento Vectorial

Sistema de **recuperación densa multilingüe** (RAG sin generación) para el reto de la Fuerza
Aeroespacial Colombiana + Universidad de los Andes. Indexa el corpus multiformato de ADL (ES/EN/PT)
sobre tres fenómenos y responde 50 consultas devolviendo **top-3 documentos + top-10 fragmentos**.
Evaluación: `NDCG@10` (fragmentos) + `F1@3` (documentos), combinados con Conteo de Borda.

**Estado: entrega completa.** 1 821 de 1 826 documentos indexados, 90 613 fragmentos,
`resultados.jsonl` con 50 líneas y cero errores de esquema, grafo de conocimiento integrado a la
recuperación. 65 pruebas automatizadas.

## Entregables (Sección 1.4)

```
entrega/
  resultados.jsonl          50 líneas q001–q050, esquema estricto validado
  generador.py              reproduce resultados.jsonl desde el índice
  informe_tecnico.pdf       informe técnico (7 págs)
  base_vectorial/
    encoder_bge-m3/
      index.faiss           IndexFlatIP, 90 613 vectores de 1024 dimensiones
      metadata.jsonl        una línea por fragmento, alineada con el id interno de FAISS
      sparse_index.npz      señal léxica de BGE-M3 (extra, no exigido)
      sparse_meta.json
    grafo/
      grafo.graphml         BONUS: 26 961 entidades, 97 182 relaciones
```

> **La base vectorial se descarga aparte.** `index.faiss` (354 MB) y `metadata.jsonl` (190 MB)
> superan el límite de 100 MB por archivo de GitHub, así que viaja como *release*:
>
> **https://github.com/JuanMCanchala/codefest-adastra-2026/releases/tag/base-vectorial-v1**
>
> ```bash
> unzip base_vectorial.zip -d entrega/     # 507 MB, md5 adf39b0dd4aee9835d19aa5371d86ed3
> python scripts/check_entrega.py --entrega entrega
> ```
>
> Alternativamente puede reconstruirse desde el corpus (ver *Reproducción desde cero*).

## Documentación

| Documento | Para qué |
|---|---|
| **[docs/SISTEMA.md](docs/SISTEMA.md)** | Referencia completa: arquitectura, cada módulo, cada decisión, runbook |
| **[docs/BITACORA.md](docs/BITACORA.md)** | Trazabilidad: qué cambió, por qué, resultados medidos, bugs corregidos |
| **[entrega/informe_tecnico.pdf](entrega/informe_tecnico.pdf)** | Informe técnico oficial |
| [docs/NOTA_UNLIMITED_OCR.md](docs/NOTA_UNLIMITED_OCR.md) | Componente parqueado y por qué |

## Las claves que deciden el puntaje

1. **Los fragmentos se juzgan por su texto; los documentos, por `doc_id`.** La Sección 10.2.1 decía
   que el emparejamiento a nivel documento era por `fuente`, pero ADL confirmó que **fue una errata**
   y que se usa el `doc_id` de `Indice_Datos_Codefest.xlsx`. Usamos ese identificador desde el
   principio. Consecuencia: la calidad de **extracción y limpieza** pone el techo del NDCG@10.
2. **Cross-lingual**: una consulta en español debe recuperar documentos en inglés y portugués →
   encoder multilingüe fuerte (**BGE-M3**).
3. **Decoders prohibidos, cross-encoders permitidos.** Consultamos formalmente al jurado; la
   respuesta fue *«sí está permitido re-ranking con cross-encoders; la restricción aplica es para
   arquitecturas decoders»*. El reranking está activo.
4. **El grafo solo puntúa si está integrado a la recuperación** (*«el solo construirlo no es
   válido»*). Entra como un ranking más en la fusión RRF, con peso calibrado (ver abajo).
5. **Completitud lingüística + 250 palabras** → chunking de **dos niveles**.

## Arquitectura

```
extracción → limpieza → chunking (2 niveles) → BGE-M3 (denso + léxico)
   → FAISS IndexFlatIP → fusión RRF ponderada → rerank cross-encoder → agregación
   → resultados.jsonl
                            ↑
                  grafo de conocimiento (bonus, peso 0,3)
```

**Fusión ponderada.** RRF pondera solo por posición, así que el primer resultado de cada ranking
vale lo mismo venga de donde venga. Medimos que con peso pleno el grafo —que ordena por
co-ocurrencia de entidades, no por relevancia semántica— **expulsaba del top-3 los documentos que
respondían la consulta**. Las tres fórmulas de fusión admiten ahora peso por ranking; con todos los
pesos a 1 el resultado es idéntico a la ecuación 7 del enunciado, verificado por test. Detalle
completo en la bitácora, hallazgo R8.

## Entorno

Validado en **Python 3.13.14**, GPU **RTX 4060 Laptop 8 GB**.

```bash
python -m venv .venv && .venv\Scripts\activate
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

> Instalar torch **antes** que el resto y comprobarlo después: varias dependencias
> (`docling`, `gliner`) arrastran la build de CPU y la sustituyen en silencio. `pip install
> torch==2.13.0` responde *"already satisfied"* aunque la instalada sea `+cpu`. Verificar con
> `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` → debe decir
> `2.13.0+cu126 True`.

Versiones verificadas: `faiss-cpu 1.14.3`, `torch 2.13.0+cu126`, `sentence-transformers 5.6.1`,
`transformers 5.13.1`, `FlagEmbedding` (BGE-M3), `pymupdf`, `easyocr`, `gliner`, `networkx`.

## Reproducir la entrega

Con la base vectorial ya colocada en `entrega/base_vectorial/` (ver enlace arriba):

```bash
python entrega/generador.py --config config.adl.yaml
```

Regenera `entrega/resultados.jsonl` **byte a byte**: dos corridas completas producen el mismo
archivo, con MD5 `39d4eee2eb7bb13d12bd51c39844a5a0`. Añadir `--pausa 2` en portátiles: los tres modelos
(encoder denso, cross-encoder y NER) comparten la GPU sin descanso y la corrida alcanza 83 °C.

## Reproducción desde cero

Requiere el corpus de ADL, que **no se redistribuye** en este repositorio. Colocarlo en
`data/adl/corpus/` respetando la estructura original; `data/adl/Indice_Datos_Codefest.xlsx` y
`data/adl/queries.jsonl` sí están incluidos.

```bash
python scripts/extract_corpus.py --config config.adl.yaml --workers 8   # ~2 min (cacheado, reanudable)
python scripts/ocr_scanned.py   --config config.adl.yaml                # ~48 min, 51 PDFs escaneados
python scripts/build_index.py   --config config.adl.yaml                # índice denso + léxico (GPU)
python scripts/build_graph.py   --config config.adl.yaml --batch 8 --pausa 1.2 --hilos 4   # ~2,4 h
python entrega/generador.py     --config config.adl.yaml --pausa 2
python scripts/check_results.py --resultados entrega/resultados.jsonl
```

`build_graph.py` es reanudable: guarda grafo y contador cada 400 fragmentos, de modo que un apagón
cuesta como mucho esos 400. Sus banderas `--batch/--pausa/--hilos` acotan el pico térmico.

Para añadir documentos a un índice ya construido sin rehacerlo:

```bash
python scripts/append_docs.py --config config.adl.yaml --dry-run   # informa qué falta
python scripts/append_docs.py --config config.adl.yaml
```

## Estructura del repositorio

```
config.adl.yaml           configuración de la entrega (corpus oficial)
entrega/                  entregables de la Sección 1.4
src/
  extraction/             pdf (pymupdf/docling), json, csv/xlsx, imágenes (OCR), pbf
  cleaning/               normalización: quita encabezados repetidos, des-hifena
  chunking/               segmentación de oraciones + chunker de dos niveles
  encoding/               BGE-M3 (denso + léxico) + índice FAISS + índice disperso
  retrieval/              fusión ponderada, reranking, agregación, pipeline
  graph/                  NER (GLiNER) → grafo.graphml + recuperación por grafo
  eval/                   métricas exactas (NDCG@10, F1@3, Borda) + arnés
  schema.py               metadata de la Tabla 1 + validador estricto de resultados
scripts/                  indexación, OCR, grafo, verificación y comparación
tests/                    65 pruebas
```

## Verificación de calidad

```bash
python -m pytest -q                                                  # 65 pruebas
python scripts/check_results.py --resultados entrega/resultados.jsonl  # 5 comprobaciones
python scripts/compare_resultados.py --a A.jsonl --b B.jsonl           # A/B entre configuraciones
```

`check_results.py` verifica esquema, coherencia temática, documentos omnipresentes, diversidad de
fuentes y fragmentos degenerados. Fue lo que detectó el defecto más grave del proyecto: el PDF con
las 50 consultas indexado como documento del corpus, que aparecía en el top-3 de 20 consultas.

## Limitaciones declaradas

- Las 50 consultas **no traen juicios de relevancia**: no se puede calcular NDCG@10 ni F1@3 sobre
  ellas. Ninguna cifra de este repositorio afirma que una configuración sea *mejor* que otra en la
  métrica del reto.
- La **coherencia temática es un indicador sesgado**: asume que un documento relevante para una
  consulta de F1 vive en la carpeta F1, y SIPRI y CEEEP están archivados bajo F3 publicando sobre
  IA militar. Marca como desalineadas consultas que devuelven documentos correctos.
- **Cinco documentos** del inventario no están indexados ni podrán estarlo: cuatro fotografías sin
  texto y un archivo de dos bytes.
- La contribución positiva demostrable del **grafo** es pequeña: con el peso calibrado altera 2 de
  las 50 consultas.
