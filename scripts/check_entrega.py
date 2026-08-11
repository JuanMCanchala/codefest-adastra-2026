"""check_entrega.py - Verifica el paquete de entrega contra la Seccion 1.4.

Ultima puerta antes de entregar. La especificacion es explicita en que los
objetos con campos faltantes o arrays de tamaño incorrecto "seran penalizados o
descartados", y en que una entrega que no se reproduce se excluye. Este script
comprueba lo que un jurado comprobaria primero:

  1. Estructura de directorios exacta de la Seccion 1.4.
  2. index.faiss se abre con faiss.read_index() sin dependencias adicionales.
  3. metadata.jsonl tiene una linea por vector y en el mismo orden: la linea i
     describe el vector i. Es el invariante del que depende todo el sistema.
  4. Todos los campos obligatorios de la Tabla 1, con fenomeno en {1,2,3}.
  5. chunk_id es el identificador interno de FAISS (FAQ, fila 42).
  6. resultados.jsonl: 50 lineas, 3 documentos, 10 fragmentos, <=250 palabras.
  7. Los doc_id citados en resultados existen en la metadata.
  8. El grafo (bonus) se abre y no esta vacio.

    python scripts/check_entrega.py --entrega entrega
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schema import REQUIRED_CHUNK_FIELDS, validate_resultados   # noqa: E402
from src.utils.io import read_jsonl                                 # noqa: E402


class Informe:
    def __init__(self) -> None:
        self.fallos: list[str] = []
        self.avisos: list[str] = []

    def check(self, condicion: bool, etiqueta: str, detalle: str = "") -> bool:
        marca = "OK  " if condicion else "FALLA"
        print(f"  [{marca}] {etiqueta}" + (f"   {detalle}" if detalle else ""))
        if not condicion:
            self.fallos.append(etiqueta)
        return condicion

    def aviso(self, texto: str) -> None:
        print(f"  [aviso] {texto}")
        self.avisos.append(texto)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrega", default="entrega")
    args = ap.parse_args()
    base = Path(args.entrega)
    inf = Informe()

    print("=== 1. ESTRUCTURA (Seccion 1.4) ===")
    for nombre in ("resultados.jsonl", "generador.py", "informe_tecnico.pdf"):
        f = base / nombre
        inf.check(f.exists(), nombre,
                  f"{f.stat().st_size/1024:.0f} KB" if f.exists() else "AUSENTE")

    bv = base / "base_vectorial"
    encoders = sorted(p for p in bv.glob("encoder_*") if p.is_dir()) if bv.exists() else []
    inf.check(bool(encoders), "base_vectorial/encoder_*",
              ", ".join(p.name for p in encoders) or "ninguno")

    metadata: list[dict] = []
    for enc in encoders:
        print(f"\n=== 2. INDICE {enc.name} ===")
        idx_path, meta_path = enc / "index.faiss", enc / "metadata.jsonl"
        if not inf.check(idx_path.exists(), "index.faiss"):
            continue
        if not inf.check(meta_path.exists(), "metadata.jsonl"):
            continue

        import faiss
        try:
            index = faiss.read_index(str(idx_path))
        except Exception as exc:
            inf.check(False, "faiss.read_index() sin dependencias", str(exc)[:70])
            continue
        inf.check(True, "faiss.read_index() sin dependencias",
                  f"{index.ntotal:,} vectores de {index.d} dimensiones")

        metadata = read_jsonl(meta_path)
        inf.check(len(metadata) == index.ntotal,
                  "una linea de metadata por vector",
                  f"{len(metadata):,} lineas / {index.ntotal:,} vectores")

        faltan = {c for c in REQUIRED_CHUNK_FIELDS for m in metadata[:2000] if c not in m}
        inf.check(not faltan, "campos obligatorios (Tabla 1)",
                  f"faltan {sorted(faltan)}" if faltan else "los 8 presentes")

        malos = [m for m in metadata[:2000] if m.get("fenomeno") not in (1, 2, 3)]
        inf.check(not malos, "fenomeno en {1,2,3}", f"{len(malos)} invalidos" if malos else "")

        # El orden es el invariante del que depende toda la trazabilidad.
        desalineados = [i for i, m in enumerate(metadata) if m.get("chunk_id") != str(i)]
        inf.check(not desalineados, "chunk_id = id interno de FAISS",
                  f"{len(desalineados)} desalineados (primero en {desalineados[0]})"
                  if desalineados else "linea i -> vector i")

        mayus = [m for m in metadata[:2000] if m.get("formato") != str(m.get("formato")).lower()]
        inf.check(not mayus, "formato en minusculas (FAQ fila 21)")

    print("\n=== 3. RESULTADOS ===")
    res_path = base / "resultados.jsonl"
    if res_path.exists():
        res = read_jsonl(res_path)
        errores = validate_resultados(res, expected_lines=50)
        inf.check(not errores, "esquema estricto (50/3/10/<=250 palabras)",
                  f"{len(errores)} errores" if errores else "sin errores")
        for e in errores[:5]:
            print(f"          - {e}")

        if metadata:
            conocidos = {m["doc_id"] for m in metadata}
            citados = {d["doc_id"] for o in res for d in o["documents"]}
            huerfanos = citados - conocidos
            inf.check(not huerfanos, "los doc_id citados existen en el indice",
                      f"{len(huerfanos)} huerfanos: {sorted(huerfanos)[:3]}"
                      if huerfanos else f"{len(citados)} documentos distintos")

    print("\n=== 4. GRAFO (bonus) ===")
    grafo = bv / "grafo" / "grafo.graphml"
    if not grafo.exists():
        inf.aviso("sin grafo.graphml: se entrega sin el componente bonus")
    else:
        import networkx as nx
        g = nx.read_graphml(str(grafo))
        inf.check(g.number_of_nodes() > 0 and g.number_of_edges() > 0,
                  "grafo.graphml legible y no vacio",
                  f"{g.number_of_nodes():,} nodos, {g.number_of_edges():,} aristas")
        # El bono exige integracion, no solo construccion (FAQ, fila 42).
        import yaml
        cfg_path = Path("config.adl.yaml")
        if cfg_path.exists():
            gcfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")).get("graph", {})
            inf.check(bool(gcfg.get("enabled") and gcfg.get("fuse_into_retrieval")),
                      "grafo integrado a la recuperacion (FAQ fila 42)",
                      f"peso {gcfg.get('fusion_weight', 1.0)}")

    print("\n=== VEREDICTO ===")
    if inf.fallos:
        print(f"  {len(inf.fallos)} FALLOS: {', '.join(inf.fallos)}")
        raise SystemExit(1)
    print("  entrega completa y consistente"
          + (f" ({len(inf.avisos)} avisos)" if inf.avisos else ""))


if __name__ == "__main__":
    main()
