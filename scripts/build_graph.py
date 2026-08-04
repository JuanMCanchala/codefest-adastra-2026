"""build_graph.py - Construye el grafo de conocimiento (componente bonus, Sec. 7).

Recorre los fragmentos del indice, extrae entidades con GLiNER (NER multilingue
zero-shot, modelo encoder: NO generativo) y construye un grafo dirigido donde las
aristas son co-ocurrencias dentro de un mismo fragmento, con su evidencia
(doc_id y chunk_id) para que toda relacion sea trazable.

Dos controles de tamano, medidos sobre el corpus real:
  - Cada fragmento produce ~13 entidades, y las co-ocurrencias crecen de forma
    cuadratica: sin limite serian ~6,8 millones de aristas. Se acota el numero de
    entidades por fragmento.
  - Al final se podan las aristas de peso 1 (entidades que coincidieron una sola
    vez), que son ruido y dominan el recuento.

Es reanudable: guarda el progreso cada N fragmentos, de modo que una
interrupcion no obliga a repetir la hora de proceso.

    python scripts/build_graph.py --config config.adl.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.encoding.index import VectorStore   # noqa: E402


def normalizar(nombre: str) -> str:
    """Unifica variantes de la misma entidad (espacios, mayusculas, comillas)."""
    return " ".join(nombre.split()).strip(" .,;:\"'()[]").lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.adl.yaml")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--max-entidades", type=int, default=8,
                    help="entidades por fragmento; acota la explosion cuadratica de aristas")
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--min-peso", type=int, default=2,
                    help="poda final: descarta relaciones vistas una sola vez")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pausa", type=float, default=0.0,
                    help="segundos de espera entre lotes; baja la temperatura a "
                         "costa de velocidad (0.5-1.0 en portatiles)")
    ap.add_argument("--hilos", type=int, default=0,
                    help="limita los hilos de CPU de torch (0 = sin limite)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    gcfg = cfg.get("graph", {})
    base = Path(cfg["paths"]["entrega"]) / "base_vectorial"
    enc_name = cfg["encoders"][0]["name"]

    store = VectorStore.load(base / f"encoder_{enc_name}")
    metadata = store.metadata
    if args.limit:
        metadata = metadata[: args.limit]
    print(f"[grafo] {len(metadata):,} fragmentos", flush=True)

    import torch
    import networkx as nx
    from gliner import GLiNER

    tipos = gcfg.get("entity_types",
                     ["persona", "organizacion", "pais", "tecnologia", "evento", "lugar"])
    modelo = GLiNER.from_pretrained(gcfg.get("ner_model", "urchade/gliner_multi-v2.1"))
    if torch.cuda.is_available():
        modelo = modelo.to("cuda")
    modelo.eval()
    print(f"[grafo] GLiNER en {'GPU' if torch.cuda.is_available() else 'CPU'}", flush=True)

    if args.hilos:
        torch.set_num_threads(args.hilos)
        print(f"[grafo] hilos de CPU limitados a {args.hilos}", flush=True)
    if args.pausa:
        print(f"[grafo] pausa de {args.pausa}s entre lotes (modo suave)", flush=True)

    salida = base / "grafo" / "grafo.graphml"
    ckpt = Path(cfg["paths"]["processed"]) / "grafo_checkpoint.json"
    parcial = Path(cfg["paths"]["processed"]) / "grafo_parcial.graphml"

    # El grafo parcial se guarda junto al contador: sin el, reanudar empezaria
    # con un grafo vacio y se perderia todo lo construido antes de la parada.
    grafo = nx.DiGraph()
    inicio = 0
    if ckpt.exists() and parcial.exists():
        estado = json.loads(ckpt.read_text(encoding="utf-8"))
        inicio = estado.get("procesados", 0)
        grafo = nx.read_graphml(str(parcial))
        print(f"[grafo] reanudando desde el fragmento {inicio:,} "
              f"({grafo.number_of_nodes():,} nodos ya construidos)", flush=True)

    t0 = time.time()
    for i in range(inicio, len(metadata), args.batch):
        lote = metadata[i:i + args.batch]
        textos = [m["texto"][: args.max_chars] for m in lote]
        try:
            with torch.no_grad():
                predicciones = modelo.batch_predict_entities(textos, tipos,
                                                             threshold=args.threshold)
        except Exception:
            predicciones = [[] for _ in textos]

        for meta, entidades in zip(lote, predicciones):
            entidades = sorted(entidades, key=lambda e: -e.get("score", 0))[: args.max_entidades]
            nombres: list[str] = []
            for ent in entidades:
                nombre = normalizar(ent["text"])
                if len(nombre) < 3:
                    continue
                if not grafo.has_node(nombre):
                    grafo.add_node(nombre, tipo=ent["label"], chunks="")
                nombres.append(nombre)
            nombres = list(dict.fromkeys(nombres))
            for n in nombres:
                previos = grafo.nodes[n]["chunks"]
                if previos.count(",") < 20:          # evidencia acotada por nodo
                    grafo.nodes[n]["chunks"] = f"{previos},{meta['chunk_id']}".strip(",")
            for a in range(len(nombres)):
                for b in range(a + 1, len(nombres)):
                    x, y = nombres[a], nombres[b]
                    if grafo.has_edge(x, y):
                        grafo[x][y]["peso"] += 1
                    else:
                        grafo.add_edge(x, y, relacion="co-ocurre", peso=1,
                                       doc_id=meta["doc_id"], chunk_id=meta["chunk_id"])

        if args.pausa:
            time.sleep(args.pausa)   # ciclo de trabajo: baja la temperatura media

        procesados = min(i + args.batch, len(metadata))
        if procesados % (args.batch * 50) == 0 or procesados >= len(metadata):
            ritmo = (procesados - inicio) / max(time.time() - t0, 1e-9)
            eta = (len(metadata) - procesados) / max(ritmo, 1e-9) / 60
            print(f"  {procesados:,}/{len(metadata):,}  {ritmo:.1f} frag/s  "
                  f"ETA {eta:.0f} min  |  {grafo.number_of_nodes():,} nodos, "
                  f"{grafo.number_of_edges():,} aristas", flush=True)
            # Se persisten contador Y grafo: asi una parada en cualquier momento
            # conserva el trabajo hecho y deja un grafo utilizable.
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            nx.write_graphml(grafo, str(parcial))
            ckpt.write_text(json.dumps({"procesados": procesados}), encoding="utf-8")

    print(f"\n[grafo] sin podar: {grafo.number_of_nodes():,} nodos, "
          f"{grafo.number_of_edges():,} aristas", flush=True)

    if args.min_peso > 1:
        debiles = [(u, v) for u, v, d in grafo.edges(data=True) if d["peso"] < args.min_peso]
        grafo.remove_edges_from(debiles)
        aislados = [n for n in grafo.nodes if grafo.degree(n) == 0]
        grafo.remove_nodes_from(aislados)
        print(f"[grafo] podado (peso>={args.min_peso}): {grafo.number_of_nodes():,} nodos, "
              f"{grafo.number_of_edges():,} aristas", flush=True)

    salida.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(grafo, str(salida))
    print(f"[grafo] guardado en {salida} "
          f"({salida.stat().st_size / 2**20:.1f} MB) en {(time.time()-t0)/60:.1f} min")
    ckpt.unlink(missing_ok=True)
    parcial.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
