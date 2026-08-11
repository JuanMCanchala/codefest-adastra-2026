"""compare_resultados.py - Compara dos resultados.jsonl generados con distinta configuracion.

Las consultas reales de ADL no traen juicios de relevancia, asi que no se puede
calcular NDCG@10 ni F1@3 sobre ellas: no hay forma de decir cual de dos corridas
es "mejor" en la metrica del reto. Lo que si se puede medir es **cuanto cambia**
una corrida respecto a otra y **en que direccion** se mueven los indicadores de
calidad que si son observables (coherencia tematica por fenomeno, concentracion
de documentos, diversidad de observatorios).

Sirve para decidir si un componente opcional -el grafo, tipicamente- aporta o
solo mueve resultados sin criterio. Un cambio del 0 % significa que el componente
es inerte; un cambio enorme sin mejora tematica es señal de que mete ruido.

    python scripts/compare_resultados.py --a entrega/resultados.jsonl \
        --b entrega/resultados_sin_grafo.jsonl --etiquetas "con grafo" "sin grafo"
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.io import read_jsonl                      # noqa: E402
from src.corpus_adl import load_inventory                # noqa: E402

FENOMENO_POR_QUERY = {**{f"q{i:03d}": 1 for i in range(1, 17)},
                      **{f"q{i:03d}": 2 for i in range(17, 33)},
                      **{f"q{i:03d}": 3 for i in range(33, 51)}}


def indicadores(objs: list[dict], fenomeno_de_doc: dict, obs_de_doc: dict) -> dict:
    """Indicadores observables sin juicios de relevancia."""
    aciertos_tema = total_docs = 0
    concentracion: Counter = Counter()
    consultas_monofuente = 0
    for o in objs:
        esperado = FENOMENO_POR_QUERY.get(o["query_id"])
        observatorios = set()
        for d in o["documents"]:
            total_docs += 1
            concentracion[d["doc_id"]] += 1
            if fenomeno_de_doc.get(d["doc_id"]) == esperado:
                aciertos_tema += 1
            obs = obs_de_doc.get(d["doc_id"])
            if obs:
                observatorios.add(obs)
        if len(observatorios) == 1:
            consultas_monofuente += 1
    mas_repetido = concentracion.most_common(1)[0] if concentracion else ("-", 0)
    return {
        "coherencia_tematica": aciertos_tema / max(total_docs, 1),
        "docs_distintos": len(concentracion),
        "doc_mas_repetido": mas_repetido,
        "consultas_monofuente": consultas_monofuente,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--etiquetas", nargs=2, default=["A", "B"])
    ap.add_argument("--inventario", default="data/adl/Indice_Datos_Codefest.xlsx")
    args = ap.parse_args()

    a, b = read_jsonl(args.a), read_jsonl(args.b)
    ea, eb = args.etiquetas
    if len(a) != len(b):
        raise SystemExit(f"distinto numero de consultas: {len(a)} vs {len(b)}")

    inv = load_inventory(args.inventario)
    fenomeno_de_doc = {e.doc_id: e.fenomeno for e in inv.values()}
    obs_de_doc = {e.doc_id: e.observatorio for e in inv.values()}

    # --------------------------------------------------------------- solapamiento
    docs_iguales = frag_iguales = 0
    solape_docs = solape_frags = 0.0
    cambiadas: list[tuple[str, int, int]] = []
    for ra, rb in zip(a, b):
        da = [d["doc_id"] for d in ra["documents"]]
        db = [d["doc_id"] for d in rb["documents"]]
        fa = {f["text"][:120] for f in ra["fragments"]}
        fb = {f["text"][:120] for f in rb["fragments"]}
        comunes_d, comunes_f = len(set(da) & set(db)), len(fa & fb)
        solape_docs += comunes_d / max(len(da), 1)
        solape_frags += comunes_f / max(len(fa), 1)
        docs_iguales += int(da == db)
        frag_iguales += int(fa == fb)
        if comunes_d < len(da) or comunes_f < len(fa):
            cambiadas.append((ra["query_id"], len(da) - comunes_d, len(fa) - comunes_f))

    n = len(a)
    print(f"Comparando  A={ea}  B={eb}   ({n} consultas)\n")
    print("SOLAPAMIENTO")
    print(f"  documentos identicos en las 3 posiciones : {docs_iguales}/{n}")
    print(f"  fragmentos identicos en los 10           : {frag_iguales}/{n}")
    print(f"  solape medio de documentos               : {solape_docs/n:>6.1%}")
    print(f"  solape medio de fragmentos               : {solape_frags/n:>6.1%}")

    print("\nINDICADORES (sin juicios de relevancia: comparan, no puntuan)")
    ia, ib = indicadores(a, fenomeno_de_doc, obs_de_doc), indicadores(b, fenomeno_de_doc, obs_de_doc)
    print(f"  {'':38} {ea:>12} {eb:>12}")
    print(f"  {'coherencia tematica (doc del fenomeno)':38} "
          f"{ia['coherencia_tematica']:>11.1%} {ib['coherencia_tematica']:>11.1%}")
    print(f"  {'documentos distintos en todo el top-3':38} "
          f"{ia['docs_distintos']:>12} {ib['docs_distintos']:>12}")
    print(f"  {'consultas con un solo observatorio':38} "
          f"{ia['consultas_monofuente']:>12} {ib['consultas_monofuente']:>12}")
    print(f"  {'documento mas repetido':38} "
          f"{ia['doc_mas_repetido'][1]:>12} {ib['doc_mas_repetido'][1]:>12}"
          f"   ({ia['doc_mas_repetido'][0]} / {ib['doc_mas_repetido'][0]})")

    if cambiadas:
        print(f"\nCONSULTAS QUE CAMBIAN: {len(cambiadas)}/{n}"
              f"   (query, docs distintos de 3, fragmentos distintos de 10)")
        for qid, nd, nf in cambiadas[:15]:
            print(f"  {qid}   docs {nd}/3   fragmentos {nf}/10")
        if len(cambiadas) > 15:
            print(f"  ... y {len(cambiadas)-15} mas")
    else:
        print("\nCONSULTAS QUE CAMBIAN: ninguna. El componente es inerte.")


if __name__ == "__main__":
    main()
