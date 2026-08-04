"""check_results.py - Verificacion de calidad de resultados.jsonl.

Sin los juicios de relevancia de ADL no se puede calcular NDCG@10 ni F1@3 sobre
las consultas reales. Lo que si se puede es detectar sintomas de que algo va mal.
Estas comprobaciones encontraron el defecto mas grave del proyecto (el PDF con
las consultas indexado como documento del corpus), asi que se sistematizan:

  1. Esquema estricto: 50 lineas, 3 documentos, 10 fragmentos, <=250 palabras.
  2. Coherencia tematica: una consulta del fenomeno N deberia devolver documentos
     del fenomeno N.
  3. Documentos omnipresentes: un documento que aparece en muchas consultas
     distintas suele ser ruido (indices, catalogos, el propio set de preguntas).
  4. Diversidad de fuentes: si los 3 documentos salen del mismo observatorio, se
     pierde cobertura.
  5. Fragmentos degenerados: vacios, duplicados o demasiado cortos.

    python scripts/check_results.py --resultados entrega/resultados.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schema import validate_resultados, word_count   # noqa: E402
from src.utils.io import read_jsonl                      # noqa: E402

# Rango de consultas por fenomeno segun el documento de ADL
FENOMENO_POR_QUERY = {**{f"q{i:03d}": 1 for i in range(1, 17)},
                      **{f"q{i:03d}": 2 for i in range(17, 33)},
                      **{f"q{i:03d}": 3 for i in range(33, 51)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resultados", default="entrega/resultados.jsonl")
    ap.add_argument("--docs", default="data/processed/docs.jsonl")
    args = ap.parse_args()

    res = read_jsonl(args.resultados)
    meta = {}
    for line in Path(args.docs).open(encoding="utf-8"):
        row = json.loads(line)
        meta[row["doc_id"]] = row

    print(f"=== 1. ESQUEMA ===")
    errors = validate_resultados(res, expected_lines=50)
    print(f"  lineas: {len(res)} | errores: {len(errors)}")
    for e in errors[:5]:
        print(f"    - {e}")

    print(f"\n=== 2. COHERENCIA TEMATICA ===")
    off_topic = []
    for obj in res:
        esperado = FENOMENO_POR_QUERY.get(obj["query_id"])
        fens = [meta.get(d["doc_id"], {}).get("fenomeno") for d in obj["documents"]]
        aciertos = sum(1 for f in fens if f == esperado)
        if aciertos == 0:
            off_topic.append((obj["query_id"], esperado, fens))
    print(f"  consultas sin ningun documento de su fenomeno: {len(off_topic)}/50")
    for qid, esp, got in off_topic[:6]:
        print(f"    {qid}: esperado F{esp}, recuperado {got}")

    print(f"\n=== 3. DOCUMENTOS OMNIPRESENTES ===")
    doc_freq = Counter(d["doc_id"] for obj in res for d in obj["documents"])
    for doc_id, n in doc_freq.most_common(5):
        fuente = meta.get(doc_id, {}).get("fuente", "?")
        flag = "  <-- REVISAR" if n >= 10 else ""
        print(f"  {n:3d}/50  {doc_id:16s} {fuente[-58:]}{flag}")

    print(f"\n=== 4. DIVERSIDAD DE FUENTES ===")
    def observatorio(doc_id: str) -> str:
        f = meta.get(doc_id, {}).get("fuente", "")
        parts = f.split("/")
        return parts[1] if len(parts) > 1 else "?"
    mono = [obj["query_id"] for obj in res
            if len({observatorio(d["doc_id"]) for d in obj["documents"]}) == 1]
    print(f"  consultas con los 3 documentos del mismo observatorio: {len(mono)}/50")
    if mono:
        print(f"    {', '.join(mono[:12])}")

    print(f"\n=== 5. FRAGMENTOS ===")
    ws = [word_count(f["text"]) for obj in res for f in obj["fragments"]]
    vacios = sum(1 for w in ws if w < 10)
    textos = [f["text"] for obj in res for f in obj["fragments"]]
    dups = len(textos) - len(set(textos))
    print(f"  palabras: min {min(ws)} media {sum(ws)//len(ws)} max {max(ws)} (limite 250)")
    print(f"  fragmentos casi vacios (<10 palabras): {vacios}")
    print(f"  fragmentos duplicados exactos: {dups}")

    print(f"\n=== VEREDICTO ===")
    problemas = len(errors) + len(off_topic) + sum(1 for _, n in doc_freq.most_common(3) if n >= 10)
    print("  OK" if problemas == 0 else f"  {problemas} señales que revisar")


if __name__ == "__main__":
    main()
