"""make_eval_corpus.py - Eval set para el corpus REAL descargado (data/corpus).

Idea: el buscador de arXiv ya es un juicio de relevancia. Si la consulta
'all:"space debris"' devolvio un paper, ese paper es relevante para una consulta
en lenguaje natural sobre desechos espaciales. Es supervision DEBIL pero real, y
permite medir NDCG@10 / F1@3 sobre documentos autenticos sin anotar a mano.

El grado de relevancia se deriva de la posicion en los resultados de arXiv:
    rank 1-2 -> 3 (muy relevante), 3-4 -> 2, 5+ -> 1

Las consultas se redactan en lenguaje natural y se reparten entre ES/EN/PT para
reproducir el escenario cross-lingual del reto (Seccion 10.1).

    python scripts/make_eval_corpus.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.io import read_jsonl, write_jsonl  # noqa: E402

# Consulta de arXiv -> (consulta en lenguaje natural, idioma)
# Se rota el idioma a proposito: la mitad de las consultas NO estan en ingles,
# que es el idioma de casi todos los papers -> mide recuperacion cross-lingual.
QUERY_MAP = {
    'all:"autonomous weapons" AND cat:cs.CY':
        ("riesgos y gobernanza de las armas autonomas letales", "es"),
    'all:"military artificial intelligence"':
        ("military applications of artificial intelligence in defense", "en"),
    'all:"AI governance" AND cat:cs.CY':
        ("governanca e regulacao da inteligencia artificial", "pt"),
    'all:"defense" AND all:"machine learning" AND cat:cs.CY':
        ("aprendizaje automatico aplicado al sector defensa", "es"),
    'all:"space debris"':
        ("desechos espaciales y basura orbital", "es"),
    'all:"low earth orbit" AND all:"collision"':
        ("collision risk between objects in low earth orbit", "en"),
    'all:"satellite constellation" AND all:"sustainability"':
        ("sustentabilidade das constelacoes de satelites", "pt"),
    'all:"orbital debris" AND all:"mitigation"':
        ("mitigacion y remocion de desechos orbitales", "es"),
    'all:"Latin America" AND all:"inequality"':
        ("desigualdad social en America Latina", "es"),
    'all:"Latin America" AND all:"violence"':
        ("violence and organized crime in Latin America", "en"),
    'all:"migration" AND all:"Latin America"':
        ("migracao e deslocamento na America Latina", "pt"),
    'all:"governance" AND all:"Latin America"':
        ("gobernanza institucional y politicas publicas en America Latina", "es"),
}


def grade_from_rank(rank: int) -> int:
    if rank <= 2:
        return 3
    if rank <= 4:
        return 2
    return 1


def main() -> None:
    manifest_path = Path("data/corpus/manifest.jsonl")
    if not manifest_path.exists():
        print(f"[!] no existe {manifest_path}. Ejecuta antes scripts/fetch_corpus.py")
        raise SystemExit(1)

    rows = read_jsonl(manifest_path)
    by_query: dict[str, dict[str, int]] = {}
    fenomeno_of: dict[str, int] = {}
    for r in rows:
        q = r["query"]
        by_query.setdefault(q, {})
        # si un doc aparece en varias consultas, conserva el grado mas alto
        grade = grade_from_rank(r["rank_en_query"])
        prev = by_query[q].get(r["fuente"], 0)
        by_query[q][r["fuente"]] = max(prev, grade)
        fenomeno_of[q] = r["fenomeno"]

    eval_set, queries = [], []
    for i, (arxiv_q, relevantes) in enumerate(sorted(by_query.items()), start=1):
        if arxiv_q not in QUERY_MAP:
            continue
        text, idioma = QUERY_MAP[arxiv_q]
        qid = f"q{i:03d}"
        eval_set.append({
            "query_id": qid,
            "text": text,
            "idioma": idioma,
            "fenomeno": fenomeno_of[arxiv_q],
            "relevantes": relevantes,
        })
        queries.append({"query_id": qid, "text": text})

    n = write_jsonl(Path("eval_interno/eval_corpus.jsonl"), eval_set)
    write_jsonl(Path("eval_interno/queries_corpus.jsonl"), queries)
    print(f"[eval] {n} consultas escritas en eval_interno/eval_corpus.jsonl")
    for item in eval_set:
        print(f"  {item['query_id']} [{item['idioma']}] fen={item['fenomeno']} "
              f"docs_relevantes={len(item['relevantes'])}  {item['text'][:50]}")


if __name__ == "__main__":
    main()
