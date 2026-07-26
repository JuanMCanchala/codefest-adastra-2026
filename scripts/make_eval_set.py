"""make_eval_set.py - Construye el eval interno (herramienta de DESARROLLO).

Sin ground truth publico, generamos consultas con relevancia conocida para medir
NDCG@10 + F1@3 en local y optimizar hiperparametros.

Estrategia recomendada para el corpus real:
  1. Muestrear chunks representativos por fenomeno e idioma.
  2. Generar 2-3 consultas naturales por chunk con un LLM OFFLINE (permitido:
     es tooling de dev, NO forma parte de generador.py ni del pipeline evaluado).
  3. La `fuente` del chunk origen queda como documento relevante (grado 3);
     opcionalmente marcar fuentes tematicamente cercanas con grado 1-2.
  4. Balancear ES/EN/PT y los 3 fenomenos, como el set real (spec 10.1).

Aqui dejamos un set semilla HAND-AUTHORED sobre el corpus proxy, para que el
arnes (src/eval/harness.py) sea ejecutable en cuanto haya encoders.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.io import write_jsonl  # noqa: E402


SEED_EVAL = [
    {"query_id": "q001", "text": "dilemas eticos de las armas autonomas en defensa",
     "idioma": "es", "fenomeno": 1,
     "relevantes": {"fenomeno1/ia_defensa_es.md": 3, "fenomeno1/autonomous_weapons_en.md": 3,
                    "fenomeno1/gobernanza_talento_ia_es.md": 1}},
    {"query_id": "q002", "text": "talent gap in machine learning for the defense sector",
     "idioma": "en", "fenomeno": 1,
     "relevantes": {"fenomeno1/ai_index_sample.json": 3, "fenomeno1/gobernanza_talento_ia_es.md": 3,
                    "fenomeno1/ia_defensa_es.md": 1}},
    {"query_id": "q003", "text": "sindrome de Kessler y colisiones en cascada en orbita baja",
     "idioma": "es", "fenomeno": 2,
     "relevantes": {"fenomeno2/kessler_syndrome_es.md": 3, "fenomeno2/space_debris_en.md": 2}},
    {"query_id": "q004", "text": "active debris removal and end-of-life disposal rules",
     "idioma": "en", "fenomeno": 2,
     "relevantes": {"fenomeno2/space_debris_en.md": 3, "fenomeno2/remocao_detritos_pt.md": 3,
                    "fenomeno2/kessler_syndrome_es.md": 1}},
    {"query_id": "q005", "text": "remocao de detritos e sustentabilidade orbital",
     "idioma": "pt", "fenomeno": 2,
     "relevantes": {"fenomeno2/remocao_detritos_pt.md": 3, "fenomeno2/kessler_syndrome_es.md": 2}},
    {"query_id": "q006", "text": "migracion y seguridad humana en America Latina",
     "idioma": "es", "fenomeno": 3,
     "relevantes": {"fenomeno3/migracion_seguridad_es.md": 3, "fenomeno3/territorial_governance_en.md": 2}},
    {"query_id": "q007", "text": "territorial governance and social perception of institutions",
     "idioma": "en", "fenomeno": 3,
     "relevantes": {"fenomeno3/territorial_governance_en.md": 3, "fenomeno3/dinamicas_territoriais_pt.md": 2,
                    "fenomeno3/migracion_seguridad_es.md": 1}},
    {"query_id": "q008", "text": "dinamicas territoriais conflito e migracao na regiao",
     "idioma": "pt", "fenomeno": 3,
     "relevantes": {"fenomeno3/dinamicas_territoriais_pt.md": 3, "fenomeno3/migracion_seguridad_es.md": 2}},
    {"query_id": "q009", "text": "autonomous weapons international humanitarian law geneva",
     "idioma": "en", "fenomeno": 1,
     "relevantes": {"fenomeno1/autonomous_weapons_en.md": 3, "fenomeno1/ia_defensa_es.md": 2}},
    {"query_id": "q010", "text": "desechos espaciales satelites obsoletos y etapas de cohetes",
     "idioma": "es", "fenomeno": 2,
     "relevantes": {"fenomeno2/space_debris_en.md": 3, "fenomeno2/kessler_syndrome_es.md": 3,
                    "fenomeno2/remocao_detritos_pt.md": 2}},
]


def main() -> None:
    out = Path("eval_interno/eval.jsonl")
    n = write_jsonl(out, SEED_EVAL)
    print(f"[eval] {n} consultas con relevancia escritas en {out}")

    # archivo de consultas para generador.py: solo {query_id, text}
    queries = [{"query_id": q["query_id"], "text": q["text"]} for q in SEED_EVAL]
    qn = write_jsonl(Path("eval_interno/queries_smoke.jsonl"), queries)
    print(f"[eval] {qn} consultas (id+texto) escritas en eval_interno/queries_smoke.jsonl")
    print("[eval] Reemplazar/ampliar con el corpus real (ver docstring).")


if __name__ == "__main__":
    main()
