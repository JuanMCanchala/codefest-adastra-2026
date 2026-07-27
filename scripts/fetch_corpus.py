"""fetch_corpus.py - Descarga un corpus de prueba REAL desde fuentes abiertas.

Sustituye al proxy sintetico por documentos publicos reales de los 3 fenomenos
del reto, en ES/EN/PT y en varios formatos. Sirve para medir el pipeline con
señal de verdad antes de que llegue el corpus de ADL.

Fuentes:
  - arXiv (API oficial): papers por tema y fenomeno. PDFs abiertos, IDs reales.
  - Instituciones (ESA, UNOOSA, CEPAL, IPEA): informes en PDF, multilingues.

Uso:
    python scripts/fetch_corpus.py --out data/corpus --per-topic 6

Es respetuoso con las fuentes: 3s entre llamadas a la API de arXiv (su guia lo
pide), user-agent identificable y omision de archivos ya descargados.
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

UA = "CODEFEST-AdAstra-2026-corpus-builder/1.0 (academic hackathon; contact via repo)"
# El DNS de algunas redes falla con export.arxiv.org; probamos varios hosts.
ARXIV_HOSTS = [
    "https://export.arxiv.org/api/query?",
    "https://arxiv.org/api/query?",
    "http://export.arxiv.org/api/query?",
]
ATOM = "{http://www.w3.org/2005/Atom}"

# Los temas viven en scripts/corpus_topics.py (fuente unica compartida con
# make_eval_corpus.py, para que descarga y eval set no se desincronicen).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_topics import arxiv_queries  # noqa: E402

ARXIV_TOPICS = arxiv_queries()

# Informes institucionales (multilingues, formato PDF real de organismos).
INSTITUTIONAL = [
    # (fenomeno, nombre de archivo, url)
    (2, "esa_space_environment_report.pdf",
     "https://www.sdo.esoc.esa.int/environment_report/Space_Environment_Report_latest.pdf"),
    (2, "unoosa_iadc_space_debris_status.pdf",
     "https://www.unoosa.org/res/oosadoc/data/documents/2025/aac_105c_12025crp/"
     "aac_105c_12025crp_10_0_html/AC105_C1_2025_CRP10E.pdf"),
    (3, "cepal_panorama_social_es.pdf",
     "https://repositorio.cepal.org/bitstream/handle/11362/48518/1/S2200947_es.pdf"),
    (1, "ipea_ia_justica_seguranca_pt.pdf",
     "https://repositorio.ipea.gov.br/server/api/core/bitstreams/"
     "aae25d12-8b9b-4e63-b9ca-a549a7ec9492/content"),
]


def _get(url: str, timeout: int = 60, retries: int = 4) -> bytes:
    """GET con reintentos y retroceso exponencial.

    Distingue dos fallos que exigen respuestas opuestas:
      - 429/503 (el servidor pide que bajemos el ritmo): se espera MUCHO mas y
        NUNCA se prueban hosts alternativos, porque eso multiplicaria la tasa de
        peticiones justo cuando nos estan pidiendo lo contrario.
      - errores de DNS/conexion (la red local falla): reintento corto.
    """
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code in (429, 503):
                wait = 20 * (2 ** attempt)          # 20s, 40s, 80s, 160s
                print(f"    [·] {exc.code}: el servidor pide esperar, {wait}s...")
                time.sleep(wait)
            else:
                break                                # 404 y similares: no insistir
        except Exception as exc:                     # DNS / conexion
            last = exc
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
    raise last if last else RuntimeError("fallo desconocido")


def search_arxiv(query: str, max_results: int) -> list[dict]:
    """Consulta la API de arXiv y devuelve [{id, title, pdf_url}]."""
    params = urllib.parse.urlencode({
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
    })
    raw, last_exc = None, None
    for host in ARXIV_HOSTS:
        try:
            raw = _get(host + params)
            break
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code in (429, 503):
                break            # el servidor pide calma: no insistir en otro host
        except Exception as exc:
            last_exc = exc       # DNS: si tiene sentido probar otro host
    if raw is None:
        print(f"    [!] fallo la consulta ({last_exc})")
        return []

    root = ET.fromstring(raw)
    out = []
    for entry in root.findall(f"{ATOM}entry"):
        arxiv_id = entry.findtext(f"{ATOM}id", "").rsplit("/", 1)[-1]
        title = " ".join((entry.findtext(f"{ATOM}title") or "").split())
        pdf_url = next(
            (l.get("href") for l in entry.findall(f"{ATOM}link") if l.get("title") == "pdf"),
            f"https://arxiv.org/pdf/{arxiv_id}",
        )
        if arxiv_id:
            out.append({"id": arxiv_id, "title": title, "pdf_url": pdf_url})
    return out


def download(url: str, dest: Path) -> bool:
    """Descarga si no existe. True si el archivo quedo disponible."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"    = ya existe: {dest.name}")
        return True
    try:
        data = _get(url)
    except Exception as exc:
        print(f"    [!] error descargando {dest.name}: {exc}")
        return False
    if len(data) < 1000:
        print(f"    [!] respuesta sospechosamente corta para {dest.name}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    print(f"    + {dest.name} ({len(data) // 1024} KB)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus")
    ap.add_argument("--per-topic", type=int, default=6, help="papers de arXiv por consulta")
    ap.add_argument("--delay", type=float, default=6.0,
                    help="segundos entre consultas a la API (la guia de arXiv pide >=3)")
    ap.add_argument("--skip-arxiv", action="store_true")
    ap.add_argument("--skip-institutional", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out)
    total = 0
    # Manifiesto: que consulta recupero cada documento. Sirve como etiqueta de
    # relevancia DEBIL para construir un eval set sobre el corpus descargado.
    # Se escribe DESPUES DE CADA TEMA y se reanuda: si el proceso muere (o la API
    # nos limita), no se pierde lo ya conseguido ni se repiten consultas.
    manifest_path = out_root / "manifest.jsonl"
    manifest: list[dict] = []
    if manifest_path.exists():
        import json as _json
        with manifest_path.open(encoding="utf-8") as fh:
            manifest = [_json.loads(l) for l in fh if l.strip()]
    done_queries = {row["query"] for row in manifest}
    if done_queries:
        print(f"[fetch_corpus] reanudando: {len(done_queries)} temas ya registrados")

    def _persist_manifest() -> None:
        import json as _json
        out_root.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8", newline="\n") as fh:
            for row in manifest:
                fh.write(_json.dumps(row, ensure_ascii=False) + "\n")

    if not args.skip_arxiv:
        for fenomeno, queries in ARXIV_TOPICS.items():
            for query in queries:
                if query in done_queries:
                    print(f"[arXiv] fenomeno {fenomeno}: (ya hecho) {query}")
                    continue
                print(f"[arXiv] fenomeno {fenomeno}: {query}")
                results = search_arxiv(query, args.per_topic)
                print(f"    {len(results)} resultados")
                for rank, r in enumerate(results):
                    safe_id = r["id"].replace("/", "_")
                    rel_path = f"fenomeno{fenomeno}/arxiv_{safe_id}.pdf"
                    dest = out_root / rel_path
                    if download(r["pdf_url"], dest):
                        total += 1
                        manifest.append({
                            "fuente": rel_path,
                            "fenomeno": fenomeno,
                            "query": query,
                            "rank_en_query": rank + 1,
                            "titulo": r["title"],
                        })
                if results:
                    _persist_manifest()   # progreso a salvo tras cada tema
                time.sleep(args.delay)    # cortesia con la API de arXiv

    if manifest:
        _persist_manifest()
        print(f"[fetch_corpus] manifiesto: {manifest_path} ({len(manifest)} entradas)")

    if not args.skip_institutional:
        print("\n[institucional]")
        for fenomeno, name, url in INSTITUTIONAL:
            dest = out_root / f"fenomeno{fenomeno}" / name
            if download(url, dest):
                total += 1
            time.sleep(1)

    print(f"\n[fetch_corpus] documentos disponibles: {total}")
    for d in sorted(out_root.glob("fenomeno*")):
        files = list(d.glob("*"))
        size = sum(f.stat().st_size for f in files) // (1024 * 1024)
        print(f"  {d.name}: {len(files)} archivos ({size} MB)")


if __name__ == "__main__":
    main()
