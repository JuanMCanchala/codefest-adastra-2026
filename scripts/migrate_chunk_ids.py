"""migrate_chunk_ids.py - Pasa chunk_id al identificador interno de FAISS.

Las FAQ del reto (fila 42) indican: "deberian usar como chunk_id el mismo
obtenido del indice FAISS". El indice se construyo con identificadores
descriptivos ('F2-SWF-012-chunk-0007'), utiles para depurar pero distintos de
los que espera el organizador.

La migracion no toca los vectores. FAISS asigna sus identificadores internos por
orden de insercion, y la base ya cumple el invariante de que la linea i de
metadata.jsonl corresponde al vector i; basta con reescribir el campo. El
identificador descriptivo se conserva en `chunk_uid`, que la especificacion
permite como campo adicional, para no perder la trazabilidad a ojo.

El indice disperso comparte ese mismo orden de insercion, asi que su lista de
chunk_ids se reescribe igual. Si no se reescribiera, la fusion RRF mezclaria dos
espacios de identificadores distintos y la senal lexica dejaria de sumar.

    python scripts/migrate_chunk_ids.py --config config.adl.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.io import read_jsonl, write_jsonl   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.adl.yaml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base = Path(cfg["paths"]["entrega"]) / "base_vectorial"

    for enc_cfg in cfg["encoders"]:
        name = enc_cfg["name"]
        carpeta = base / f"encoder_{name}"
        meta_path = carpeta / "metadata.jsonl"
        if not meta_path.exists():
            print(f"[migra] {name}: sin metadata.jsonl, se omite")
            continue

        import faiss
        index = faiss.read_index(str(carpeta / "index.faiss"))
        # read_jsonl lee linea a linea: str.splitlines() tambien parte en U+2028 y
        # en el avance de pagina, que JSON no escapa y que aparecen dentro del
        # texto de algunos fragmentos.
        registros = read_jsonl(meta_path)
        if index.ntotal != len(registros):
            raise SystemExit(f"[migra] {name}: {index.ntotal} vectores vs "
                             f"{len(registros)} lineas de metadata; abortado")

        ya_migrado = sum(1 for r in registros if r.get("chunk_id", "").isdigit())
        print(f"[migra] {name}: {len(registros):,} fragmentos "
              f"({ya_migrado:,} ya con id numerico)")

        antiguo_a_nuevo = {}
        for i, reg in enumerate(registros):
            anterior = reg["chunk_id"]
            if not anterior.isdigit():
                reg["chunk_uid"] = anterior
            antiguo_a_nuevo[anterior] = str(i)
            reg["chunk_id"] = str(i)

        if args.dry_run:
            print(f"[migra] (dry-run) ejemplo: {registros[0]['chunk_uid']} -> "
                  f"{registros[0]['chunk_id']}")
            continue

        write_jsonl(meta_path, registros)
        print(f"[migra] {name}: metadata.jsonl reescrito")

        # Indice disperso: misma lista, mismo orden.
        meta_sparse = carpeta / "sparse_meta.json"
        if meta_sparse.exists():
            payload = json.loads(meta_sparse.read_text(encoding="utf-8"))
            previos = payload["chunk_ids"]
            if len(previos) != len(registros):
                raise SystemExit(f"[migra] {name}: el indice disperso tiene "
                                 f"{len(previos)} chunks y el denso {len(registros)}")
            payload["chunk_ids"] = [antiguo_a_nuevo.get(c, c) for c in previos]
            with meta_sparse.open("w", encoding="utf-8", newline="\n") as fh:
                json.dump(payload, fh, ensure_ascii=False)
            # El disperso se construye insertando en el mismo orden que el denso;
            # si eso dejara de cumplirse, la fusion emparejaria fragmentos que no
            # se corresponden, asi que se comprueba en vez de suponerlo.
            desalineados = [i for i, c in enumerate(payload["chunk_ids"]) if c != str(i)]
            if desalineados:
                raise SystemExit(f"[migra] {name}: {len(desalineados)} chunks del "
                                 f"indice disperso no coinciden con el denso "
                                 f"(primero en la posicion {desalineados[0]})")
            print(f"[migra] {name}: sparse_meta.json reescrito y verificado")


if __name__ == "__main__":
    main()
