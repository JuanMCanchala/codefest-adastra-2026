"""Lectura/escritura de JSON Lines de forma determinista y UTF-8.

En resultados.jsonl 'cada linea es un objeto JSON valido e independiente'
(spec 9.3). Escribimos con ensure_ascii=False y sin espacios extra para que
generador.py reproduzca el archivo byte a byte.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, objs: Iterable[dict]) -> int:
    """Escribe objetos como JSON Lines. Devuelve el numero de lineas escritas."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for obj in objs:
            fh.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
            fh.write("\n")
            count += 1
    return count
