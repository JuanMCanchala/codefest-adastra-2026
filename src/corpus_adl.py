"""Mapeo oficial del corpus de ADL a partir de Indice_Datos_Codefest.xlsx.

ADL entrega un inventario con un DOC_ID unico por archivo. Usarlo elimina la
mayor incertidumbre del reto: la evaluacion a nivel documento empareja por el
campo `fuente` (Seccion 10.2.1), y aqui tenemos los identificadores y nombres
que el propio organizador asigno, en vez de inventarlos nosotros.

Columnas del inventario:
    Fenómeno | Observatorio | Código Observatorio | DOC_ID |
    Nombre estandarizado | Carpeta | Tipo

Nota: hay 1826 archivos pero solo 1699 nombres unicos, asi que el nombre por si
solo es ambiguo. Se conserva la ruta relativa completa como `fuente` y el
nombre estandarizado y el DOC_ID oficial como campos adicionales.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AdlEntry:
    doc_id: str          # DOC_ID oficial de ADL (ej. F1-AIINDEX-001)
    fuente: str          # ruta relativa dentro del corpus (unica)
    nombre: str          # nombre estandarizado del archivo
    fenomeno: int        # 1, 2 o 3
    observatorio: str
    tipo: str            # PDF | JSON | CSV | Imagen | Excel | Texto | Otro


def load_inventory(xlsx_path: str | Path) -> dict[str, AdlEntry]:
    """Devuelve {ruta_relativa_normalizada: AdlEntry} desde el inventario oficial."""
    import pandas as pd

    df = pd.read_excel(xlsx_path, sheet_name="Inventario de Archivos")
    entries: dict[str, AdlEntry] = {}
    for row in df.itertuples(index=False):
        carpeta = str(getattr(row, "Carpeta", "")).strip().replace("\\", "/").strip("/")
        nombre = str(getattr(row, "_4", "") or getattr(row, "Nombre estandarizado", "")).strip()
        fuente = f"{carpeta}/{nombre}" if carpeta else nombre
        fen_raw = str(getattr(row, "Fenómeno", "")).strip()
        fenomeno = int(fen_raw[1]) if fen_raw[:1].upper() == "F" and fen_raw[1:2].isdigit() else 0
        entries[fuente] = AdlEntry(
            doc_id=str(getattr(row, "DOC_ID", "")).strip(),
            fuente=fuente,
            nombre=nombre,
            fenomeno=fenomeno,
            observatorio=str(getattr(row, "Observatorio", "")).strip(),
            tipo=str(getattr(row, "Tipo", "")).strip(),
        )
    return entries


def fenomeno_from_path(rel_path: str) -> int:
    """Respaldo cuando un archivo no esta en el inventario: el fenomeno se lee
    del primer segmento de la ruta (F1_..., F2_..., F3_...)."""
    head = rel_path.replace("\\", "/").split("/", 1)[0]
    if len(head) >= 2 and head[0].upper() == "F" and head[1].isdigit():
        return int(head[1])
    return 0
