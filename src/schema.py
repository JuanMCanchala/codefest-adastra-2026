"""Esquemas de datos del reto: metadata de fragmento (Tabla 1) y objeto de
resultado por consulta (Seccion 9.3). Incluye un validador estricto porque
'objetos con campos faltantes o arrays de tamano incorrecto seran penalizados
o descartados' (recuadro 9.3.2).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field


# --------------------------------------------------------------------------- #
#  Metadata obligatoria por fragmento (Tabla 1)
# --------------------------------------------------------------------------- #
@dataclass
class ChunkMeta:
    doc_id: str          # id unico del documento de origen
    chunk_id: str        # id unico del fragmento dentro del documento
    fuente: str          # nombre/URL del archivo original de ADL  <- CLAVE F1@3
    formato: str         # 'pdf' | 'html' | 'md' | 'csv' | ...
    fenomeno: int        # 1, 2 o 3
    posicion: int        # indice ordinal del fragmento en el doc (empieza en 0)
    num_tokens: int      # numero de tokens del fragmento
    texto: str           # texto original del fragmento, sin modificaciones

    # Campos opcionales permitidos (spec: "pueden anadir campos adicionales")
    idioma: str | None = None
    titulo: str | None = None
    fecha: str | None = None

    def to_json(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


REQUIRED_CHUNK_FIELDS = (
    "doc_id", "chunk_id", "fuente", "formato",
    "fenomeno", "posicion", "num_tokens", "texto",
)


def validate_chunk_meta(obj: dict) -> list[str]:
    """Devuelve lista de errores (vacia = valido)."""
    errors = []
    for field_name in REQUIRED_CHUNK_FIELDS:
        if field_name not in obj:
            errors.append(f"falta campo obligatorio '{field_name}'")
    if "fenomeno" in obj and obj["fenomeno"] not in (1, 2, 3):
        errors.append(f"fenomeno debe ser 1, 2 o 3 (es {obj.get('fenomeno')!r})")
    if "posicion" in obj and (not isinstance(obj["posicion"], int) or obj["posicion"] < 0):
        errors.append("posicion debe ser entero >= 0")
    return errors


# --------------------------------------------------------------------------- #
#  Objeto de resultado por consulta (Seccion 9.3, Tabla 2)
# --------------------------------------------------------------------------- #
N_DOCUMENTS = 3
N_FRAGMENTS = 10
MAX_WORDS = 250


@dataclass
class DocResult:
    rank: int            # 1..3
    doc_id: str


@dataclass
class FragmentResult:
    rank: int            # 1..10
    chunk_id: str        # id del chunk original del indice (trazabilidad)
    doc_id: str
    text: str            # <= 250 palabras  <- CLAVE NDCG@10 (se juzga el texto)


@dataclass
class QueryResult:
    query_id: str        # q001..q050
    documents: list[DocResult] = field(default_factory=list)
    fragments: list[FragmentResult] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "query_id": self.query_id,
            "documents": [{"rank": d.rank, "doc_id": d.doc_id} for d in self.documents],
            "fragments": [
                {"rank": f.rank, "chunk_id": f.chunk_id, "doc_id": f.doc_id, "text": f.text}
                for f in self.fragments
            ],
        }


def word_count(text: str) -> int:
    """Conteo de palabras usado para el limite de 250 (whitespace split)."""
    return len(text.split())


def validate_query_result(obj: dict) -> list[str]:
    """Valida un objeto de resultado contra el esquema estricto (9.3.2).

    Reglas duras: 3 documentos, 10 fragmentos, <= 250 palabras por fragmento.
    """
    errors: list[str] = []
    qid = obj.get("query_id", "?")

    if "query_id" not in obj:
        errors.append("falta 'query_id'")

    docs = obj.get("documents")
    if not isinstance(docs, list) or len(docs) != N_DOCUMENTS:
        errors.append(f"[{qid}] 'documents' debe tener exactamente {N_DOCUMENTS} elementos")
    else:
        for d in docs:
            if "rank" not in d or "doc_id" not in d:
                errors.append(f"[{qid}] documento sin 'rank'/'doc_id'")

    frags = obj.get("fragments")
    if not isinstance(frags, list) or len(frags) != N_FRAGMENTS:
        errors.append(f"[{qid}] 'fragments' debe tener exactamente {N_FRAGMENTS} elementos")
    else:
        for f in frags:
            for k in ("rank", "chunk_id", "doc_id", "text"):
                if k not in f:
                    errors.append(f"[{qid}] fragmento sin '{k}'")
            if "text" in f and word_count(f["text"]) > MAX_WORDS:
                errors.append(
                    f"[{qid}] fragmento rank={f.get('rank')} excede {MAX_WORDS} palabras "
                    f"({word_count(f['text'])})"
                )
    return errors


def validate_resultados(objs: list[dict], expected_lines: int = 50) -> list[str]:
    """Valida el archivo resultados.jsonl completo (debe tener 50 lineas,
    q001..q050 en orden). Devuelve lista de errores (vacia = valido)."""
    errors: list[str] = []
    if len(objs) != expected_lines:
        errors.append(f"resultados.jsonl debe tener {expected_lines} lineas (tiene {len(objs)})")
    for i, obj in enumerate(objs):
        expected_qid = f"q{i + 1:03d}"
        if obj.get("query_id") != expected_qid:
            errors.append(f"linea {i + 1}: query_id esperado {expected_qid}, hallado {obj.get('query_id')!r}")
        errors.extend(validate_query_result(obj))
    return errors
