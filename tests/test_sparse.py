"""Tests del indice disperso (senal lexical de BGE-M3), sin cargar el modelo."""
from src.encoding.sparse import SparseIndex


def _index():
    idx = SparseIndex()
    # pesos lexicos simulados: {token: peso}
    idx.add(
        [
            {"kessler": 0.9, "orbita": 0.5},          # c1
            {"orbita": 0.6, "satelite": 0.4},         # c2
            {"defensa": 0.8},                          # c3
        ],
        ["c1", "c2", "c3"],
    )
    return idx


def test_search_ranks_by_lexical_overlap():
    idx = _index()
    res = dict(idx.search({"kessler": 1.0}))
    assert "c1" in res and "c3" not in res       # solo c1 contiene 'kessler'


def test_search_scores_are_dot_products():
    idx = _index()
    res = dict(idx.search({"orbita": 2.0}))
    # c1: 2.0*0.5 = 1.0 ; c2: 2.0*0.6 = 1.2 -> c2 gana
    assert abs(res["c2"] - 1.2) < 1e-9
    assert abs(res["c1"] - 1.0) < 1e-9
    assert res["c2"] > res["c1"]


def test_search_no_match_is_empty():
    idx = _index()
    assert idx.search({"token_inexistente": 1.0}) == []


def test_zero_weights_ignored():
    idx = SparseIndex()
    idx.add([{"a": 0.0, "b": 0.5}], ["c1"])
    assert idx.search({"a": 1.0}) == []
    assert dict(idx.search({"b": 1.0}))["c1"] == 0.5


def test_roundtrip_save_load(tmp_path):
    idx = _index()
    idx.save(tmp_path)
    loaded = SparseIndex.load(tmp_path)
    assert loaded is not None
    assert loaded.n_chunks == 3
    assert dict(loaded.search({"kessler": 1.0})) == dict(idx.search({"kessler": 1.0}))


def test_load_returns_none_when_absent(tmp_path):
    assert SparseIndex.load(tmp_path) is None
