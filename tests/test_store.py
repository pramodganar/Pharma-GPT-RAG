"""ensure_collection is the boot path: app.py calls it on first run and a fresh
deploy has no index at all. The branches are cheap to test with a stub collection,
so they are, rather than being exercised for the first time in production.
"""

import pytest

from src import config as cfg
from src import embed_store


class _Collection:
    def __init__(self, n):
        self.n = n

    def count(self):
        return self.n


def _stub_build(monkeypatch):
    built = []
    monkeypatch.setattr(embed_store, "build", lambda: built.append(True) or _Collection(444))
    return built


def test_existing_index_is_reused(monkeypatch):
    existing = _Collection(444)
    monkeypatch.setattr(embed_store, "get_collection", lambda: existing)
    built = _stub_build(monkeypatch)

    assert embed_store.ensure_collection() is existing
    assert not built  # a populated store must never be rebuilt on boot


def test_missing_collection_is_built(monkeypatch):
    # Fresh deploy: get_collection raises because the collection does not exist.
    def _missing():
        raise ValueError("collection pharmacy_glossary does not exist")

    monkeypatch.setattr(embed_store, "get_collection", _missing)
    built = _stub_build(monkeypatch)

    assert embed_store.ensure_collection().count() == 444
    assert built


def test_empty_collection_is_rebuilt(monkeypatch):
    # A collection that exists but holds no vectors is as useless as a missing one.
    monkeypatch.setattr(embed_store, "get_collection", lambda: _Collection(0))
    built = _stub_build(monkeypatch)

    assert embed_store.ensure_collection().count() == 444
    assert built


def _raise(exc):
    def _f(*_args, **_kwargs):
        raise exc

    return _f


def test_unreadable_store_names_the_clean_command(monkeypatch, tmp_path):
    # The fourth state the original three branches missed: a store that exists and is
    # non-empty but was written by another Chroma version, so get_collection AND build
    # both fail on the same unparseable row. The user must get the fix, not a KeyError.
    store = tmp_path / "chroma_db"
    store.mkdir()
    monkeypatch.setattr(cfg, "CHROMA_DIR", store)
    monkeypatch.setattr(embed_store, "get_collection", _raise(KeyError("_type")))
    monkeypatch.setattr(embed_store, "build", _raise(KeyError("_type")))

    with pytest.raises(RuntimeError, match="--clean"):
        embed_store.ensure_collection()


def test_unreadable_store_message_survives_friendly_error(monkeypatch, tmp_path):
    # app.py and the rag_chain CLI both render failures through friendly_error, which
    # would otherwise report a broken index as "the model backend is unavailable".
    from src import llm_factory

    store = tmp_path / "chroma_db"
    store.mkdir()
    monkeypatch.setattr(cfg, "CHROMA_DIR", store)
    monkeypatch.setattr(embed_store, "get_collection", _raise(KeyError("_type")))
    monkeypatch.setattr(embed_store, "build", _raise(KeyError("_type")))

    try:
        embed_store.ensure_collection()
    except RuntimeError as exc:
        message = llm_factory.friendly_error(exc)

    assert "--clean" in message
    assert "backend is unavailable" not in message
    assert "\n" not in message


def test_build_failure_without_a_store_is_not_reported_as_staleness(monkeypatch, tmp_path):
    # No store on disk means an ordinary build failure (missing entries.json, no disk).
    # Blaming a stale store there would send the user to the wrong fix.
    monkeypatch.setattr(cfg, "CHROMA_DIR", tmp_path / "never-built")
    monkeypatch.setattr(embed_store, "get_collection", _raise(ValueError("does not exist")))
    monkeypatch.setattr(embed_store, "build", _raise(FileNotFoundError("entries.json")))

    with pytest.raises(FileNotFoundError):
        embed_store.ensure_collection()


def test_clean_store_removes_the_directory(monkeypatch, tmp_path):
    store = tmp_path / "chroma_db"
    (store / "segment-uuid").mkdir(parents=True)
    (store / "chroma.sqlite3").write_text("x")
    monkeypatch.setattr(cfg, "CHROMA_DIR", store)
    monkeypatch.setattr(embed_store, "_client", None)

    embed_store.clean_store()

    assert not store.exists()


def test_clean_store_is_a_noop_on_a_missing_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(cfg, "CHROMA_DIR", tmp_path / "never-built")
    monkeypatch.setattr(embed_store, "_client", None)
    embed_store.clean_store()  # must not raise


def test_clean_store_refuses_while_a_client_is_open(monkeypatch, tmp_path):
    # Deleting the store under an open client leaves a half-removed directory on
    # Windows, so the guard is the point of the function, not a formality.
    store = tmp_path / "chroma_db"
    store.mkdir()
    monkeypatch.setattr(cfg, "CHROMA_DIR", store)
    monkeypatch.setattr(embed_store, "_client", object())

    with pytest.raises(RuntimeError, match="already open"):
        embed_store.clean_store()
    assert store.exists()
