"""ensure_collection is the boot path: app.py calls it on first run and a fresh
deploy has no index at all. The branches are cheap to test with a stub collection,
so they are, rather than being exercised for the first time in production.
"""

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
