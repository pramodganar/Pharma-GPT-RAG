"""Fusion is pure and tested directly; the retrieve path is an integration test that
skips without the embedding model, matching test_retrieval.py's contract.
"""

import pytest

from src import config as cfg
from src import rag_chain
from src.hybrid import rrf


def test_rrf_ranks_agreement_above_either_retriever_alone():
    # "b" is second in both rankings and first in neither; agreement should carry it
    # to the top. This is the whole reason to fuse rather than concatenate.
    fused = rrf([["a", "b", "c"], ["d", "b", "e"]])
    assert fused[0] == "b"


def test_rrf_keeps_items_only_one_retriever_found():
    fused = rrf([["a"], ["z"]])
    assert set(fused) == {"a", "z"}


def test_rrf_breaks_ties_deterministically():
    # Symmetric input: both lists rank their own item first, so scores tie. Order
    # must still be stable, or retrieval stops being reproducible run to run.
    assert rrf([["a"], ["b"]]) == rrf([["a"], ["b"]]) == ["a", "b"]


def test_rrf_of_one_ranking_preserves_it():
    assert rrf([["a", "b", "c"]]) == ["a", "b", "c"]


def test_rrf_k_damps_the_top_rank():
    # With a small rrf_k a single first place outweighs agreement further down; with
    # the default it does not. Guards the constant against being "simplified" away.
    assert rrf([["a", "b", "c"], ["d", "b", "e"]], rrf_k=0)[0] == "a"
    assert rrf([["a", "b", "c"], ["d", "b", "e"]], rrf_k=60)[0] == "b"


def test_retriever_flag_defaults_to_dense():
    # The published hit@k numbers are dense; a silent default flip would invalidate
    # every number in the README.
    assert cfg.RETRIEVER == "dense"


def test_rag_chain_dispatches_to_hybrid_when_configured(monkeypatch):
    from src import hybrid

    monkeypatch.setattr(cfg, "RETRIEVER", "hybrid")
    monkeypatch.setattr(hybrid, "retrieve", lambda q, k: [{"term": "from-hybrid"}])
    assert rag_chain.retrieve("q")[0]["term"] == "from-hybrid"

    monkeypatch.setattr(cfg, "RETRIEVER", "dense")
    monkeypatch.setattr(rag_chain, "ensure_collection", _stub_collection)
    assert rag_chain.retrieve("q")[0]["term"] == "from-dense"


def _stub_collection():
    class _Coll:
        def query(self, query_embeddings, n_results):
            return {
                "ids": [["0"]],
                "documents": [["Dense\ntext"]],
                "metadatas": [[{"term": "from-dense", "page_start": 1, "source": ""}]],
                "distances": [[0.1]],
            }

    return _Coll()


pytest.importorskip("rank_bm25", reason="hybrid retrieval needs rank_bm25")

try:
    from src.embed_store import embed

    embed(["warmup"])
except Exception as exc:  # model not cached / offline
    pytest.skip(f"embedding model unavailable: {exc}", allow_module_level=True)


def test_hybrid_retrieve_returns_rag_chain_shaped_docs():
    from src.hybrid import retrieve

    docs = retrieve("What is bioavailability?", k=5)

    assert len(docs) == 5
    assert docs[0]["term"] == "Bioavailability"
    for d in docs:
        assert set(d) == {"text", "term", "page", "source", "distance"}
        # Every doc carries a real cosine distance, including BM25-only hits the
        # dense query never returned -- the UI renders relevance from this.
        assert 0.0 <= d["distance"] <= 2.0


def test_hybrid_finds_an_acronym_dense_alone_ranks_lower():
    from src.hybrid import retrieve

    # The category the fusion was built for: lexical match rescues bare acronyms.
    terms = [d["term"] for d in retrieve("DDD", k=5)]
    assert any("Defined Daily Dose" in t for t in terms)
