"""Retrieval regression: a known query must return its own entry as top-1, and
the result must be identical across two independent clients (determinism).

Integration test: needs the local embedding model. It is skipped with a clear
reason if the model cannot be loaded offline, never silently passed.
"""

import chromadb
import pytest
from chromadb.config import Settings

from src import config as cfg

try:
    from src.embed_store import embed

    embed(["warmup"])
except Exception as exc:  # model not cached / offline
    pytest.skip(f"embedding model unavailable: {exc}", allow_module_level=True)


DOCS = {
    "Bioavailability": "Bioavailability\nthe rate and extent to which a drug reaches the circulation",
    "Quality-adjusted Life Years (QALYS)": "Quality-adjusted Life Years (QALYS)\na measure of disease burden combining quality and quantity of life",
    "Wholesaler": "Wholesaler\nentities that perform wholesale distribution of medicines",
    "Cancer": "Cancer\na group of diseases involving abnormal cell growth",
    "Generic Substitution": "Generic Substitution\nsubstituting a medicine with a cheaper equivalent that shares the active ingredient",
}


def _build(path):
    client = chromadb.PersistentClient(
        path=str(path), settings=Settings(anonymized_telemetry=False)
    )
    coll = client.create_collection(
        "regression",
        metadata={
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 200,
            "hnsw:search_ef": 200,
            "hnsw:M": 32,
        },
    )
    terms = list(DOCS)
    coll.add(
        ids=terms,
        embeddings=embed(list(DOCS.values())),
        documents=list(DOCS.values()),
        metadatas=[{"term": t} for t in terms],
    )
    return coll


def _top1(coll, query):
    res = coll.query(query_embeddings=embed([query]), n_results=1)
    return res["metadatas"][0][0]["term"]


def test_embedding_dimension_is_384_and_normalized():
    # all-MiniLM-L6-v2 is 384-dim; the store and every query embed with this model,
    # so a silent model swap that changed the dimension would corrupt retrieval.
    vec = embed(["bioavailability"])[0]
    assert len(vec) == 384
    norm = sum(x * x for x in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-3  # normalize_embeddings=True


def test_known_queries_return_their_entry(tmp_path):
    coll = _build(tmp_path)
    assert _top1(coll, "what is bioavailability") == "Bioavailability"
    assert _top1(coll, "QALY") == "Quality-adjusted Life Years (QALYS)"
    assert _top1(coll, "generic substitution") == "Generic Substitution"


def test_retrieval_is_deterministic_across_clients(tmp_path):
    _build(tmp_path)
    # A second, independent client over the same store must rank identically.
    client2 = chromadb.PersistentClient(
        path=str(tmp_path), settings=Settings(anonymized_telemetry=False)
    )
    coll2 = client2.get_collection("regression")
    assert _top1(coll2, "QALY") == "Quality-adjusted Life Years (QALYS)"


def test_production_index_configures_wide_beam():
    # Guards the recall fix: if the real store reverts to Chroma's default
    # search_ef=10, retrieval can silently miss the true nearest neighbour.
    from src.embed_store import get_collection

    try:
        coll = get_collection()
    except Exception:
        pytest.skip("no built index; run python -m src.embed_store")
    assert (coll.metadata or {}).get("hnsw:search_ef", 10) >= 100
