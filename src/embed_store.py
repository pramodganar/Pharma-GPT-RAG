"""Embed chunks with all-MiniLM-L6-v2 and persist them to a local ChromaDB store.

Running this rebuilds the collection from scratch. Build with: python -m src.embed_store
"""

import logging

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from . import config as cfg
from .chunking import load_chunks

# chromadb 0.5.x emits a harmless posthog telemetry error on every call; mute it.
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

_model = None
_client = None


def get_embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer(cfg.EMBED_MODEL)
    return _model


def get_client():
    # One shared client. Creating a fresh PersistentClient per query left the HNSW
    # index in an inconsistent state across processes and silently dropped true
    # nearest neighbours from results.
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(cfg.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def embed(texts):
    return get_embedder().encode(list(texts), normalize_embeddings=True).tolist()


def build():
    chunks = load_chunks()
    client = get_client()

    # Idempotent by delete-and-recreate: the store is derived wholesale from one
    # static PDF, so a clean rebuild is the honest operation. Upsert would leave
    # stale vectors behind whenever an entry is removed or re-split.
    try:
        client.delete_collection(cfg.CHROMA_COLLECTION)
    except Exception:
        pass
    # search_ef defaults to 10, which on this small store (444 vectors) can miss
    # the true nearest neighbour for low-signal queries like bare acronyms. A wide
    # beam makes retrieval effectively exact and deterministic across runs.
    coll = client.create_collection(
        cfg.CHROMA_COLLECTION,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 200,
            "hnsw:search_ef": 200,
            "hnsw:M": 32,
        },
    )

    texts = [c["text"] for c in chunks]
    coll.add(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=embed(texts),
        documents=texts,
        metadatas=[c["metadata"] for c in chunks],
    )
    return coll


def get_collection():
    return get_client().get_collection(cfg.CHROMA_COLLECTION)


def ensure_collection():
    """Return the collection, building it from entries.json if it is missing or empty.
    Lets a fresh deploy (e.g. Streamlit Cloud) come up without a pre-built index."""
    try:
        coll = get_collection()
        if coll.count() > 0:
            return coll
    except Exception:
        pass
    return build()


def query(text, k=None):
    k = k or cfg.TOP_K
    return get_collection().query(query_embeddings=embed([text]), n_results=k)


def main():
    coll = build()
    print(f"built collection '{cfg.CHROMA_COLLECTION}' at {cfg.CHROMA_DIR}")
    print(f"vectors: {coll.count()}")
    for q in ["bioavailability", "generic substitution", "budget impact"]:
        res = query(q, k=3)
        print(f"\nquery: {q}")
        for term, dist in zip(
            (m["term"] for m in res["metadatas"][0]), res["distances"][0]
        ):
            print(f"  {dist:.4f}  {term}")


if __name__ == "__main__":
    main()
