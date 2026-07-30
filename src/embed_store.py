"""Embed chunks with all-MiniLM-L6-v2 and persist them to a local ChromaDB store.

Running this rebuilds the collection from scratch. Build with: python -m src.embed_store
Add --clean to wipe the store directory first, reclaiming the segment dirs Chroma
leaves behind on every rebuild.
"""

import logging
import shutil
import sys

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


def clean_store():
    """Delete the persist directory so the next build starts from nothing.

    delete_collection drops the collection from Chroma's metadata db but leaves its
    HNSW segment directory on disk, so repeated rebuilds accumulate orphaned segments
    (8 dirs / 26 MB for one collection here). Removing the directory is the only
    cleanup that does not depend on Chroma's internals, which differ across the
    versions this repo has been run on.

    Only valid before a client is open — an open client holds the sqlite file, and on
    Windows the delete would fail half-done. Hence a deliberate maintenance command
    (`python -m src.embed_store --clean`) in a fresh process, never something
    ensure_collection does behind the running app's back.
    """
    global _client
    if _client is not None:
        raise RuntimeError(
            "a Chroma client is already open in this process; run "
            "`python -m src.embed_store --clean` on its own instead"
        )
    if cfg.CHROMA_DIR.exists():
        shutil.rmtree(cfg.CHROMA_DIR)


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
    Lets a fresh deploy (e.g. Streamlit Cloud) come up without a pre-built index.

    A store written by a different Chroma version is a fourth case, and it is not
    recoverable here: get_collection cannot parse it, and neither can build(), because
    create_collection reads the same unparseable row while checking for a name clash.
    clean_store is the fix but refuses to run once a client is open — and one is, by
    now — so this raises a sentence naming the command instead of letting a Chroma
    internal (KeyError: '_type') reach the user as a traceback.
    """
    try:
        coll = get_collection()
        if coll.count() > 0:
            return coll
    except Exception:
        pass
    try:
        return build()
    except Exception as exc:
        # Only claim staleness when a store actually exists; otherwise this is an
        # ordinary build failure (missing entries.json, no disk) and must surface as is.
        if cfg.CHROMA_DIR.exists():
            raise RuntimeError(
                f"cannot read or rebuild the Chroma store at {cfg.CHROMA_DIR} "
                f"(chromadb {chromadb.__version__}); it was most likely written by a "
                "different Chroma version. Rebuild it with "
                "`python -m src.embed_store --clean`."
            ) from exc
        raise


def query(text, k=None):
    k = k or cfg.TOP_K
    return get_collection().query(query_embeddings=embed([text]), n_results=k)


def main():
    if "--clean" in sys.argv:
        clean_store()
        print(f"removed {cfg.CHROMA_DIR}")
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
