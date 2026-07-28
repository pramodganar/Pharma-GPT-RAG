"""Hybrid retrieval: reciprocal rank fusion of the dense and BM25 rankings.

The retrieval eval showed BM25 tying the dense retriever overall while missing a
different set of queries, so combining them was the documented first upgrade. The
two scores are not comparable (cosine distance vs BM25 term saturation), so they
are fused by rank rather than by score: RRF sums 1/(rrf_k + rank) across rankings,
which needs no score normalisation and no tuned weights.

Note what the README's "a hybrid union would reach hit@3 0.90" actually was: an
oracle ceiling, counting a query as a hit if *either* retriever found it anywhere
in its top-3. A real fused ranking has to fit both retrievers' candidates into the
same 3 slots, so it can land below that ceiling. eval_retrieval reports both.
"""

from . import config as cfg
from .chunking import load_chunks
from .embed_store import embed, ensure_collection

_bm25 = None
_chunks = None


def _bm25_index():
    """BM25 over the same chunks the vector store holds. Built once per process;
    at 444 chunks this is milliseconds and a trivial amount of memory."""
    global _bm25, _chunks
    if _bm25 is None:
        from rank_bm25 import BM25Okapi

        from .eval_retrieval import _tokenize

        _chunks = load_chunks()
        _bm25 = BM25Okapi([_tokenize(c["text"]) for c in _chunks])
    return _bm25, _chunks


def rrf(rankings, rrf_k=60):
    """Fuse ranked id lists into one ranked id list by reciprocal rank fusion.

    rrf_k damps the contribution of top ranks so a single retriever cannot dominate
    on one confident hit; 60 is the value from the original RRF paper and is left
    untuned here — tuning it on a 30-query set would be fitting noise.
    """
    scores = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking):
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
    # Ties broken by first appearance, so the order is deterministic.
    order = {key: i for i, key in enumerate(k for r in rankings for k in r)}
    return sorted(scores, key=lambda key: (-scores[key], order[key]))


def _bm25_ranking(question, n):
    bm25, chunks = _bm25_index()
    from .eval_retrieval import _tokenize

    scores = bm25.get_scores(_tokenize(question))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    # Chunk ids in the vector store are the index of the chunk in this same list
    # (embed_store.build ids them with range(len(chunks))), which is what lets the
    # two rankings be fused by id at all.
    return [str(i) for i in top]


def retrieve(question, k=None, candidates=None):
    """Return rag_chain-shaped docs from the fused ranking, closest-first by fusion.

    Each retriever contributes `candidates` results; fusing deeper lists than k lets
    a term that neither ranks first, but both rank well, surface into the top k.
    """
    k = k or cfg.TOP_K
    candidates = candidates or cfg.HYBRID_CANDIDATES
    coll = ensure_collection()

    res = coll.query(query_embeddings=embed([question]), n_results=candidates)
    dense_ids = res["ids"][0]
    known = {
        cid: {"text": text, "meta": meta, "distance": dist}
        for cid, text, meta, dist in zip(
            dense_ids, res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    }

    fused = rrf([dense_ids, _bm25_ranking(question, candidates)])[:k]

    # A chunk BM25 found but dense did not has no distance yet, and the UI shows a
    # relevance score for every source. One get() fills them in rather than leaving
    # a hole or inventing a number.
    missing = [cid for cid in fused if cid not in known]
    if missing:
        got = coll.get(ids=missing, include=["documents", "metadatas", "embeddings"])
        qvec = embed([question])[0]
        for cid, text, meta, vec in zip(
            got["ids"], got["documents"], got["metadatas"], got["embeddings"]
        ):
            # Both sides are L2-normalised, so cosine distance is 1 - dot.
            known[cid] = {
                "text": text,
                "meta": meta,
                "distance": 1.0 - sum(a * b for a, b in zip(qvec, vec)),
            }

    docs = []
    for cid in fused:
        hit = known[cid]
        docs.append(
            {
                "text": hit["text"],
                "term": hit["meta"]["term"],
                "page": hit["meta"]["page_start"],
                "source": hit["meta"].get("source", ""),
                "distance": hit["distance"],
            }
        )
    return docs
