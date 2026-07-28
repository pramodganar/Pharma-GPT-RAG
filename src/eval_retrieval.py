"""Retrieval evaluation: hit@k of the gold query set against the vector store,
with a BM25 lexical baseline over the same chunks and queries.

The baseline answers the question the dense score alone cannot: is the embedding
model earning its cost, or would plain lexical match do? On a glossary, direct
queries contain the term verbatim, so BM25 is expected to win those for free —
the dense retriever has to justify itself on paraphrases.

Run with: python -m src.eval_retrieval
"""

import json
import re

from . import config as cfg
from .embed_store import embed, get_collection

KS = (1, 3, 5)


def _retrieved_terms(query, k):
    res = get_collection().query(query_embeddings=embed([query]), n_results=k)
    return [m["term"] for m in res["metadatas"][0]]


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_retriever():
    """Return a query -> ranked terms function over the same chunks the dense
    store indexes, or None if rank_bm25 is not installed."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return None
    from .chunking import load_chunks

    chunks = load_chunks()
    bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])

    def retrieve(query, k):
        scores = bm25.get_scores(_tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [chunks[i]["metadata"]["term"] for i in top]

    return retrieve


def evaluate(queries=None, retriever=None):
    if queries is None:
        with open(cfg.EVAL_QUERIES_JSON, encoding="utf-8") as f:
            queries = json.load(f)
    retriever = retriever or _retrieved_terms
    maxk = max(KS)
    results = []
    for q in queries:
        terms = retriever(q["query"], maxk)
        hits = {k: q["expected_term"] in terms[:k] for k in KS}
        results.append({**q, "retrieved": terms, "hits": hits})
    return results


def _rate(rows, k):
    return sum(r["hits"][k] for r in rows) / len(rows) if rows else 0.0


def _table(results):
    cats = ["direct", "paraphrased", "abbreviation"]
    header = f"{'category':<14}{'n':>4}" + "".join(f"{'hit@'+str(k):>9}" for k in KS)
    lines = [header, "-" * len(header)]
    for cat in cats:
        rows = [r for r in results if r["category"] == cat]
        lines.append(
            f"{cat:<14}{len(rows):>4}" + "".join(f"{_rate(rows, k):>9.2f}" for k in KS)
        )
    lines.append("-" * len(header))
    lines.append(
        f"{'overall':<14}{len(results):>4}" + "".join(f"{_rate(results, k):>9.2f}" for k in KS)
    )
    return "\n".join(lines)


def main():
    results = evaluate()
    print("dense (all-MiniLM-L6-v2):")
    print(_table(results))

    bm25 = bm25_retriever()
    if bm25 is None:
        print("\nrank_bm25 not installed; skipping the lexical baseline.")
    else:
        print("\nBM25 lexical baseline (same chunks, same queries):")
        print(_table(evaluate(retriever=bm25)))

    misses = [r for r in results if not r["hits"][3]]
    print(f"\nmisses at hit@3: {len(misses)}")
    for r in misses:
        print(f"\n  [{r['category']}] {r['query']!r}")
        print(f"  expected: {r['expected_term']}")
        print(f"  retrieved top-3: {r['retrieved'][:3]}")


if __name__ == "__main__":
    main()
