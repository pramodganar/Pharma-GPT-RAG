"""Retrieval evaluation: hit@k of the gold query set against the vector store.

Run with: python -m src.eval_retrieval
"""

import json

from . import config as cfg
from .embed_store import embed, get_collection

KS = (1, 3, 5)


def _retrieved_terms(query, k):
    res = get_collection().query(query_embeddings=embed([query]), n_results=k)
    return [m["term"] for m in res["metadatas"][0]]


def evaluate(queries=None):
    if queries is None:
        queries = json.load(open(cfg.EVAL_QUERIES_JSON, encoding="utf-8"))
    maxk = max(KS)
    results = []
    for q in queries:
        terms = _retrieved_terms(q["query"], maxk)
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
    print(_table(results))
    misses = [r for r in results if not r["hits"][3]]
    print(f"\nmisses at hit@3: {len(misses)}")
    for r in misses:
        print(f"\n  [{r['category']}] {r['query']!r}")
        print(f"  expected: {r['expected_term']}")
        print(f"  retrieved top-3: {r['retrieved'][:3]}")


if __name__ == "__main__":
    main()
