"""Retrieval-augmented answering over the glossary.

Smoke test: python -m src.rag_chain "What is bioavailability?"
"""

import sys

from langchain_core.prompts import ChatPromptTemplate

from . import config as cfg
from .embed_store import embed, ensure_collection
from .llm_factory import friendly_error, get_llm

PROMPT = ChatPromptTemplate.from_template(
    """You are a reference assistant for pharmaceutical terminology. Answer using only the glossary context provided.

The context is a list of glossary terms ordered most relevant first.

Rules:
- Use only the context below. Do not add outside knowledge or infer beyond it.
- Answer from the single most relevant term. Use another term only if the question genuinely spans several; do not default to a lower one.
- If the context does not cover the question, reply that the glossary does not define it, and stop. Do not guess.
- Cite the term name(s) and page number(s) you drew on.
- Write plainly and professionally, for pharmacists and students.

Context:
{context}

Question: {question}

Answer:"""
)


def retrieve(question, k=None):
    k = k or cfg.TOP_K
    # ensure_ rather than get_: an unbuilt index raises a Chroma "collection does not
    # exist" that friendly_error can only report as a backend failure, blaming the LLM
    # for a missing index. Building it is what the caller wanted anyway.
    res = ensure_collection().query(query_embeddings=embed([question]), n_results=k)
    docs = []
    for text, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        docs.append(
            {
                "text": text,
                "term": meta["term"],
                "page": meta["page_start"],
                "source": meta.get("source", ""),
                "distance": dist,
            }
        )
    return docs


def format_context(docs):
    return "\n\n".join(f"Term: {d['term']} (page {d['page']})\n{d['text']}" for d in docs)


def unique_sources(docs):
    """Collapse retrieved docs to one per (term, page), keeping the closest hit.

    k=5 often returns several sub-chunks of one long entry; showing them as
    separate sources reads as duplicates. Lives here rather than in the UI so the
    citation view is testable without importing Streamlit.
    """
    best = {}
    for d in docs:
        key = (d["term"], d["page"])
        if key not in best or d["distance"] < best[key]["distance"]:
            best[key] = d
    return list(best.values())


def cited_terms(docs):
    seen = []
    for d in docs:
        key = (d["term"], d["page"])
        if key not in seen:
            seen.append(key)
    return seen


def answer(question, k=None, llm=None):
    docs = retrieve(question, k)
    llm = llm or get_llm()
    messages = PROMPT.format_messages(context=format_context(docs), question=question)
    response = llm.invoke(messages)
    return {"answer": response.content, "sources": docs}


def answer_stream(question, k=None, llm=None):
    """Retrieve, then return (sources, token generator). Retrieval is done up front
    so the caller has citations before generation starts."""
    docs = retrieve(question, k)
    llm = llm or get_llm()
    messages = PROMPT.format_messages(context=format_context(docs), question=question)

    def tokens():
        for chunk in llm.stream(messages):
            yield chunk.content

    return docs, tokens()


def main():
    question = " ".join(sys.argv[1:]).strip() or "What is bioavailability?"
    try:
        out = answer(question)
    except Exception as exc:
        print(friendly_error(exc))
        return
    print(out["answer"])
    print("\nRetrieved terms:")
    for term, page in cited_terms(out["sources"]):
        print(f"  - {term} (page {page})")


if __name__ == "__main__":
    main()
