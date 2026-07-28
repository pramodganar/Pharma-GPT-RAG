from src import rag_chain
from src.rag_chain import (
    PROMPT,
    answer,
    answer_stream,
    cited_terms,
    format_context,
    unique_sources,
)


def _doc(term, page, dist=0.2):
    return {"text": f"{term}\nsome definition text", "term": term, "page": page,
            "source": "", "distance": dist}


class _CapturingLLM:
    """Stand-in for a chat model: records the messages it was invoked with and
    returns a fixed answer, so the chain can be tested with no network or index."""

    def __init__(self):
        self.seen = None

    def invoke(self, messages):
        self.seen = messages

        class _Resp:
            content = "canned grounded answer"

        return _Resp()


class _StreamingLLM:
    """Stand-in that streams fixed chunks and records when streaming started, so the
    'citations are ready before generation' contract can be asserted."""

    def __init__(self, chunks=("canned ", "grounded ", "answer")):
        self.chunks = chunks
        self.seen = None
        self.stream_started = False

    def stream(self, messages):
        self.seen = messages
        self.stream_started = True
        for c in self.chunks:
            yield type("_Chunk", (), {"content": c})


def test_format_context_carries_term_and_page():
    ctx = format_context([_doc("Bioavailability", 15)])
    assert "Bioavailability" in ctx
    assert "15" in ctx


def test_cited_terms_dedupes_preserving_order():
    docs = [_doc("Bioavailability", 15), _doc("Bioavailability", 15), _doc("Bioequivalence", 16)]
    assert cited_terms(docs) == [("Bioavailability", 15), ("Bioequivalence", 16)]


def test_unique_sources_keeps_the_closest_hit_per_term():
    # Sub-chunks of one long entry share term+page; the Sources panel must show the
    # entry once, at its best distance, not once per sub-chunk.
    docs = [
        _doc("Bioavailability", 15, dist=0.42),
        _doc("Bioavailability", 15, dist=0.19),
        _doc("Bioequivalence", 16, dist=0.31),
    ]
    uniq = unique_sources(docs)
    assert [(d["term"], d["distance"]) for d in uniq] == [
        ("Bioavailability", 0.19),
        ("Bioequivalence", 0.31),
    ]


def test_unique_sources_separates_the_same_term_on_different_pages():
    docs = [_doc("Access", 11), _doc("Access", 90)]
    assert len(unique_sources(docs)) == 2


def test_unique_sources_handles_no_results():
    assert unique_sources([]) == []


def test_prompt_enforces_grounding_and_refusal():
    # The refusal contract and grounding rules live in the prompt, so guard them:
    # a reword that drops "answer only from context" or the refusal path would break
    # the whole safety story silently.
    rendered = PROMPT.format(context="X", question="Y")
    low = rendered.lower()
    assert "only" in low and "context" in low
    assert "does not" in low  # "the glossary does not define it"
    assert "cite" in low


def test_answer_injects_retrieved_context_into_llm_call(monkeypatch):
    # Mock retrieval so no index is needed; capture what the LLM actually receives.
    docs = [_doc("Bioavailability", 15)]
    monkeypatch.setattr(rag_chain, "retrieve", lambda q, k=None: docs)
    llm = _CapturingLLM()

    out = answer("What is bioavailability?", llm=llm)

    assert out["answer"] == "canned grounded answer"
    assert out["sources"] == docs
    # The retrieved term and page must appear in the prompt the model saw.
    prompt_text = " ".join(m.content for m in llm.seen)
    assert "Bioavailability" in prompt_text
    assert "15" in prompt_text
    assert "What is bioavailability?" in prompt_text


# answer_stream is the only path app.py calls, so it carries the same guarantees as
# answer() and is tested to the same depth.


def test_answer_stream_yields_the_full_answer(monkeypatch):
    docs = [_doc("Bioavailability", 15)]
    monkeypatch.setattr(rag_chain, "retrieve", lambda q, k=None: docs)
    llm = _StreamingLLM()

    sources, tokens = answer_stream("What is bioavailability?", llm=llm)

    assert sources == docs
    assert "".join(tokens) == "canned grounded answer"


def test_answer_stream_returns_sources_before_generation(monkeypatch):
    # The UI renders citations off the returned sources, so retrieval must be done
    # up front: nothing may be streamed until the caller consumes the generator.
    monkeypatch.setattr(rag_chain, "retrieve", lambda q, k=None: [_doc("Bioavailability", 15)])
    llm = _StreamingLLM()

    sources, tokens = answer_stream("What is bioavailability?", llm=llm)

    assert sources and not llm.stream_started
    next(tokens)
    assert llm.stream_started


def test_answer_stream_uses_the_same_grounded_prompt(monkeypatch):
    # A divergence here would mean the deployed path loses the refusal/citation rules
    # that only answer()'s test covers.
    monkeypatch.setattr(rag_chain, "retrieve", lambda q, k=None: [_doc("Pharmacovigilance", 88)])
    llm = _StreamingLLM()

    _, tokens = answer_stream("What is pharmacovigilance?", llm=llm)
    list(tokens)

    prompt_text = " ".join(m.content for m in llm.seen)
    assert "Pharmacovigilance" in prompt_text
    assert "88" in prompt_text
    assert "What is pharmacovigilance?" in prompt_text


def test_answer_stream_passes_k_through(monkeypatch):
    # The sidebar slider sets k; it has to reach the retriever.
    seen_k = []
    monkeypatch.setattr(
        rag_chain, "retrieve", lambda q, k=None: seen_k.append(k) or [_doc("X", 1)]
    )

    answer_stream("q", k=9, llm=_StreamingLLM())

    assert seen_k == [9]
