from src import chunking
from src import config as cfg


def _entry(term, body, page=10, sources=None):
    return {
        "term": term,
        "definition_text": body,
        "page_start": page,
        "sources": sources or [],
    }


def test_short_entry_is_one_chunk():
    chunks = chunking.chunk_entries([_entry("Adherence", "A short definition.")])
    assert len(chunks) == 1
    c = chunks[0]
    assert c["text"].startswith("Adherence")
    assert c["metadata"]["chunk_index"] == 0


def test_long_entry_splits_with_term_prefix():
    body = "word " * 500  # ~2500 chars, well over the threshold
    chunks = chunking.chunk_entries([_entry("Long Term", body)])
    assert len(chunks) > 1
    assert all(c["text"].startswith("Long Term\n") for c in chunks)
    assert all(len(c["text"]) <= cfg.MAX_CHUNK_CHARS for c in chunks)
    assert [c["metadata"]["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_metadata_integrity():
    chunks = chunking.chunk_entries(
        [_entry("Batch (Lot)", "Some text.", page=21, sources=["Directive 2001/83/EC"])]
    )
    meta = chunks[0]["metadata"]
    assert set(meta) == {"term", "page_start", "source", "chunk_index"}
    assert meta["term"] == "Batch (Lot)"
    assert meta["page_start"] == 21
    assert meta["source"] == "Directive 2001/83/EC"


def test_multiple_sources_joined():
    chunks = chunking.chunk_entries([_entry("X", "def", sources=["A", "B"])])
    assert chunks[0]["metadata"]["source"] == "A | B"
