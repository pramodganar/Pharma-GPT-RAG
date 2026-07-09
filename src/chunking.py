"""Turn parsed entries into retrieval chunks.

Default is one entry per chunk: a glossary entry is a self-contained semantic
unit. The few entries longer than the char threshold are split on natural
boundaries, and every sub-chunk is prefixed with the term name so the split
pieces still retrieve on the term. See DECISIONS.md.
"""

import json

from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config as cfg


def _chunk_text(term, body):
    return f"{term}\n{body}"


def chunk_entries(entries):
    chunks = []
    for entry in entries:
        term = entry["term"]
        body = entry["definition_text"]
        source = " | ".join(entry.get("sources", []))
        meta_base = {"term": term, "page_start": entry["page_start"], "source": source}

        full = _chunk_text(term, body)
        if len(full) <= cfg.MAX_CHUNK_CHARS:
            pieces = [full]
        else:
            # Prefix each sub-chunk with the term; budget its length against the split size.
            prefix = f"{term}\n"
            sub_splitter = RecursiveCharacterTextSplitter(
                chunk_size=cfg.MAX_CHUNK_CHARS - len(prefix),
                chunk_overlap=cfg.CHUNK_OVERLAP_CHARS,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            pieces = [prefix + p for p in sub_splitter.split_text(body)]

        for i, text in enumerate(pieces):
            chunks.append({"text": text, "metadata": {**meta_base, "chunk_index": i}})
    return chunks


def load_chunks():
    entries = json.load(open(cfg.ENTRIES_JSON, encoding="utf-8"))
    return chunk_entries(entries)


def main():
    import statistics

    chunks = load_chunks()
    sizes = [len(c["text"]) for c in chunks]
    split_terms = {c["metadata"]["term"] for c in chunks if c["metadata"]["chunk_index"] > 0}
    print(f"chunks: {len(chunks)} from entries")
    print(f"size chars min/median/max: {min(sizes)} / {int(statistics.median(sizes))} / {max(sizes)}")
    print(f"entries that were split: {len(split_terms)}")


if __name__ == "__main__":
    main()
