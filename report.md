# Pharma-GPT — project report

## Problem

Build a reference assistant that answers natural-language questions about
pharmaceutical terminology, grounded strictly in one source: the WHO Collaborating
Centre / PPRI *Glossary of Pharmaceutical Terms* (2016). The assistant must answer
from the glossary only, refuse questions the glossary does not cover, and cite the
term and page it used.

## Data profile

- Source: the [2016 update of the PPRI glossary](https://ppri.goeg.at/sites/ppri.goeg.at/files/inline-files/Glossary_Update2016_final.pdf)
  — one 140-page A4 PDF with a real text layer (no OCR).
- Glossary content is pages 9-128. Pages 1-8 are front matter (cover, intro, PPRI
  background); pages 129-140 are the reference list and acknowledgements. Both are
  excluded.
- Every page carries a footer (page number + a copyright line) that is stripped.
- Parsed result: **413 entries**. Each entry is `{term, definition_text, sources[],
  page_start}`. 7 entries have no `[Source:]` line (cross-reference/redirect terms);
  a few carry more than one source.

The one non-obvious thing about the PDF: pdfplumber renders bold text (every term
heading, plus inline emphasised terms) as each glyph repeated four times —
`AAAABBBBCCCC` for `ABC`. That artifact is both the problem and the solution: I
collapse it to recover the term, and I use "is this line entirely bold?" as the
heading detector. See DECISIONS.md for why pdfplumber replaced pypdf.

## Pipeline

1. **Ingest** (`src/ingest.py`): pdfplumber text extraction, footer stripping,
   front/back-matter exclusion, de-bolding, and a line-by-line parser that groups
   headings, multi-paragraph definitions, and multi-line `[Source:]` attributions
   into entries.
2. **Chunk** (`src/chunking.py`): one entry = one chunk by default. The 23 entries
   over 1200 chars are split on natural boundaries with 150-char overlap, and every
   sub-chunk is prefixed with its term name so the split pieces still retrieve on the
   term. Result: **444 chunks**, median 324 chars, max 1200.
3. **Embed + store** (`src/embed_store.py`): `all-MiniLM-L6-v2`, normalized, stored in
   ChromaDB (cosine space) at `chroma_db/`. Rebuild is delete-and-recreate.
4. **Retrieve + answer** (`src/rag_chain.py`): dense top-k (k=5) → a prompt that
   forbids outside knowledge, requires refusal when uncovered, and requires term+page
   citation → the configured LLM.
5. **Serve** (`app.py`): Streamlit chat with a per-answer Sources panel.

The LLM sits behind a factory (`src/llm_factory.py`) with two providers: Gemini
`gemini-2.5-flash` (default, free tier, deployable) and Ollama `llama3.1` (local,
private). One env value picks the provider; nothing else in the code knows which is
active.

## Retrieval evaluation

30 gold queries, 10 per category, expected terms drawn from parsed entries. Metric is
hit@k — the expected term appears in the top-k retrieved chunk metadata. A BM25
lexical baseline runs over the same chunks and queries.

```
dense (all-MiniLM-L6-v2):              BM25 lexical baseline:
category      n  hit@1  hit@3  hit@5   category      n  hit@1  hit@3  hit@5
------------------------------------   ------------------------------------
direct       10   1.00   1.00   1.00   direct       10   0.80   1.00   1.00
paraphrased  10   0.40   0.70   0.70   paraphrased  10   0.50   0.70   0.70
abbreviation 10   0.70   0.90   0.90   abbreviation 10   0.80   0.90   0.90
------------------------------------   ------------------------------------
overall      30   0.70   0.87   0.87   overall      30   0.70   0.87   0.87
```

BM25 ties dense overall and edges it at hit@1 on paraphrases and abbreviations;
dense wins direct hit@1. The retrievers miss different queries: 3 of 30 are missed
by both at hit@3, so a hybrid union would reach 0.90. The shared misses ("copay"
and two terse paraphrases) have neither a lexical nor a semantic bridge to their
entries — those need acronym/alias expansion, not a better ranker.

These numbers are reproducible: retrieval uses a wide HNSW search beam
(`hnsw:search_ef=200`), so `python -m src.eval_retrieval` returns the same table on
every run. An earlier default beam (`search_ef=10`) silently missed true nearest
neighbours for low-signal queries and made the score vary run to run.

The paraphrase queries are worded to share few words with the definitions they
target, so the paraphrase score measures semantic retrieval rather than lexical
overlap. hit@3 == hit@5 everywhere: when the right term is retrievable it is already
in the top 3, so the remaining failures are recall, not ranking. The four misses at
hit@3 are one abbreviation ("copay", spelled "Co-payment") and three terse paraphrases
(e.g. affordability of a new treatment -> Budget Impact Analysis) — the known weakness
of dense-only retrieval on short or low-content queries.

## Known limitations

- **Single document, 2016 vintage.** The assistant knows this glossary and nothing
  else. Terminology and figures are frozen at 2016; it is a health-economics/policy
  glossary, so it has no dosage-form entries (e.g. no "enteric coating").
- **Dense-only retrieval.** `all-MiniLM-L6-v2` is small and fast but weak on bare
  acronyms and closed-form spellings. Measured against the BM25 baseline, a hybrid
  union lifts hit@3 from 0.87 to 0.90; the remaining misses are shared by both
  retrievers and need alias expansion.
- **Extraction residue.** The PDF's list-bullet glyphs (which decode as `»`/`(cid:2)`)
  are stripped during parsing; author names in source strings keep their accented
  Unicode. No characters are lost to replacement chars.
- **No answer-level grounding check.** Refusal and citation are enforced by the
  prompt, not verified programmatically after generation.

## Next steps

- Hybrid retrieval (BM25 union dense) with a reranker; re-run the eval set.
- An acronym/synonym expansion map built from the parenthetical aliases already in
  the term headings.
- A post-generation check that every cited term actually appears in the retrieved
  context.
- Add a second glossary and namespace collections to test multi-source retrieval.
