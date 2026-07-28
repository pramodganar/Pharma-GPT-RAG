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
dense wins direct hit@1. The retrievers miss different queries, so a hybrid was
built and measured (`RETRIEVER=hybrid`, reciprocal rank fusion):

```
overall   hit@1  hit@3  hit@5
dense      0.70   0.87   0.87
BM25       0.70   0.87   0.87
hybrid     0.77   0.87   0.87
ceiling    0.83   0.90   0.90
```

The prediction in the previous version of this report — that a hybrid would reach
hit@3 0.90 — was wrong, and worth recording as wrong. 0.90 is the oracle ceiling:
the score if a query counts as a hit whenever *either* retriever has the term in its
own top-3. A fused ranking has to place both retrievers' candidates in the same 3
slots, and the swaps cost what they gain, so hit@3 stays 0.87. The gain is at hit@1
(0.70 -> 0.77), concentrated on abbreviations (0.70 -> 0.90) and paraphrases
(0.40 -> 0.50), and it costs one direct query (1.00 -> 0.90). Since the prompt asks
the model to answer from the single most relevant term, hit@1 is the metric worth
buying — but the direct-category regression is why hybrid is a flag and dense stays
the default. The shared misses ("copay" and two terse paraphrases) have neither a
lexical nor a semantic bridge to their entries and need acronym/alias expansion; no
ranker can retrieve what neither signal reaches.

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

## Generation evaluation

Layered on the retrieval hit@k (which is generation-free and non-circular):

- **Out-of-scope refusal 3/3.** A drug-dose question, a general-knowledge question,
  and a clinical recommendation are each refused with the
  glossary-does-not-define-it response. Checked offline from committed records; a
  refusal must match a marker *and* stay short, so refuse-then-guess counts as a
  failure.
- **RAGAS** over the 6 in-scope records, judged locally with `llama3.1` 8B via
  Ollama: faithfulness 0.68, answer relevancy 0.91, context precision 1.00.
  Context precision 1.00 agrees with direct hit@1 = 1.00. Faithfulness is the
  noisiest metric under a small judge (imperfect claim extraction, under-credited
  paraphrase) — it is a regression baseline pinned to this judge, not ground truth.

## Known limitations

- **Single document, 2016 vintage.** The assistant knows this glossary and nothing
  else. Terminology and figures are frozen at 2016; it is a health-economics/policy
  glossary, so it has no dosage-form entries (e.g. no "enteric coating").
- **Retrieval recall is capped by vocabulary, not by ranking.** `all-MiniLM-L6-v2`
  is small and fast but weak on bare acronyms and closed-form spellings. Hybrid
  fusion lifts hit@1 but leaves hit@3 at 0.87, because the remaining misses are
  shared by both retrievers and need alias expansion.
- **Extraction residue.** The PDF's list-bullet glyphs (which decode as `»`/`(cid:2)`)
  are stripped during parsing; author names in source strings keep their accented
  Unicode. No characters are lost to replacement chars.
- **No answer-level grounding check.** Refusal and citation are enforced by the
  prompt, not verified programmatically after generation.

## Next steps

- A cross-encoder reranker over the fused candidate set: fusion moved hit@1 but not
  hit@3, so reordering candidates is the next lever.
- An acronym/synonym expansion map built from the parenthetical aliases already in
  the term headings.
- A post-generation check that every cited term actually appears in the retrieved
  context.
- Add a second glossary and namespace collections to test multi-source retrieval.
