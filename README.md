# Pharma-GPT

A retrieval-augmented chatbot that answers questions about pharmaceutical
terminology, grounded strictly in the WHO/PPRI *Glossary of Pharmaceutical Terms*
(2016). It retrieves the relevant glossary entries and asks an LLM to answer only
from them, citing the term and page. If the glossary does not cover a question, it
says so instead of guessing.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pharma-gpt-rag-dpstrt54vjs64peusbfvzm.streamlit.app/)

**Live demo:** https://pharma-gpt-rag-dpstrt54vjs64peusbfvzm.streamlit.app/

![Pharma-GPT answering "What is bioavailability?": the answer streams in, cites the glossary term and page, and the Sources panel lists each retrieved entry with its page and relevance score](docs/demo.gif)

## Problem

Regulatory-affairs and market-access teams repeatedly hit unfamiliar pharma-policy
and health-economics terms (ATC, DDD, QALY, biosimilar, budget impact analysis).
The authoritative reference is a 140-page PDF, so a lookup means scrolling and the
answer carries no traceability. Pharma-GPT turns that PDF into a grounded lookup:
ask in natural language, get the definition **with its glossary term and page**, and
get an explicit "not covered" when the question falls outside the glossary.

Success criteria: (1) every answer grounded in a retrieved entry and cites term +
page; (2) out-of-scope questions (e.g. drug dosages) are refused, not guessed;
(3) retrieval finds the right entry (hit@3 0.87); (4) interactive latency (streamed
tokens; a one-time ~9s index build on a cold deploy).

## Corpus

The WHO Collaborating Centre / PPRI [*Glossary of Pharmaceutical Terms*, 2016
update](https://ppri.goeg.at/sites/ppri.goeg.at/files/inline-files/Glossary_Update2016_final.pdf)
(listed on the [PPRI glossary page](https://ppri.goeg.at/about_translations)) —
one 140-page PDF with a real text layer (no OCR). Glossary content is pages 9-128;
front matter (1-8) and the reference list (129-140) are excluded. Its scope is
pharmaceutical **policy and health economics** — pricing, reimbursement, HTA,
ATC/DDD, pharmacovigilance — not drug formulations or clinical dosing, so formulation
questions (e.g. enteric coating) fall outside it by design and are refused. Parsed
into **413 entries** `{term, definition_text, sources[], page_start}`. The one
non-obvious thing: pdfplumber renders bold text (every term heading) as each glyph
repeated four times — `AAAABBBBCCCC` for `ABC`. That artifact is both the problem
and the solution — it is collapsed to recover the term and used as the heading
detector. See [DECISIONS.md](DECISIONS.md).

## Architecture

```
  Pharmacy_Dictionary.pdf
          |  ingest.py      pdfplumber -> de-bold -> parse entries
          v
  413 entries (entries.json)
          |  chunking.py    one entry = one chunk; 23 long entries split
          v                 with term prefix + 150-char overlap
  444 chunks
          |  embed_store.py all-MiniLM-L6-v2 (384-dim, normalized, cosine)
          v
  ChromaDB  <-- HNSW, search_ef=200 (deterministic at this scale)
          |  rag_chain.py   query -> embed -> top-5 retrieve
          v
  grounded prompt  (answer only from context, cite term+page, refuse if uncovered)
          |  llm_factory.py Gemini gemini-2.5-flash  |  Ollama llama3.1
          v
  streamed answer + Sources panel   (app.py, Streamlit)
```

## Retrieval evaluation

30 gold queries, 10 per category; the expected term must appear in the top-k
retrieved chunks. Reproducible — `python -m src.eval_retrieval` prints the same
tables every run (see *Design decisions* for why). A BM25 baseline over the same
chunks and queries is scored alongside, so the dense numbers have something dumb
to beat.

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

The baseline is a finding, not a formality: BM25 ties dense overall (0.70 / 0.87 /
0.87) and edges it at hit@1 on paraphrases and abbreviations, while dense wins
direct hit@1 (1.00 vs 0.80). The two retrievers miss different queries, which is why
hybrid BM25+dense was built next rather than a bigger embedding model.

### Hybrid retrieval, and a prediction that did not hold

`RETRIEVER=hybrid` fuses the two rankings with reciprocal rank fusion (rank-based,
so no score normalisation between cosine distance and BM25). Measured on the same
30 queries:

```
overall hit@1  hit@3  hit@5      per-category hit@1 (hybrid):
dense    0.70   0.87   0.87        direct       0.90  (dense 1.00)
BM25     0.70   0.87   0.87        paraphrased  0.50  (dense 0.40)
hybrid   0.77   0.87   0.87        abbreviation 0.90  (dense 0.70)
ceiling  0.83   0.90   0.90
```

An earlier draft of this README predicted a hybrid would reach hit@3 0.90. **It does
not.** That 0.90 is the *oracle ceiling* — the score if a query counts as a hit
whenever either retriever has the term anywhere in its own top-3. A real fused
ranking has to fit both retrievers' candidates into the same 3 slots, and here the
swaps cost exactly what they gain: hit@3 stays 0.87. The ceiling row is now printed
by the eval so the difference between "either retriever knew it" and "the fused
ranking ranked it" cannot be conflated again.

What the fusion does buy is **hit@1: 0.70 → 0.77**, which is the metric that matters
most here, because the prompt tells the model to answer from the single most relevant
term. The gain is concentrated where each retriever was weak alone — abbreviations
0.70 → 0.90, paraphrases 0.40 → 0.50 — and it costs one direct query (1.00 → 0.90),
where dense alone was perfect.

Dense stays the default so the published numbers above remain the reproducible ones;
hybrid is one env var (`RETRIEVER=hybrid`). Given the trade is +0.07 overall hit@1
against a regression on the category the glossary is most often queried with, that
is a judgement call, not a free win — which is why it is a flag rather than a
silent replacement.

Paraphrase queries are worded to share few words with their target definition, so
that score measures semantic retrieval, not lexical overlap. `hit@3 == hit@5`
everywhere: when the right term is retrievable it is already in the top 3, so the
remaining misses are recall failures, not ranking. Generation quality is evaluated
separately, below.

## Generation evaluation

Two checks on the answers themselves, layered on the retrieval hit@k above:

- **Out-of-scope refusal — 3/3.** The adversarial probes (a drug dose, a
  general-knowledge question, a clinical recommendation) are each refused with the
  glossary-does-not-define-it response rather than answered. Computed by
  `python -m src.eval_ragas --refusal-only` straight from the committed records — no
  LLM judge, so it reproduces offline.
- **RAGAS faithfulness / answer relevancy / context precision.** The harness
  (`src/eval_ragas.py`, deps in `requirements-eval.txt`) scores the 6 in-scope
  answers with an LLM-as-judge. Judged with the repo's own local provider
  (`llama3.1` 8B via Ollama) because the triad needs ~40 judge calls, over the
  Gemini free-tier daily cap:

  ```
  faithfulness         0.68
  answer relevancy     0.91
  context precision    1.00
  ```

  Context precision 1.00: the relevant glossary entry was ranked usefully in the
  context for all 6 questions, consistent with direct hit@1 = 1.00. Faithfulness
  0.68 is the judge-noisiest metric: an 8B judge extracts and verifies claims
  imperfectly and under-credits paraphrase, so read it as a floor and a regression
  baseline pinned to this judge, not an absolute — a stronger judge moves the
  number. The judged inputs are committed (`data/processed/ragas_records.json`),
  so the records are inspectable without a run; reproduce with
  `LLM_PROVIDER=ollama python -m src.eval_ragas`. RAGAS judges an LLM with an
  LLM — the non-circular anchors remain the retrieval hit@k and the refusal check
  above.

## Setup

Python 3.11–3.12 (tested on 3.11; developed on Windows, requirements resolve on
macOS/Linux too — the pinned wheels have no 3.13+/3.14 builds). From the project
root:

```
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

The parsed corpus (`data/processed/entries.json`) is committed, so a fresh clone can
build the index without the source PDF:

```
python -m src.embed_store
```

Re-parsing the PDF is only needed to change the ingest logic. Download the 2016
glossary PDF (link in *Corpus* above), save it as `data/raw/Pharmacy_Dictionary.pdf`,
then:

```
python -m src.ingest        # rewrites data/processed/entries.json
python -m src.embed_store
```

The Streamlit app also builds the index on first boot if it is missing, so running
`src.embed_store` by hand is optional locally. Rebuilding leaves Chroma's previous
HNSW segment directory behind each time; `python -m src.embed_store --clean` wipes
`chroma_db/` first and rebuilds from scratch.

Pick a provider. **Path A — Gemini (the hosted-demo path):** get a free key at
Google AI Studio, then create `.env` from `.env.example`:

```
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_key_here
```

**Path B — Ollama (local, private):** install Ollama, then:

```
ollama pull llama3.1
```

and set `LLM_PROVIDER=ollama` in `.env`.

Check retrieval and run the app:

```
python -m src.eval_retrieval
streamlit run app.py
```

Generation eval (optional, needs a built index + a working provider):

```
python -m src.eval_ragas --refusal-only   # offline refusal check, no judge calls
pip install -r requirements-eval.txt
python -m src.eval_ragas --generate       # run the pipeline, save records
python -m src.eval_ragas                  # refusal check + RAGAS (live judge calls)
```

## Deploy to Streamlit Community Cloud

The default Gemini path is chosen so the demo runs on the free tier with no local
model, and `requirements.txt` pins CPU-only torch so the image stays small enough for
it. Deploy steps:

1. Push this repo to GitHub (the committed `entries.json` is all the app needs; the
   PDF and `chroma_db/` stay untracked).
2. On [share.streamlit.io](https://share.streamlit.io), create an app pointing at
   `app.py` on this repo. In **Advanced settings**, set the Python version to
   **3.12** — Cloud now defaults to 3.14, for which the pinned wheels (pillow, grpcio,
   chroma-hnswlib, tokenizers, and the `torch==2.12.1+cpu` build) have no build and
   would fail compiling from source.
3. In the app's **Settings → Secrets**, add (this is the deployment equivalent of
   `.env` — do not commit a `.env`):

   ```
   LLM_PROVIDER = "gemini"
   GOOGLE_API_KEY = "your_key_here"
   ```

4. First boot builds the vector index from `entries.json` (~9s, cached for the life
   of the container); later starts are a no-op. Ollama is not reachable from
   Community Cloud, so the hosted demo must use Gemini.

## Layout

```
src/ingest.py         PDF -> structured entries (data/processed/entries.json)
src/chunking.py       entries -> term-aware chunks
src/embed_store.py    chunks -> ChromaDB (chroma_db/)
src/hybrid.py         BM25 + dense fused by reciprocal rank (RETRIEVER=hybrid)
src/eval_retrieval.py hit@k over a 30-query gold set (dense, BM25, hybrid)
src/eval_ragas.py     generation eval (RAGAS + out-of-scope refusal rate)
src/llm_factory.py    provider selection (gemini | ollama)
src/rag_chain.py      retrieve -> prompt -> answer
app.py                Streamlit chat UI
src/config.py         all paths, model names, and parameters
report.md             long-form project report (data profile, eval, limitations)
DECISIONS.md          running log of design choices and rejected alternatives
```

## Key design decisions

Full log with rejected alternatives in [DECISIONS.md](DECISIONS.md); the ones worth
knowing:

- **One entry = one chunk**, not fixed-size windows. A glossary entry is already the
  natural unit of retrieval and answer; only 23 of 413 entries exceed the 1200-char
  cap, and those split on natural boundaries with a term-name prefix so the pieces
  still retrieve on the term.
- **all-MiniLM-L6-v2** for CPU size/speed on a free hosted demo. The honest cost —
  weaker recall on bare acronyms and terse paraphrases — is what the eval measures.
- **Hybrid behind a flag, not by default.** Reciprocal rank fusion of BM25 and dense
  buys +0.07 hit@1 but nothing at hit@3, and regresses direct hit@1 from 1.00 to
  0.90. A measured trade-off with a real downside belongs behind `RETRIEVER=hybrid`,
  not silently swapped in under numbers that were published for dense.
- **ChromaDB** for on-disk persistence + per-vector metadata (term, page, source) in
  one call, which is exactly what citation needs. At 444 vectors the deciding factor
  was less glue code, not speed.
- **Wide HNSW beam (search_ef=200).** Chroma's default of 10 silently dropped true
  nearest neighbours at this small scale and made hit@k vary run to run; widening it
  makes retrieval effectively exact and deterministic. A test guards the fix.
- **Dual LLM provider behind a factory.** One env value picks Gemini (deployable) or
  Ollama (local/private); nothing outside `llm_factory` knows which is active.
- **Build the index on boot** from the committed `entries.json` instead of committing
  `chroma_db/`: a persisted HNSW binary is tied to the Chroma build and OS and can
  fail to load on a different host, and rebuilding keeps the index and embedding
  model in sync.

## Limitations

- **Single document, 2016 vintage.** It knows this glossary and nothing else;
  terminology and figures are frozen at 2016. It is a health-economics/policy
  glossary, so it has no dosage-form or clinical entries — those questions are
  refused, not answered. **Not medical advice.**
- **Retrieval recall is capped by the corpus vocabulary.** Dense alone is weak on
  bare acronyms and closed-form spellings ("copay" vs the glossary's "Co-payment");
  hybrid fixes some of that at hit@1 but not hit@3, because 3 of 30 queries ("copay"
  among them) are missed by *both* retrievers. Those need acronym/alias expansion —
  no ranker can retrieve a term that neither signal reaches.
- **Small eval sets.** Retrieval is scored on 30 gold queries and generation on
  6 in-scope + 3 adversarial records — enough to catch regressions, not to make
  fine-grained comparisons between retrievers or models.
- **No answer-level grounding check.** Refusal and citation are enforced by the
  prompt, not yet verified programmatically after generation.

## License

Code is released under the [MIT License](LICENSE). The WHO/PPRI *Glossary of
Pharmaceutical Terms* (2016) content it references belongs to its respective authors
and is used for reference under its own terms, not relicensed here — see
[NOTICE](NOTICE). (The corpus note lives there rather than appended to `LICENSE`,
which stops GitHub recognising the file as MIT.)

## Future work

- A cross-encoder reranker over the fused candidates — the fusion improved hit@1 but
  left hit@3 flat, so reordering the candidate set is the next lever to try.
- Acronym/synonym expansion built from the parenthetical aliases in term headings.
  The 3 queries both retrievers miss are the ones this would fix.
- A post-generation check that every cited term appears in the retrieved context.
- A second glossary with namespaced collections to test multi-source retrieval.

## Example

```
Q: What is bioavailability?

A: Bioavailability means the rate and extent to which the active substance or
   active moiety is absorbed from a pharmaceutical form and becomes available at
   the site of action. It is useful to distinguish absolute bioavailability
   (compared with intravenous administration) from relative bioavailability
   (compared with another non-intravenous form).

   Term and page: Bioavailability (page 15)

Q: What is the capital of France?

A: The glossary does not define the capital of France.
```
