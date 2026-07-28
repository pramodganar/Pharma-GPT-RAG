# Decisions

Running log of design choices and rejected alternatives, one section per stage.

## Stage 1 — Environment and skeleton

- Pinned exact versions in requirements.txt rather than floating ranges: a reference project should reproduce byte-for-byte a year from now; floating deps rot silently.
- Single config.py as the only source of paths/models/params: keeps magic values out of the modules and makes the k / chunk-size / model swaps interview-legible.
- LLM provider, Ollama model, and base URL read from env with defaults: lets the backend be swapped without editing code, which Stage 6's factory depends on.
- Renamed the source PDF to Pharmacy_Dictionary.pdf (was "Pharmacy Dictionary.pdf"): a space in a hardcoded path is a portability trap on scripts and CLIs.
- chroma_db/ and data/raw contents gitignored: the vector store is a rebuildable artifact and the source PDF is not mine to redistribute; .gitkeep preserves the empty dirs.

## Stage 2 — Ingestion and parsing

- Switched primary parser from pypdf to pdfplumber. pypdf's output was unusable on the two things that matter: (1) bold term headings came out as irregular word-fragment quadruplication ("ABC Analys ABC Analys ABC Analys ABC Analysis is is is") with no clean recovery, and (2) justified body text got spurious mid-word spaces ("the 1 0 to 20 percent", "inclu ding"). pdfplumber renders bold as regular per-character 4x repetition ("AAAABBBBCCCC") which is deterministically recoverable, and its body text is clean.
- Exploited the 4x bold rendering as the heading detector instead of font metadata: a line is a heading only if it is entirely bold, i.e. raw_len / debold_len >= 3.0. This cleanly separates real headings (~3.75-4x) from lines that merely start with a bold sub-label (e.g. "Gene therapy medicine: a product ...", ~1.8x), which are entry body, not new entries. Rejected a fixed shrink-ratio threshold (0.6) — it promoted those sub-labels to false entries (451 -> 413 after fixing).
- De-bold only collapses "bold regions" of 2+ consecutive quadrupled glyphs, never a lone 4-run. This protects body text like "10000" (a single run of four zeros) from being mangled to "10", while still cleaning inline bold terms inside definitions.
- Two extra heading guards: must start with an alphanumeric and must contain no colon. This folds bold bullet list-items ("� Residential health facilities services ... : combined lodging") back into their parent entry instead of splitting them out.
- Footer stripping keys on structure, not the exact copyright string: the © and Ö glyphs decode to replacement chars, so I match a bare-number line plus any line containing "Collaborating Centre" and "Glossary 2016". Front matter is pages 1-8 (config constant); back matter is detected by the "List of references and data sources used" heading rather than a hardcoded page.
- Multi-line [Source: ...] attributions are stitched by accumulating lines until the closing "]"; entries carry a list of sources since several have more than one. 413 entries, 0 footer leaks, plausible count.
- Pinned pdfplumber==0.11.4 (not latest 0.11.10): the latest requires Pillow>=12.2 which conflicts with streamlit 1.41.1's Pillow<12. 0.11.4 produces byte-identical parse output and resolves the dependency conflict.

## Stage 3 — Chunking

- One entry = one chunk by default, not fixed-size splitting. A glossary entry is already the natural unit of retrieval and answer: term heading + definition + attribution. Fixed 500/1000-char windows would cut definitions mid-sentence, strand the term name away from its definition, and return half-answers that read as incoherent context to the LLM. Only 23 of 413 entries exceed the 1200-char threshold, so splitting is the exception, not the rule.
- Kept RecursiveCharacterTextSplitter, but as the fallback splitter for the 23 long entries rather than the primary strategy. It splits on paragraph/sentence/word boundaries in that priority, which keeps sub-chunks readable. Term-aware-first, recursive-splitter-second gets the best of both: semantic units where they fit, graceful degradation where they don't.
- Every sub-chunk of a split entry is prefixed with the term name ("Advanced Therapy Medicine\n...") and the splitter's chunk_size is reduced by the prefix length so the prefixed chunk still respects the cap. Without the prefix, sub-chunk 2 of a long entry has no lexical or semantic tie to its term and would miss a "what is X" query. Overlap (150 chars) carries sentence context across the cut.
- Chunk text embeds term + definition only; the [Source: ...] attribution rides in metadata, not in the embedded text. Attribution strings (journal names, directive numbers) are not what a user query matches on and would dilute the embedding.
- Metadata per chunk: term, page_start, source, chunk_index. term + page_start drive citation; chunk_index disambiguates sub-chunks of the same entry. Result: 444 chunks, median 324 chars, max capped at 1200.

## Provider change (folded back into Stage 1)

- Moved to dual LLM providers behind the factory: Gemini gemini-2.5-flash as the repo default (free tier, so a hosted Streamlit Cloud demo can run without a local model) and Ollama llama3.1 as the local/private option. Provider is chosen by one value, LLM_PROVIDER; nothing outside llm_factory knows which is active. config default is gemini.
- config now calls load_dotenv() so GOOGLE_API_KEY and LLM_PROVIDER come from a gitignored .env; .env.example ships the placeholders. The real key never lands in a committed file.
- Dependency resolution was the hard part. langchain-google-genai 4.x forces langchain-core 1.x and breaks langchain 0.3.27, so pinned 2.1.12 (core stays 0.3.x). That stack then wants protobuf>=6 while streamlit 1.41.1 needs <6; pinned protobuf 5.29.6 plus the matching grpcio/grpcio-status 1.71.0 satisfies both. pip check is clean.

## Stage 4 — Embeddings and vector store

- Embeddings computed with sentence-transformers directly and passed to Chroma as precomputed vectors, rather than handing Chroma an embedding_function. Keeps the same normalized all-MiniLM-L6-v2 vectors used everywhere (eval, ad-hoc queries) and removes a hidden model load inside Chroma.
- Cosine space with L2-normalized embeddings (hnsw:space=cosine). all-MiniLM-L6-v2 is trained for cosine similarity; distances land in a readable 0-1 range (bioavailability -> its own entry at 0.28).
- Idempotent by delete-and-recreate, not upsert. The store is derived wholesale from one static PDF; a full rebuild is the honest operation and can never leave orphaned vectors from a removed or re-split entry. Upsert only wins for incremental corpora, which this isn't.
- Integer string ids over term-based ids: two entries could in principle share a term, and sub-chunks already repeat the term; a running index guarantees uniqueness. term still travels in metadata for citation.
- Chroma over FAISS: Chroma persists to disk and stores per-vector metadata (term, page, source) in the same call, which is exactly what citation needs. FAISS is a raw similarity index — it returns row indices, and I would have to build my own persistence and a parallel metadata store and keep them in sync. At 444 vectors neither is faster in any way a user notices, so the deciding factor was less glue code, not speed. FAISS would start to win at millions of vectors or when a GPU index matters.
- Muted chromadb's telemetry logger: 0.5.x has a posthog signature bug that prints a harmless "Failed to send telemetry event" on every call, and the anonymized_telemetry=False setting does not stop it in this version.
- Verify: 444 vectors (= chunk count); the three probe queries each return the exact term as top-1 (Bioavailability 0.28, Generic Substitution 0.29, Budget Impact 0.27).

## Stage 5 — Retrieval evaluation

- 30 gold queries, 10 per category (direct / paraphrased / abbreviation), every expected_term copied verbatim from a parsed entry so the metric is an exact term match against retrieved chunk metadata. hit@k for k in {1,3,5}, computed over the top-5 once per query.
- Paraphrase queries are deliberately worded to share few content words with the definition they target (content-word overlap 0.2-0.4, not 0.9+). An earlier draft had near-verbatim paraphrases (e.g. "the patient's ability to obtain medical care and medicines" lifted straight from the Access definition), which inflated the paraphrase score into a lexical-match test rather than a semantic one.
- Results (reproducible): overall hit@1 0.70, hit@3 0.87, hit@5 0.87. Direct 1.00 across the board; paraphrased 0.40/0.70/0.70; abbreviation 0.70/0.90/0.90. Above the 0.8 hit@3 bar, so I did not force a chunking change.
- The eval is only reproducible because retrieval uses a wide HNSW beam (hnsw:search_ef=200). Chroma's default search_ef=10 was too small for this 444-vector store and silently dropped the true nearest neighbour on low-signal queries (e.g. "QALY" returned its own entry in one process and missed it entirely in another), so hit@k varied run to run. Widening the beam makes retrieval effectively exact and deterministic. Lesson: ANN defaults tuned for million-scale corpora hurt recall at small scale.
- hit@3 == hit@5: when the right term is retrievable at all it is already in the top 3; pushing k to 5 buys nothing here. That argues the failures are recall failures (term absent from the candidate set), not ranking failures.
- The remaining miss that exposes the dense-only limit is "copay": the glossary spells it "Co-payment" and a bi-encoder has no lexical bridge for the closed-form. Textbook case for hybrid lexical+dense retrieval (BM25 union dense), logged as a next step rather than built now, since dense-only clears the bar and rag_chain reuses this exact retriever.
- Kept the query set honest: dropped the prompt's "enteric coating" example because this glossary covers pharmaceutical policy and health economics, not dosage forms, and has no such entry. Verified every expected term exists before writing the set.

## Stage 6 — LLM factory and RAG chain

- Dual providers behind get_llm(): gemini (gemini-2.5-flash) as default, ollama (llama3.1) local. The rest of the code only ever calls get_llm(); the retriever, prompt and CLI are provider-agnostic. Local Ollama is free and private for development; hosted Gemini free tier makes the Streamlit Cloud demo deployable with no local model. The swap is one env value.
- Missing GOOGLE_API_KEY raises a RuntimeError that names the variable and points at .env, instead of surfacing a LangChain auth stack trace three frames deep. Unknown provider names fail fast with the supported list.
- temperature=0 on both providers. This is a grounded reference tool, not a creative one; determinism makes the refusal behaviour and citations reproducible.
- Why refusal matters here specifically: the users are pharmacists and students, and the subject is medicines. A confident wrong answer about a term, a scope, or (worse) anything a reader mistakes for clinical guidance can be acted on. The glossary defines policy and economics terms, not doses or interactions, so the safe failure is "the glossary does not define this" rather than a plausible-sounding guess. This is why the adversarial probe "maximum safe dose of paracetamol" must refuse, and why refusal is a first-class requirement, not a nicety. Verified: it refuses that probe with no invented dosing.
- Prompt enforces four things: answer only from context, refuse (glossary does not define it) when uncovered, cite term + page, plain professional register. Context blocks are labelled "Term: X (page N)" so the model has the citation material inline rather than having to infer it.
- Retriever reuses the same normalized-embedding dense query as eval_retrieval, so eval numbers actually predict production retrieval. k comes from config (5).
- Live verification ran on Gemini: 3 in-glossary answers grounded and cited (Bioavailability p15, Generic Substitution p47, Budget Impact Analysis p20), and the out-of-scope "capital of France" was refused with the glossary-does-not-define-it response. The Ollama swap was proven by construction (get_llm builds ChatOllama for llama3.1 on the env flip); the live 1-question Ollama run was deferred because the daemon is not installed, and Gemini is the default/deployed path.

## Stage 7 — Streamlit UI

- Secrets bridge runs before importing any src module: env wins, then st.secrets fills gaps. config reads GOOGLE_API_KEY at import time, so the bridge has to set os.environ first. Same code path works locally (.env) and on Streamlit Cloud (secrets manager) with no branching.
- Accessing st.secrets with no secrets file raises, so the bridge is wrapped in try/except and silently no-ops locally. That is the one place a bare except is justified: a missing secrets file is expected, not an error.
- Errors are translated to human sentences by matching the exception text (missing key, 429/quota, Ollama connection refused) instead of dumping a stack trace into the chat. Anything unrecognised still shows a short message, never a traceback.
- k is a sidebar slider (1-10, default from config) so retrieval breadth is tunable live; provider is display-only text; the rebuild action is a note, not a button, because destructive index operations do not belong in a shared UI.
- Sources render per answer in an expander: term, page, and a trimmed snippet with the leading term line stripped (it is redundant with the bold term label). History lives in st.session_state so the transcript and its sources survive reruns.

## Stage 8 — Tests, report, README

- Tests cover only the deterministic layers: de-bolding, footer detection, entry parsing on a bold-rendered fixture, chunk split threshold + term prefix, chunk metadata integrity, and factory provider selection. LLM generations are not tested; they are non-deterministic and would make the suite depend on a live backend.
- To make parsing testable, split ingest into parse_pages(pages, start_page) (pure, string-in) and parse_pdf (I/O wrapper). The fixture reproduces the 4x bold rendering with a helper so the tests exercise the real heading/de-bold path, not a simplified one. Refactor kept output byte-identical (413 entries).
- Factory tests monkeypatch config.GOOGLE_API_KEY so the missing-key path is deterministic even when a real .env is present, and they pass the provider explicitly so the suite is green with LLM_PROVIDER unset. No test touches the network.
- k = 5 (config.TOP_K): the eval shows hit@5 == hit@3, so going past 5 adds no recall, and a small k keeps the prompt context tight and cheap. 5 leaves a little headroom over the k=3 the metric rewards.
- Embedding model all-MiniLM-L6-v2 chosen for size/speed on CPU (384-dim, ~90 MB, runs locally with no service) at the cost of weaker acronym handling. The alternative I weighed was all-mpnet-base-v2 (768-dim): higher retrieval quality on paraphrases but ~3-4x slower to embed and larger, which matters for the free-tier hosted demo and the build-on-boot step. MiniLM is the single biggest quality lever, so this is the honest place to say hybrid retrieval or mpnet is the first upgrade.
- Known failure modes (what it answers badly and why): (1) bare acronyms/closed-form spellings with no lexical bridge, e.g. "copay" vs the glossary's "Co-payment"; (2) very terse paraphrases that share little vocabulary with the definition (the affordability -> Budget Impact Analysis miss); (3) the model occasionally cites a lower-ranked term when several retrieved entries are related (observed: a post-market-safety paraphrase answered from Clinical Trial's Phase IV line instead of the top-ranked Pharmacovigilance entry), because the prompt does not tell it the context is ranked; (4) anything outside the glossary's policy/economics scope, which is handled by refusal rather than a wrong answer. (1)-(3) are dense-retrieval and prompt limits; the fix directions are hybrid retrieval and a "answer from the single most relevant term" instruction.
- Refusal behaviour is prompt-enforced, not verified post-generation. Recorded as a known limitation and a next step (check that every cited term appears in the retrieved context).

## UI pass

- Upgraded the Streamlit app past the original plain-defaults constraint: example-question buttons on the empty state, a clear-chat control, provider + model shown in the sidebar, and a light custom stylesheet (accent colour, source pills). Kept it emoji-free and used Streamlit-native components where they suffice.
- Sources now dedupe by term+page keeping the best (lowest-distance) hit, and show a relevance score = max(0, 1 - cosine_distance) so the same numbers the retriever ranks on are visible to the user. k=5 often collapses to fewer unique terms because sub-chunks of one entry share a term; the dedupe makes that legible rather than showing repeats.

## Deploy

- The app builds its own index on first boot (embed_store.ensure_collection, cached with st.cache_resource) instead of committing chroma_db/. Chose build-on-boot over committing the 19 MB store because a persisted HNSW binary is tied to the Chroma build and OS and can fail to load on a different host; entries.json is committed and small, and rebuilding from it takes ~9s once per container. Locally the check is a 0.14s no-op when the index already exists.

## Post-audit hardening

- Bullet glyphs stripped in parsing (»/•/(cid:2)): the PDF uses two bullet encodings that were handled inconsistently (one kept, one dropped). Both now go, so definitions and retrieved snippets are clean. Reparse kept 413 entries and left the eval unchanged.
- Prompt now states the context is ranked most-relevant-first and tells the model to answer from the single most relevant term. Fixes a case where a post-market-safety question was answered from a lower-ranked Clinical Trial chunk instead of the top-ranked Pharmacovigilance entry.
- friendly_error moved to llm_factory and reused by both the Streamlit app and the rag_chain CLI, so `python -m src.rag_chain` with Ollama down prints one clean sentence instead of an httpx traceback. The Ollama host/port and model name in the message come from config, not string literals.
- k slider bounds (K_MIN/K_MAX) and everything else tunable now live in config; removed the unused COPYRIGHT_FOOTER constant and the unused chunk id field.

## Post-audit hardening, round 2

- Added a BM25 lexical baseline (rank_bm25) to eval_retrieval over the same chunks and queries, because the dense hit@k had nothing dumb to beat. Finding, stated honestly: BM25 ties dense overall (0.70/0.87/0.87) and edges it at hit@1 on paraphrases (0.50 vs 0.40) and abbreviations (0.80 vs 0.70); dense only clearly wins direct hit@1 (1.00 vs 0.80). The retrievers miss different queries — 3 of 30 are missed by both at hit@3, so a hybrid union would score 0.90. This reframes the upgrade path: hybrid buys a modest +0.03; the shared misses ("copay", two terse paraphrases) need acronym/alias expansion, not a better ranker.
- Hardened the refusal check: a marker match alone would count a "the glossary does not define X, but generally..." answer as a refusal. Now requires the marker plus a length cap (200 chars) — committed refusals run 51-84 chars, in-scope answers start at ~215, so the threshold has clear margin on both sides. Tested with a refuse-then-guess fixture.
- Guarded the parser's one crash path: a [Source: ...] line arriving before any heading dereferenced current=None. Skipped with a test instead of crashing.
- Corrected the Python claim from "3.11+" to "3.11-3.12": the deploy section already said the pinned wheels have no 3.14 builds, so the plus sign was overstating.
- Documented where the corpus PDF comes from (the PPRI glossary page) — entries.json kept full reproduction possible, but ingest-onward reproduction had no download pointer.
- Linked report.md and DECISIONS.md from the README layout; report.md was committed but referenced nowhere.
- Completed the RAGAS triad with a local llama3.1 (8B, Ollama) judge, closing the "harness exists but no numbers" gap: faithfulness 0.68, answer relevancy 0.91, context precision 1.00 over the 6 committed in-scope records (~76 min on CPU, one judge call timed out and was retried by the RunConfig). Chose the local judge over a paid Gemini key because it is the repo's own documented provider and reproduces for free. Interpretation recorded with the numbers: context precision 1.00 corroborates direct hit@1 = 1.00; faithfulness is judge-limited at 8B (imperfect claim extraction, paraphrase under-credited), so it is pinned to this judge and read as a regression floor, not ground truth. The non-circular anchors stay hit@k and the refusal check.

## Post-audit hardening, round 3

- Tested the path the deployed app actually runs. `answer()` was covered but the Streamlit UI only ever calls `answer_stream()`, so streaming shipped unverified: it now has the same depth of coverage (full text reassembled from the chunks, the same grounded prompt with term + page, k passed through from the sidebar slider) plus the guarantee the UI depends on — sources are returned before a single token is generated, asserted by checking the stub LLM has not started streaming when the call returns.
- `friendly_error` is user-facing in two places and had no test; added one that pins the three recognised failures (missing key, 429/quota, connection refused, with the Ollama message naming the configured host and model) and one that holds the general contract: any exception, recognised or not, comes back as a single line with no traceback.
- `rag_chain.retrieve` now calls `ensure_collection()` instead of `get_collection()`. With an unbuilt index the CLI raised a Chroma "collection does not exist", which `friendly_error` could only render as "the model backend is unavailable" — blaming the LLM for a missing index. Building it is what the caller wanted anyway, and the app already did this on boot. `ensure_collection`'s three branches (populated / missing / empty) are now tested with a stub collection rather than first meeting a fresh deploy in production.
- Added a GitHub Actions run of the suite on push and PR (Python 3.12, the top of the supported range). It also re-runs `python -m src.chunking` and greps for the 444 / 324 / 23 figures the README and report.md publish, so the docs cannot drift away from the pipeline silently — the failure mode this log has hit before.
- Corrected the judge-call estimate in eval_ragas.py from ~18 to ~40: context precision alone is one call per retrieved context, which at 6 records and k=5 is 30 before faithfulness and relevancy. The README already said ~40; the code comment was the wrong one.
