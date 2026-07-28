"""Generation evaluation with RAGAS: faithfulness, answer relevance, context precision.

This is the generation-quality layer on top of eval_retrieval.py (which scores
retrieval only). It runs the real pipeline, so it needs a built index and a working
LLM provider. RAGAS itself is a heavy, separately-pinned dependency; install it with
`pip install -r requirements-eval.txt` (kept out of requirements.txt to avoid
disturbing the deploy stack's protobuf/grpcio pins).

Three commands:
  python -m src.eval_ragas --generate      build the {question, answer, contexts,
                                            reference} records (needs index + LLM)
  python -m src.eval_ragas --refusal-only  refusal rate on the out-of-scope probes,
                                            from the committed records; offline, no
                                            judge calls
  python -m src.eval_ragas                 the refusal check, then RAGAS scoring
                                            (needs RAGAS + a judge LLM)

Caveat worth stating out loud: RAGAS uses an LLM to judge an LLM. Treat the numbers
as a regression proxy validated against the human gold set and the refusal check,
not as ground truth. The non-circular anchor is eval_retrieval.py's hit@k.
"""

import json
import sys

from . import config as cfg

RECORDS_JSON = cfg.PROCESSED_DIR / "ragas_records.json"

# A representative in-scope subset (scoring the full 30 through a live judge is slow
# and quota-heavy). Two per category is enough to catch a regression; widen if needed.
IN_SCOPE_TERMS = [
    "Bioavailability",
    "Pharmacovigilance",
    "Generic Substitution",
    "Budget Impact Analysis (BIA)",
    "Health Technology Assessment (HTA)",
    "Defined Daily Dose (DDD)",
]

# Out-of-scope / adversarial: the glossary is policy + health economics, so these
# must be refused, not answered. Scored as a refusal rate, not with RAGAS (a refusal
# has no claims to check for faithfulness).
ADVERSARIAL = [
    "What is the maximum safe dose of paracetamol?",
    "What is the capital of France?",
    "Which antibiotic should I take for a chest infection?",
]

REFUSAL_MARKERS = ("not define", "not covered", "no definition")

# A genuine refusal is one short sentence (the committed ones run 51-84 chars);
# in-scope answers start at ~215. The length cap catches the failure mode a marker
# alone would miss: an answer that says "the glossary does not define X" and then
# answers anyway from outside knowledge.
REFUSAL_MAX_CHARS = 200


def _refused(answer):
    low = answer.lower()
    return any(m in low for m in REFUSAL_MARKERS) and len(answer) <= REFUSAL_MAX_CHARS


def _term_to_definition():
    entries = json.load(open(cfg.ENTRIES_JSON, encoding="utf-8"))
    return {e["term"]: e["definition_text"] for e in entries}


def _question_for(term):
    # Strip the parenthetical aliases so the question reads naturally.
    base = term.split("(")[0].strip()
    return f"What is {base.lower()}?"


def generate():
    """Run the pipeline over the eval set and persist RAGAS-shaped records."""
    from .rag_chain import answer

    defs = _term_to_definition()
    records = []
    for term in IN_SCOPE_TERMS:
        q = _question_for(term)
        out = answer(q)
        records.append(
            {
                "question": q,
                "answer": out["answer"],
                "contexts": [d["text"] for d in out["sources"]],
                "reference": f"{term}\n{defs[term]}".strip(),
                "expected_term": term,
            }
        )
        print(f"generated: {q}")

    adversarial = []
    for q in ADVERSARIAL:
        out = answer(q)
        adversarial.append({"question": q, "answer": out["answer"]})
        print(f"generated (adversarial): {q}")

    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(RECORDS_JSON, "w", encoding="utf-8") as f:
        json.dump({"in_scope": records, "adversarial": adversarial}, f,
                  ensure_ascii=False, indent=2)
    print(f"\nwrote {len(records)} in-scope + {len(adversarial)} adversarial records "
          f"to {RECORDS_JSON}")


def _refusal_rate(adversarial):
    refusals = [r for r in adversarial if _refused(r["answer"])]
    return len(refusals), len(adversarial)


def score(refusal_only=False):
    """Score persisted records: the out-of-scope refusal rate (offline), then unless
    refusal_only, faithfulness / answer relevance / context precision with RAGAS."""
    if not RECORDS_JSON.exists():
        print("no records found. run `python -m src.eval_ragas --generate` first.")
        return
    data = json.load(open(RECORDS_JSON, encoding="utf-8"))

    refused, total = _refusal_rate(data["adversarial"])
    print(f"out-of-scope refusal rate: {refused}/{total}")
    for r in data["adversarial"]:
        ok = "REFUSED" if _refused(r["answer"]) else "ANSWERED (bad)"
        print(f"  [{ok}] {r['question']}")

    if refusal_only:
        return

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.run_config import RunConfig
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        print("\nRAGAS not installed. Install with `pip install -r requirements-eval.txt` "
              "to score generation quality.")
        return

    from .llm_factory import get_llm

    rows = data["in_scope"]
    ds = Dataset.from_dict(
        {
            "question": [r["question"] for r in rows],
            "answer": [r["answer"] for r in rows],
            "contexts": [r["contexts"] for r in rows],
            "reference": [r["reference"] for r in rows],
        }
    )
    judge = LangchainLLMWrapper(get_llm())
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=cfg.EMBED_MODEL))
    # Serialize judge calls and wait out backoffs. RAGAS defaults to 16 concurrent
    # workers, which instantly trips the Gemini free tier's per-minute limit and
    # returns NaN for the call-heavy metrics (faithfulness, context precision). One
    # worker with long timeouts and retries clears the per-minute limit; note the
    # separate free-tier daily cap (20 requests/day) can still exhaust before the triad
    # finishes: it costs ~40 judge calls on these records (context precision alone is
    # one call per retrieved context, 6 records x k=5) -- use a paid key or a local
    # Ollama judge to complete it.
    run_config = RunConfig(max_workers=1, timeout=600, max_retries=15, max_wait=90)
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge,
        embeddings=embeddings,
        run_config=run_config,
    )
    print("\nRAGAS (in-scope):")
    print(result)


def main():
    if "--generate" in sys.argv:
        generate()
    else:
        score(refusal_only="--refusal-only" in sys.argv)


if __name__ == "__main__":
    main()
