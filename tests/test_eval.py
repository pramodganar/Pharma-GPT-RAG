from src.eval_ragas import _refused, _refusal_rate


def test_short_refusal_counts():
    assert _refused("The glossary does not define the capital of France.")
    assert _refused("This term is not covered by the glossary.")


def test_in_scope_answer_is_not_a_refusal():
    assert not _refused(
        "Bioavailability means the rate and extent to which the active substance "
        "is absorbed from a pharmaceutical form. Term and page: Bioavailability (page 15)."
    )


def test_refuse_then_guess_is_not_a_refusal():
    # The marker alone would pass this; the length cap is what catches it.
    answer = (
        "The glossary does not define the maximum safe dose of paracetamol. "
        "However, generally speaking, the usual adult dose is 500mg to 1g every "
        "4 to 6 hours, with a maximum of 4g in 24 hours, though this depends on "
        "body weight, liver function, and other medicines being taken."
    )
    assert not _refused(answer)


def test_refusal_rate_counts_only_real_refusals():
    adversarial = [
        {"answer": "The glossary does not define this."},
        {"answer": "It is defined as " + "x" * 300},
    ]
    assert _refusal_rate(adversarial) == (1, 2)
