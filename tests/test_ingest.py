from src import ingest


def bold(s):
    # Reproduce pdfplumber's bold rendering: each glyph 4x, spaces single.
    return "".join(c if c == " " else c * 4 for c in s)


def test_debold_recovers_heading():
    assert ingest.debold(bold("ABC Analysis")) == "ABC Analysis"
    assert ingest.debold(bold("Access (Accessibility)")) == "Access (Accessibility)"


def test_debold_leaves_body_and_lone_runs():
    assert ingest.debold("class A items, 10000 units") == "class A items, 10000 units"


def test_is_footer():
    assert ingest._is_footer("5")
    assert ingest._is_footer("© WHO Collaborating Centre, GÖG, Glossary 2016")
    assert not ingest._is_footer("Method by which medicines are divided.")


def test_is_heading_rejects_partial_bold_and_colon():
    assert ingest._is_heading(bold("ABC Analysis"))
    assert not ingest._is_heading(bold("Gene therapy medicine") + ": a product obtained")
    assert not ingest._is_heading(bold("Gene therapy medicine: a product"))
    assert not ingest._is_heading("plain definition text with no bold")


def _fixture_page():
    return "\n".join(
        [
            bold("ABC Analysis"),
            "Method by which medicines are divided.",
            "More detail in a second paragraph.",
            "[Source: Quick et al. 1997]",
            bold("Access"),
            "The patient's ability to obtain care.",
            "[Source: WHO Glossary",
            "continued on a second line]",
            "5",
            "© WHO Collaborating Centre, GÖG, Glossary 2016",
        ]
    )


def test_parse_pages_two_entries():
    entries = ingest.parse_pages([_fixture_page()], start_page=9)
    assert len(entries) == 2

    abc = entries[0]
    assert abc["term"] == "ABC Analysis"
    assert abc["page_start"] == 9
    assert "Method by which" in abc["definition_text"]
    assert "second paragraph" in abc["definition_text"]
    assert abc["sources"] == ["Quick et al. 1997"]

    access = entries[1]
    assert access["term"] == "Access"
    assert access["sources"] == ["WHO Glossary continued on a second line"]


def test_parse_pages_strips_footers():
    for entry in ingest.parse_pages([_fixture_page()], start_page=9):
        assert "Collaborating Centre" not in entry["definition_text"]
        assert "Glossary 2016" not in entry["definition_text"]


def test_parse_pages_stops_at_back_matter():
    back = "List of references and data sources used\nAdamski J. some reference"
    entries = ingest.parse_pages([_fixture_page(), back], start_page=9)
    assert len(entries) == 2
    assert all("Adamski" not in e["definition_text"] for e in entries)


def test_parse_pages_multiple_sources():
    page = "\n".join(
        [
            bold("Adverse Reaction"),
            "A harmful and unintended response to a medicine.",
            "[Source: WHO 2002]",
            "[Source: EU Directive 2001/83/EC]",
        ]
    )
    entries = ingest.parse_pages([page], start_page=9)
    assert len(entries) == 1
    assert entries[0]["sources"] == ["WHO 2002", "EU Directive 2001/83/EC"]


def test_parse_pages_source_before_any_heading():
    # A stray [Source: ...] line arriving before the first heading (e.g. bleed-over
    # from an excluded page) has no entry to attach to and must be skipped, not crash.
    page = "\n".join(
        [
            "[Source: orphaned attribution]",
            bold("ABC Analysis"),
            "Method by which medicines are divided.",
            "[Source: Quick et al. 1997]",
        ]
    )
    entries = ingest.parse_pages([page], start_page=9)
    assert len(entries) == 1
    assert entries[0]["sources"] == ["Quick et al. 1997"]


def test_parse_pages_keeps_cross_reference():
    page = "\n".join(
        [
            bold("Co-insurance"),
            "Cost-sharing as a set proportion of the cost of a service.",
            "See also: out-of pocket payments",
            "[Source: PPRI Glossary]",
        ]
    )
    entries = ingest.parse_pages([page], start_page=9)
    assert "See also: out-of pocket payments" in entries[0]["definition_text"]
    assert entries[0]["sources"] == ["PPRI Glossary"]
