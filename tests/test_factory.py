import pytest

from src import config as cfg
from src import llm_factory


def test_gemini_without_key_raises_clear_error(monkeypatch):
    monkeypatch.setattr(cfg, "GOOGLE_API_KEY", None)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        llm_factory.get_llm("gemini")


def test_unknown_provider_raises(monkeypatch):
    with pytest.raises(ValueError, match="unknown LLM_PROVIDER"):
        llm_factory.get_llm("bogus")


def test_ollama_provider_builds():
    llm = llm_factory.get_llm("ollama")
    assert type(llm).__name__ == "ChatOllama"
    assert llm.model == cfg.OLLAMA_MODEL


def test_friendly_error_never_leaks_a_traceback():
    # Both the Streamlit app and the rag_chain CLI show this string to a user, so an
    # unrecognised exception must still come back as one short sentence.
    for exc in (RuntimeError("GOOGLE_API_KEY is not set"),
                RuntimeError("429 ResourceExhausted: quota"),
                ConnectionError("connection refused"),
                ValueError("something unexpected")):
        message = llm_factory.friendly_error(exc)
        assert message and "\n" not in message
        assert "Traceback" not in message


def test_friendly_error_identifies_the_known_failures(monkeypatch):
    assert "API key" in llm_factory.friendly_error(RuntimeError("google_api_key missing"))
    assert "quota" in llm_factory.friendly_error(RuntimeError("429 ResourceExhausted"))

    # The connection message names the provider that is actually configured.
    monkeypatch.setattr(cfg, "LLM_PROVIDER", "ollama")
    ollama_message = llm_factory.friendly_error(ConnectionError("connection refused"))
    assert cfg.OLLAMA_BASE_URL in ollama_message and cfg.OLLAMA_MODEL in ollama_message

    monkeypatch.setattr(cfg, "LLM_PROVIDER", "gemini")
    assert "Gemini" in llm_factory.friendly_error(ConnectionError("connection refused"))


def test_active_model_name_follows_provider():
    # The UI reads the model name through the factory, so display can never
    # disagree with what get_llm would build.
    assert llm_factory.active_model_name("gemini") == cfg.GEMINI_MODEL
    assert llm_factory.active_model_name("ollama") == cfg.OLLAMA_MODEL
    with pytest.raises(ValueError, match="unknown LLM_PROVIDER"):
        llm_factory.active_model_name("bogus")
