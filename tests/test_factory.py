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
