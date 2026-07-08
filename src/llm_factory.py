"""Return a LangChain chat model for the configured provider.

One branch per provider; nothing else in the codebase knows which is active.
Adding a provider means adding a branch here and a config value, nothing more.
"""

from . import config as cfg


def get_llm(provider=None):
    provider = (provider or cfg.LLM_PROVIDER).lower()

    if provider == "gemini":
        if not cfg.GOOGLE_API_KEY:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Add it to .env (see .env.example) "
                "or export it, or set LLM_PROVIDER=ollama to use the local model."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=cfg.GEMINI_MODEL,
            google_api_key=cfg.GOOGLE_API_KEY,
            temperature=0,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=cfg.OLLAMA_MODEL,
            base_url=cfg.OLLAMA_BASE_URL,
            temperature=0,
        )

    raise ValueError(f"unknown LLM_PROVIDER {provider!r}; supported: gemini, ollama")


def friendly_error(exc):
    """Translate a backend exception into one human sentence, no stack trace."""
    text = str(exc)
    low = text.lower()
    if "google_api_key" in low:
        return "No Gemini API key found. Add GOOGLE_API_KEY to your .env or the app secrets."
    if "429" in text or "quota" in low or "resourceexhausted" in low:
        return "The Gemini free-tier quota is exhausted for now. Try again later, or run locally with Ollama."
    if isinstance(exc, ConnectionError) or "connection" in low or "refused" in low:
        return f"Cannot reach the local Ollama server ({cfg.OLLAMA_BASE_URL}). Start Ollama and pull {cfg.OLLAMA_MODEL}, or switch the provider to Gemini."
    return f"The model backend is unavailable: {text}"
