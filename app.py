"""Streamlit chat UI for the glossary assistant.

Run with: streamlit run app.py
"""

import os
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Pharma-GPT", layout="centered")

_SECRETS_PATHS = (
    Path.home() / ".streamlit" / "secrets.toml",
    Path.cwd() / ".streamlit" / "secrets.toml",
)


def _bridge_secrets():
    # Env (.env) wins; fall back to st.secrets so the same code runs on
    # Streamlit Community Cloud without a .env file. Only touch st.secrets when a
    # secrets file exists, otherwise Streamlit logs a spurious "no secrets" error.
    if not any(p.exists() for p in _SECRETS_PATHS):
        return
    try:
        for key in ("GOOGLE_API_KEY", "LLM_PROVIDER", "GEMINI_MODEL",
                    "OLLAMA_MODEL", "OLLAMA_BASE_URL"):
            if not os.getenv(key) and key in st.secrets:
                os.environ[key] = str(st.secrets[key])
    except Exception:
        pass


_bridge_secrets()

from src import config as cfg
from src.embed_store import ensure_collection
from src.llm_factory import active_model_name, friendly_error
from src.rag_chain import answer_stream


@st.cache_resource(show_spinner="Building the glossary index (first run only)...")
def _bootstrap_index():
    ensure_collection()
    return True

EXAMPLES = [
    "What is bioavailability?",
    "Define generic substitution",
    "What is an ATC classification?",
    "What is a budget impact analysis?",
]

# Colours are theme-aware: light defaults, brighter accent + translucent pills under
# a dark OS/browser theme. (Covers prefers-color-scheme; Streamlit's own manual theme
# toggle is not exposed to CSS, so that edge case falls back to the light palette.)
STYLE = """
<style>
:root { --pharma-accent: #0e7c86; --pharma-muted: #5c6b73; --pharma-pill: #e6f2f3; }
@media (prefers-color-scheme: dark) {
  :root { --pharma-accent: #4fd1c5; --pharma-muted: #9aa7ad;
          --pharma-pill: rgba(79,209,197,0.16); }
}
h1 { color: var(--pharma-accent); font-weight: 700; letter-spacing: -0.5px; margin-bottom: 0; }
.pharma-sub { color: var(--pharma-muted); font-size: 0.95rem; margin: 2px 0 2px 0; }
.pharma-scope { color: var(--pharma-muted); font-size: 0.8rem; margin: 0 0 10px 0; }
.src-head { margin-bottom: 2px; }
.src-term { color: var(--pharma-accent); font-weight: 600; }
.src-pill { background: var(--pharma-pill); color: var(--pharma-accent); padding: 1px 8px;
            border-radius: 10px; font-size: 0.72rem; margin-left: 6px; }
.src-rel { color: var(--pharma-muted); font-size: 0.75rem; margin-left: 8px; }
.stButton > button { border-radius: 8px; }
</style>
"""


def _unique_sources(sources):
    best = {}
    for d in sources:
        key = (d["term"], d["page"])
        if key not in best or d["distance"] < best[key]["distance"]:
            best[key] = d
    return list(best.values())


def _render_sources(sources):
    # term/page are interpolated into raw HTML; they come from the parsed glossary
    # (a trusted, static corpus), never from user input, so this is not injectable.
    uniq = _unique_sources(sources)
    with st.expander(f"Sources ({len(uniq)})"):
        for d in uniq:
            relevance = max(0.0, 1.0 - d["distance"])
            snippet = d["text"].split("\n", 1)[-1].strip()
            snippet = snippet[:300] + ("..." if len(snippet) > 300 else "")
            st.markdown(
                f"<div class='src-head'><span class='src-term'>{d['term']}</span>"
                f"<span class='src-pill'>page {d['page']}</span>"
                f"<span class='src-rel'>relevance {relevance:.2f}</span></div>",
                unsafe_allow_html=True,
            )
            st.caption(snippet)


st.markdown(STYLE, unsafe_allow_html=True)
st.title("Pharma-GPT")

_bootstrap_index()
st.markdown(
    "<div class='pharma-sub'>Answers about pharmaceutical terms, grounded in the "
    "WHO/PPRI Glossary of Pharmaceutical Terms (2016).</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='pharma-scope'>Scope: pharmaceutical policy &amp; health economics "
    "(pricing, HTA, ATC/DDD, pharmacovigilance). Not clinical or dosing advice.</div>",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Settings")
    k = st.slider("Chunks retrieved (k)", min_value=cfg.K_MIN, max_value=cfg.K_MAX, value=cfg.TOP_K)
    st.text(f"Provider: {cfg.LLM_PROVIDER}")
    st.text(f"Model: {active_model_name()}")
    st.divider()
    if st.button("Clear chat", use_container_width=True, disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("In-scope examples")
    for i, q in enumerate(EXAMPLES):
        if st.button(q, key=f"side_ex_{i}", use_container_width=True):
            st.session_state.pending = q
            st.rerun()
    st.divider()
    st.caption("To rebuild the index, run `python -m src.embed_store` from the project root.")

typed = st.chat_input("Ask about a pharmaceutical term")
queued = st.session_state.pop("pending", None)
prompt = typed or queued

# Empty state: a short prompt plus one-click example questions. A click queues the
# question and reruns, so the examples do not linger above the conversation.
if not st.session_state.messages and not prompt:
    st.write("Ask a question, or start with an example:")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLES):
        if cols[i % 2].button(q, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending = q
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("error"):
            st.error(msg["content"])
        else:
            st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            _render_sources(msg["sources"])

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the glossary..."):
                sources, stream = answer_stream(prompt, k=k)
            text = st.write_stream(stream)
            _render_sources(sources)
            st.session_state.messages.append(
                {"role": "assistant", "content": text, "sources": sources}
            )
        except Exception as exc:
            message = friendly_error(exc)
            st.error(message)
            st.session_state.messages.append(
                {"role": "assistant", "content": message, "sources": [], "error": True}
            )
