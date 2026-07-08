"""Central configuration. All paths, model names, and tunable parameters live here."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

# Paths
RAW_PDF = ROOT / "data" / "raw" / "Pharmacy_Dictionary.pdf"
PROCESSED_DIR = ROOT / "data" / "processed"
ENTRIES_JSON = PROCESSED_DIR / "entries.json"
EVAL_QUERIES_JSON = PROCESSED_DIR / "eval_queries.json"
CHROMA_DIR = ROOT / "chroma_db"

# PDF layout (1-indexed pages, verified against the file)
FRONT_MATTER_LAST_PAGE = 8  # pages 1-8 are cover/intro/background, excluded

# Chunking
MAX_CHUNK_CHARS = 1200
CHUNK_OVERLAP_CHARS = 150

# Embeddings
EMBED_MODEL = "all-MiniLM-L6-v2"

# Vector store
CHROMA_COLLECTION = "pharmacy_glossary"

# Retrieval
TOP_K = 5
K_MIN = 1
K_MAX = 10

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

# gemini (repo default: free tier, deployable)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ollama (local option)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
