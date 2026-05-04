"""Single source of truth for paths, model names, and tunables."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ENTITIES_FILE = DATA_DIR / "entities.json"

CATALOG_DB = DATA_DIR / "catalog.sqlite"
CHROMA_DIR = ROOT / "chroma_db"
CHROMA_COLLECTION = "wikipedia_rag"

OLLAMA_HOST = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
MIN_CHUNK_CHARS = 80

DEFAULT_TOP_K = 5
COMPARISON_TOP_K_PER_ENTITY = 4
QUERY_EXPANSION_ENABLED = True

LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 512

# Models the UI lets you choose from. Each must be `ollama pull`'ed before use.
MODEL_OPTIONS = ["llama3.2:3b", "phi3", "mistral"]

# Multi-turn memory: last N user/assistant pairs threaded into the prompt.
MAX_HISTORY_TURNS = 3

# In-memory response cache (LRU) — query+top_k+model+history is the key.
RESPONSE_CACHE_SIZE = 64

WIKIPEDIA_USER_AGENT = "Wikipedia-RAG/1.0 (educational; contact ceyda.akin@sentez.co)"
WIKIPEDIA_LANGUAGE = "en"

for d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, CHROMA_DIR):
    d.mkdir(parents=True, exist_ok=True)
