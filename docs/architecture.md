# Architecture

```
                ┌─────────────────────┐
                │  Wikipedia API      │
                │  (one-time fetch)   │
                └──────────┬──────────┘
                           │
                           ▼
   ┌────────────────────────────────────────────┐
   │  Ingestion (scripts/ingest.py)             │
   │  src/ingest/ → src/chunking/ →             │
   │  src/embeddings/ → src/store/              │
   └──────────┬─────────────────────────────────┘
              │
              ▼
  ┌──────────────────────┐    ┌─────────────────────┐
  │  SQLite              │    │  Chroma             │
  │  (data/catalog.sqlite│    │  (chroma_db/)       │
  │   raw text + meta)   │    │  vectors + metadata │
  └──────────────────────┘    └──────────┬──────────┘
                                         │
                                         ▼
                          ┌─────────────────────────────┐
                          │  Retriever                  │
                          │  src/retrieval/classifier   │
                          │  src/retrieval/retriever    │
                          └──────────┬──────────────────┘
                                     │
                                     ▼
                          ┌─────────────────────────────┐
                          │  Generator                  │
                          │  src/generation/prompts     │
                          │  src/generation/llm  (Ollama)│
                          └──────────┬──────────────────┘
                                     │
                                     ▼
                       ┌──────────────────────────────────┐
                       │  src/pipeline.py                 │
                       │      ▲              ▲            │
                       │      │              │            │
                       │ src/ui/streamlit  src/ui/cli     │
                       └──────────────────────────────────┘
```

## Module responsibilities

| Module                      | Responsibility                                                  |
|-----------------------------|-----------------------------------------------------------------|
| `src/ingest/wikipedia.py`   | Fetch + cache raw Wikipedia pages                               |
| `src/ingest/cleaner.py`     | Strip references, trailing meta sections, citation brackets     |
| `src/chunking/splitter.py`  | Section-aware sliding window                                    |
| `src/embeddings/embedder.py`| Ollama `nomic-embed-text` HTTP client                           |
| `src/store/catalog.py`      | SQLite raw-text catalog                                         |
| `src/store/vector_store.py` | Chroma persistent vector store                                  |
| `src/retrieval/classifier.py`| Rule-based query type detection                                |
| `src/retrieval/retriever.py`| Top-K vector search + comparison-aware retrieval                |
| `src/generation/prompts.py` | Prompt template with grounding + "I don't know" rule            |
| `src/generation/llm.py`     | Ollama `generate` HTTP client (stream + non-stream)             |
| `src/pipeline.py`           | End-to-end glue used by both UIs                                |
| `src/ui/streamlit_app.py`   | Streamlit chat UI                                               |
| `src/ui/cli.py`             | Terminal chat fallback                                          |

## Data flow at query time

1. User submits a question in the UI.
2. `src/pipeline.stream_answer()` is invoked.
3. `classifier.classify()` looks at the query + the catalog entity list.
4. `embedder.embed_text()` produces a 768-dim vector.
5. `vector_store.query()` runs HNSW with a metadata filter.
6. Comparison queries also run per-entity searches and merge.
7. `prompts.build_prompt()` assembles the grounded prompt.
8. `llm.generate_stream()` streams tokens from Ollama back to the UI.
