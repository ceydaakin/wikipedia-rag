# Local Wikipedia RAG Assistant — BLG483E HW3

A fully-local ChatGPT-style assistant that answers questions about 20 famous
people and 20 famous places using a retrieval-augmented generation (RAG)
pipeline. **No external LLM API is used.** Embeddings, the vector store, and
the language model all run on your machine.

## Stack

| Layer        | Choice                                    |
|--------------|-------------------------------------------|
| Language     | Python 3.11+                              |
| LLM          | Ollama · `llama3.2:3b` (default)          |
| Embeddings   | Ollama · `nomic-embed-text` (768-dim)     |
| Vector store | Chroma (persistent, file-backed)          |
| Catalog DB   | SQLite (stdlib `sqlite3`)                 |
| UI           | Streamlit chat app + CLI fallback         |

The chunker, query classifier, Ollama HTTP client, and prompt assembly are
written in plain Python — no LangChain.

## Prerequisites

1. **Python 3.11+** (`python3 --version`)
2. **Ollama** — install from <https://ollama.com>
   - Confirm with `ollama --version`
   - Make sure the daemon is running (`ollama serve` in a separate terminal,
     or just launch the desktop app on macOS).

## Setup

```bash
# from the project root
bash scripts/setup.sh
```

That script will:
1. Create a `.venv` virtualenv
2. Install Python deps from `requirements.txt`
3. Pull the two required Ollama models (`llama3.2:3b`, `nomic-embed-text`)

If you'd rather do it manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

## Ingest the Wikipedia data

```bash
source .venv/bin/activate
python scripts/ingest.py
```

This fetches the 40 entities listed in `data/entities.json` (20 people + 20
places), cleans the text, splits it into section-aware overlapping chunks,
embeds every chunk with `nomic-embed-text`, and stores everything in:

- `chroma_db/` — Chroma vector store
- `data/catalog.sqlite` — SQLite raw-text catalog
- `data/raw/*.json` — cached Wikipedia text (so re-ingestion is instant)

Useful flags:

```bash
python scripts/ingest.py --refresh                  # re-fetch from Wikipedia
python scripts/ingest.py --only "Albert Einstein"   # ingest a single entity
```

## Run the chat UI

### Streamlit (recommended)

```bash
streamlit run src/ui/streamlit_app.py
```

Then open <http://localhost:8501>. The sidebar lets you change top-K, toggle
context display, see indexing status, and clear chat history.

### CLI fallback

```bash
python -m src.ui.cli
```

In-CLI commands:
- `:context` — toggle showing retrieved chunks
- `:reset` — start a fresh chat
- `:quit` — exit

## Reset the system

```bash
python scripts/reset.py     # wipes vector store + SQLite, keeps cached pages
```

## Example queries

The system is tested against the queries from the project brief:

**People**
- *Who was Albert Einstein and what is he known for?*
- *What did Marie Curie discover?*
- *Why is Nikola Tesla famous?*
- *Compare Lionel Messi and Cristiano Ronaldo*
- *What is Frida Kahlo known for?*

**Places**
- *Where is the Eiffel Tower located?*
- *Why is the Great Wall of China important?*
- *What is Machu Picchu?*
- *What was the Colosseum used for?*
- *Where is Mount Everest?*

**Mixed**
- *Which famous place is located in Turkey?*
- *Which person is associated with electricity?*
- *Compare Albert Einstein and Nikola Tesla*
- *Compare the Eiffel Tower and the Statue of Liberty*

**Failure cases (system answers "I don't know.")**
- *Who is the president of Mars?*
- *Tell me about a random unknown person John Doe*

## How retrieval works

1. **Classify** — a rule-based router looks for known entity titles, last-name
   mentions, and person/place keyword hints to decide whether the query is
   about a `person`, a `place`, or `both`.
2. **Filter + search** — the query is embedded, then Chroma is searched with a
   metadata filter that matches the routing decision.
3. **Comparison shortcut** — when 2+ specific entities are mentioned with a
   comparison keyword (`compare`, `vs`, `versus`), each entity gets its own
   small batch of chunks, then the merged set is sorted by similarity. This
   guarantees both sides are represented in context.

The full prompt instructs the LLM to use only the retrieved context and to
say `I don't know.` when it can't find an answer there.

## Project layout

```
.
├── PLAN.md                    # original implementation plan
├── README.md                  # this file
├── Product_prd.md             # PRD (rebuild-from-scratch spec)
├── recommendation.md          # production-deployment recommendations
├── config.py                  # paths, model names, chunk params
├── requirements.txt
├── scripts/
│   ├── setup.sh               # venv + deps + ollama pull
│   ├── ingest.py              # fetch -> chunk -> embed -> store
│   └── reset.py
├── src/
│   ├── ingest/                # Wikipedia fetch + cleaner
│   ├── chunking/              # section-aware sliding window
│   ├── embeddings/            # Ollama nomic-embed-text wrapper
│   ├── store/                 # SQLite catalog + Chroma vector store
│   ├── retrieval/             # query classifier + retriever
│   ├── generation/            # Ollama generate + prompts
│   ├── pipeline.py            # high-level glue
│   └── ui/                    # streamlit_app.py + cli.py
├── docs/
│   ├── design_choices.md
│   ├── demo_script.md
│   └── architecture.md
├── data/
│   └── entities.json
└── tests/                     # offline unit tests (chunking, cleaner, classifier)
```

## Tests

```bash
python -m pytest tests/ -v
```

Tests cover the cleaner, chunker, and query classifier without requiring
Ollama or network access. **15 tests, all passing.**

### End-to-end smoke test

Once Ollama is running and ingestion is complete:

```bash
python scripts/smoke_test.py
```

Runs all 14 example queries from the brief (people / places / mixed /
comparison / failure cases) and prints answers with retrieval debug
info. Use this before recording the demo video.

## Demo video

[YouTube / Loom link goes here once recorded]

## Notes & limitations

- First generation after starting Ollama is slower while the model is loaded
  into memory. Subsequent calls are fast (typically 2–6 seconds on Apple
  Silicon for `llama3.2:3b`).
- The Wikipedia-API library is the only place we don't roll our own; it
  handles MediaWiki content extraction and section parsing in a way that
  isn't worth re-implementing for a homework project.
- See `recommendation.md` for what would change to make this production-ready.
