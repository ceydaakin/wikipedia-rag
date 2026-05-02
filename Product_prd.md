# Product Requirements Document — Local Wikipedia RAG Assistant

> A self-contained PRD that an AI agent (or a human engineer) can use to
> rebuild this project from scratch. Reading this file plus the example
> queries in the README should be enough — no extra context required.

## 1. Problem & Goal

Build a local, ChatGPT-style assistant that answers natural-language
questions about a fixed set of famous people and famous places, using
**only** content sourced from Wikipedia. The system must run end-to-end on
a developer laptop with no external LLM API calls.

The product exists because:
- It demonstrates the full retrieval-augmented generation (RAG) pipeline:
  ingest → chunk → embed → store → retrieve → generate.
- It must work offline (after one-time Wikipedia fetch + model pull).
- It is a course deliverable for BLG483E HW3.

## 2. Users & Use Cases

| User                    | Need                                                           |
|-------------------------|----------------------------------------------------------------|
| Course instructor       | Run the project from `README.md` instructions only.            |
| Curious end-user        | Ask factual questions about famous people / places.            |
| Developer (us)          | Inspect retrieved chunks, swap models, reset state cleanly.    |

The instructor is the most demanding user: the project **must** be runnable
following only README instructions, with no extra guidance.

## 3. Functional Requirements

### FR-1 Ingest
- The system ingests at least 20 famous people and 20 famous places from
  Wikipedia (the brief lists 10 of each as the required minimum set).
- Required entities: Albert Einstein, Marie Curie, Leonardo da Vinci,
  William Shakespeare, Ada Lovelace, Nikola Tesla, Lionel Messi,
  Cristiano Ronaldo, Taylor Swift, Frida Kahlo, Eiffel Tower,
  Great Wall of China, Taj Mahal, Grand Canyon, Machu Picchu, Colosseum,
  Hagia Sophia, Statue of Liberty, Pyramids of Giza, Mount Everest.
- Ingestion is idempotent: re-running it without `--refresh` should be a
  no-op for already-fetched entities.
- Ingestion is resumable: failure on one entity must not block the others.
- Raw Wikipedia text + metadata is persisted in SQLite so the system has a
  full record independent of the vector store.

### FR-2 Chunk
- Documents are split into smaller chunks before embedding.
- Strategy must be deterministic and produce overlap-bearing chunks.
- Each chunk carries: parent entity title, entity type (person|place),
  section name, sequential index.

### FR-3 Embed & Store
- Embeddings are generated **locally** (no external API).
- Vectors are persisted to disk; restarts must not lose state.
- The store supports metadata filtering (at minimum by `entity_type` and
  `entity_title`).

### FR-4 Retrieve
- Given a free-text query, the system classifies it as person / place / both
  using a rule-based or keyword-based approach.
- The vector search is filtered by the classification result so that, e.g.,
  a "where is X" question doesn't surface chunks about people.
- Comparison queries that mention two specific entities must surface chunks
  for both entities.

### FR-5 Generate
- A local LLM produces the final answer from the retrieved context.
- The prompt grounds the LLM strictly to the retrieved context.
- If the context does not support an answer, the system returns
  `I don't know.` rather than hallucinating.

### FR-6 Chat interface
- A user can ask questions, see the answer, optionally see retrieved
  context, and reset the chat.
- Two interfaces ship: a Streamlit web UI (primary) and a CLI fallback.

### FR-7 Documentation
- `README.md` covers install, model pull, ingestion, run, and example
  queries — sufficient for a first-time user.
- `Product_prd.md` (this file) describes the system at the requirements
  level so it can be rebuilt without seeing the source.
- `recommendation.md` describes how to take the system to production.

## 4. Non-Functional Requirements

- **Local-only.** No outbound calls except (a) Wikipedia for one-time
  ingestion and (b) the local Ollama daemon at `localhost:11434`.
- **Reproducible.** Pinned dependency versions; deterministic chunk IDs.
- **Fast enough.** End-to-end Q&A latency under ~10s on Apple Silicon
  with `llama3.2:3b`.
- **Maintainable.** Files under ~400 lines; one concept per module;
  immutable data structures where possible.
- **Honest failure.** Adversarial / out-of-domain queries return
  `I don't know.` rather than fabricating an answer.
- **Resettable.** A single `scripts/reset.py` command wipes derived state
  without destroying cached source data.

## 5. Architecture (Reference)

```
Wikipedia ──┐
            ▼
    Ingestion (fetch + clean)
            ▼
       Chunker (section-aware,
        sliding window)
            ▼
   ┌────────┴────────┐
   ▼                 ▼
SQLite catalog   Local embedder (Ollama)
   │                 │
   │                 ▼
   │            Chroma vector store
   │                 │
   ▼                 ▼
        Retriever (classifier + filtered top-K)
                     │
                     ▼
            Local LLM (Ollama generate)
                     │
                     ▼
              Streamlit UI / CLI
```

## 6. Acceptance Criteria

The build is "done" when **all** of the following are true:

1. A clean machine can run `bash scripts/setup.sh` followed by
   `python scripts/ingest.py` and `streamlit run src/ui/streamlit_app.py`
   with no errors and no extra setup.
2. All required entities (FR-1) appear in the catalog after ingestion.
3. The 14 example queries in the README produce sensible, on-topic answers.
4. The two failure-case queries return `I don't know.`.
5. `python -m pytest tests/` passes locally with no network or Ollama
   dependency.
6. A 5-minute demo video is recorded and linked in the README.
7. `README.md`, `Product_prd.md`, `recommendation.md` are present and
   committed at the repository root.

## 7. Out of Scope

- Ingestion sources other than Wikipedia.
- Multi-user / multi-tenant deployment.
- Persistent chat history across sessions.
- Authentication.
- Re-ranking via cross-encoders or LLM-based judges.
- Live (online) updates to Wikipedia content.

## 8. Open Decisions Already Made

- **Vector store option B (single store + metadata).** Comparison queries
  need to search across types, and metadata filtering keeps the index
  unified.
- **Section-aware sliding window** with `chunk_size=600`, `overlap=100`.
- **Classifier is rule-based.** Keyword + entity-name match. No ML
  classifier or LLM-based router. This is per the brief's guidance and
  keeps latency low.
- **`llama3.2:3b` as default LLM.** Chosen for speed/quality on consumer
  hardware. `phi3` and `mistral` work as drop-in replacements via
  `config.py`.
