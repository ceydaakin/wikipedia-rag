# Design Choices

This file explains the meaningful technical decisions in the codebase,
so a reader can tell intentional choices apart from accidents.

## 1. Vector store — Option B (single store with metadata)

**Choice:** one Chroma collection, every chunk tagged with
`entity_type ∈ {person, place}` and `entity_title`.

**Why:**
- Comparison queries — *"Compare Albert Einstein and the Eiffel Tower"*,
  *"Which person is associated with electricity?"* — need to search
  across both types in one call. Two separate stores would force us to
  union the results client-side and re-sort by similarity, which is
  exactly what a single store with a metadata filter already does.
- Chroma's `where={"entity_type": "person"}` filter is essentially free
  (it filters during HNSW traversal), so we get Option A's behavior on
  type-specific queries for no extra cost.
- One index = one set of HNSW parameters to tune, one place to back up,
  one rebuild path.

**Tradeoff accepted:**
- Slightly noisier search when the classifier returns `both` and the
  embedding for the query is ambiguous between a person and a place.
  In practice the comparison-aware retriever handles this by retrieving
  per-entity batches when the query mentions specific named entities.

## 2. Chunking — section-aware sliding window

**Choice:** primary split on Wikipedia section headers (`== Heading ==`),
then a 600-character sliding window with 100-character overlap inside
each section. Minimum chunk size 80 characters.

**Why:**
- Section boundaries on Wikipedia are semantic ("Early life", "Career",
  "Death and legacy") so they make natural retrieval units.
- 600 chars ≈ 150 tokens, which is small enough that 5–10 chunks fit
  comfortably in `llama3.2:3b`'s 4–8k token context with room for the
  prompt and the answer.
- The 100-char overlap prevents losing context for a fact that straddles
  a window boundary.
- Small min-chars filter drops noisy single-line headers and one-line
  sections that hurt retrieval more than they help.

**Tradeoff accepted:**
- A naive recursive-character splitter would produce slightly more
  uniform chunks, but at the cost of cutting across section topics.

## 3. Query classifier — rule-based

**Choice:** rule-based router using (a) substring match against known
entity titles, (b) last-name match for multi-word people, (c) keyword
hints (`where`, `who`, `compared`, ...) as a fallback.

**Why:**
- Rule-based / keyword-based routing is sufficient at this scale.
- Latency is ~0ms vs. ~hundreds-of-ms for an LLM-based router.
- The space of entities is closed and small (40 names) so substring
  matching is reliable and easy to debug.
- Comparison detection only triggers when 2+ specific entities are
  named *and* a comparison keyword is present, avoiding false positives.

**Tradeoff accepted:**
- Misses paraphrases like "the Italian Renaissance painter who painted
  the Mona Lisa". Vector search alone still recovers most of these via
  the `both` fallback; the impact is bounded for a 40-entity corpus.

## 4. Embeddings — Ollama's `nomic-embed-text`

**Choice:** 768-dim, runs through the same Ollama daemon as the LLM.

**Why:**
- One runtime to install (Ollama) for both embedding and generation.
- `nomic-embed-text` is a strong general-purpose retrieval model and
  pairs well with Llama-class generators.
- Calling it via raw HTTP keeps us on stdlib and avoids another
  dependency.

**Tradeoff accepted:**
- One HTTP call per chunk during ingestion. For 40 entities × ~30 chunks
  ≈ 1200 calls. On a laptop this takes a few minutes, which is fine for
  a one-time setup. A production version would batch.

## 5. LLM — `llama3.2:3b` via Ollama

**Choice:** 3B-parameter Llama, temperature 0.1, max 512 tokens.

**Why:**
- Fast on consumer hardware (Apple Silicon).
- Strong-enough reasoning for grounded factoid Q&A. Hallucination is
  controlled via the prompt + low temperature, not via model size.
- Trivially swappable in `config.py` to `phi3` or `mistral` if you
  want to compare.

**Tradeoff accepted:**
- Comparison answers from a 3B model are sometimes shallow. At this
  scope this is acceptable; the recommendation file calls out
  cross-encoder rerank + larger model as the production fix.

## 6. Catalog DB — SQLite

**Choice:** stdlib `sqlite3`, single file at `data/catalog.sqlite`.

**Why:**
- Stdlib `sqlite3` — zero deps to install.
- Storing the cleaned raw text alongside metadata lets us re-chunk
  without re-fetching from Wikipedia.
- A single file is trivial to back up, inspect, and reset.

## 7. UI — Streamlit primary, CLI fallback

**Choice:** ship both.

**Why:**
- Streamlit gives a nice chat UI with retrieval-context expanders
  for free.
- CLI exists so the system can be verified end-to-end without
  installing Streamlit (e.g., on a server with no browser).
- Both call the same `src/pipeline.py`, so behavior is identical.

## 8. "Language native" interpretation

We favored language-native functionality over fully-featured
libraries. Concretely we wrote from scratch:

- The Wikipedia text cleaner.
- The section-aware chunker.
- The query classifier.
- The Ollama HTTP client (embedding + generation, stdlib `urllib`).
- The prompt assembly.

We use third-party libraries only for **infrastructure** that would be
inappropriate to reimplement: `chromadb` (vector DB), `wikipedia-api`
(MediaWiki extraction), and `streamlit` (UI). No LangChain, no LlamaIndex,
no sentence-transformers — embeddings come straight from Ollama.
