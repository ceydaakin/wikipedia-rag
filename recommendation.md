# Production Deployment Recommendations

The homework version runs entirely on one laptop. Below is what I would
change to take it to a real product, organized by concern. Each section
calls out the **why**, the **change**, and the **tradeoff**.

## 1. Hosting & Inference

| Concern              | Homework                       | Production                                                                         |
|----------------------|--------------------------------|------------------------------------------------------------------------------------|
| LLM runtime          | Ollama on the dev box          | vLLM / TGI on a GPU pool (A10/A100), or managed Bedrock / Vertex if "fully local" relaxes |
| Embedding runtime    | Ollama (CPU)                   | Same model on a dedicated embedder pod (batch + GPU) — embeddings dominate ingest cost |
| Concurrency          | One user at a time             | Horizontal autoscaling behind a load balancer; per-user request budget             |
| Model selection      | Hardcoded `llama3.2:3b`        | Routed at request time: small model for short factoid queries, large for comparison/multi-hop |

**Tradeoff:** GPUs cost real money. Stay on CPU-only Ollama until p95
latency or QPS forces a move; the architecture supports swapping the
inference backend without touching retrieval code.

## 2. Vector Store

- Replace local Chroma with a **managed vector DB** (pgvector on Postgres,
  Pinecone, Qdrant Cloud, or Weaviate) once the corpus exceeds ~100k chunks
  or multiple replicas need to read it.
- Keep the **single-collection + metadata-filter** design (Option B from
  this project). It scales fine to millions of vectors.
- Add **HNSW parameter tuning** (`ef_construction`, `M`) and benchmark
  recall@k on a labeled eval set before locking values in.
- Maintain a **document → chunk → vector** lineage so re-embeddings can be
  triggered surgically when the embedder model changes.

## 3. Ingestion Pipeline

- Move from "run a script on my laptop" to a **scheduled job** (Airflow,
  Prefect, or a simple Cloud Run job + Scheduler).
- Track **content hashes** per Wikipedia page; only re-embed when the page
  has actually changed.
- Add a **dead-letter queue** for entities that fail ingestion, with
  alerting and a one-click retry.
- Persist **ingestion provenance** (who ran it, what model versions,
  what chunk parameters) so retrieval results are reproducible.

## 4. Retrieval Quality

The homework uses a rule-based classifier and pure vector search. For
production, layer on:

1. **Hybrid search** — combine BM25 keyword scores with dense scores
   (Reciprocal Rank Fusion). Catches name- and number-heavy queries that
   embeddings under-rank.
2. **Cross-encoder rerank** — top-50 → cross-encoder (e.g.,
   `bge-reranker-v2`) → top-K. Big quality win, modest latency cost.
3. **Query rewriting** — let an LLM expand short queries
   ("Tesla?" → "Who was Nikola Tesla and what was he known for?") before
   embedding. Especially helps the chat-history case.
4. **Negative cache** — remember which queries returned no useful context
   so the system answers `I don't know.` faster on repeats.

## 5. Grounding & Hallucination Defense

- Add a **citation-required prompt mode**: the model must cite chunk IDs;
  any uncited claim is stripped.
- Run a **post-generation validator** that re-asks the model "is every
  factual claim in the answer supported by the provided context? Reply yes
  or no". Reject and regenerate on `no` (with backoff).
- Maintain an **adversarial query set** ("Who is the president of Mars?")
  and gate releases on the system maintaining `I don't know.` behavior.

## 6. Observability

- **Structured logs** for every request: query, classified type, retrieved
  chunk IDs + scores, generation latency, model version.
- **Metrics**: latency p50/p95/p99 per stage (embed / retrieve / generate),
  cache hit rates, "I don't know" rate, tokens generated.
- **Traces** linking the user request → retrieval call → LLM call (OTel).
- A **per-query feedback button** in the UI ("was this helpful?") feeding
  a labeled dataset for offline eval.

## 7. Evaluation

- Build a **golden Q&A set** (50–200 questions) labeled by humans with the
  correct answer and the correct source chunk(s).
- Run **Recall@K** and **Faithfulness** metrics on every PR via CI; block
  merges that regress either.
- Keep **A/B testing** lightweight: route 5% of traffic to a variant
  retriever or model and compare quality + latency before promoting.

## 8. Cost & Capacity

- **Cache** the full `(query, retrieved_chunk_ids, answer)` tuple by query
  hash with a TTL — RAG queries have a long tail and cache well.
- **Pre-compute embeddings** in batch; never embed at query time except
  for the user's query itself.
- **Quantize** the LLM (Q4_K_M or Q5_K_M GGUF) — small quality drop, big
  memory + speed gain.

## 9. Security & Privacy

- Treat user queries as PII. **Redact and tokenize** before logging if
  you're shipping to multi-user production.
- Rate limit per IP / per session.
- Sanitize retrieved chunks before they enter the prompt — Wikipedia
  content is generally safe but external corpora can contain prompt
  injection ("ignore previous instructions"). Use **delimiter tokens** and
  **a fixed system prompt** the user can't override.
- Do **not** echo retrieved content verbatim if the corpus contains
  proprietary data — switch to summary mode or restrict by user role.

## 10. UX

- **Streaming** is already implemented; keep it — perceived latency wins.
- Add **inline source links** that scroll to the supporting chunk on
  hover (Perplexity-style citations).
- Add **chat history with retrieval re-use** — short follow-ups
  ("and his wife?") need the previous turn's context to embed correctly.
- Provide an **admin dashboard** showing index status, last ingestion run,
  failure counts, and a "force re-index" button.

## 11. Deployment Topology (Recommended)

```
        ┌─────────────────────────────┐
        │       Front-end (Next.js)    │
        └────────────┬────────────────┘
                     │
                     ▼
      ┌──────────────────────────────┐
      │   API gateway (FastAPI)       │  ← auth, rate limit, telemetry
      └──┬──────────────────────┬─────┘
         │                      │
         ▼                      ▼
  ┌──────────────┐       ┌──────────────────┐
  │ Embedder pod │       │ LLM pod (vLLM)   │
  └──────┬───────┘       └────────┬─────────┘
         │                        │
         ▼                        ▼
  ┌──────────────────┐    ┌─────────────────┐
  │ Vector DB (pg-   │    │ Cache (Redis)   │
  │ vector / Qdrant) │    └─────────────────┘
  └──────────────────┘
         ▲
         │ batch
  ┌──────┴────────┐
  │ Ingestion job │  ← scheduled, content-hash-based incremental
  └───────────────┘
```

## Summary

The homework architecture is the right shape — what changes for production
is **scale, eval, and operational rigor**, not the core RAG flow. The two
biggest immediate wins on the path to production are:

1. **Cross-encoder rerank** for retrieval quality.
2. **Golden eval set + CI gating** so quality is measurable instead of
   anecdotal.

Everything else is incremental.
