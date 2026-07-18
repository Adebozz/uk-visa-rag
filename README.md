# UK Visa & Immigration RAG Assistant

Production-grade RAG over official gov.uk visa guidance: hybrid retrieval
(BM25 + dense embeddings + reciprocal rank fusion), Claude-powered answers
with enforced citations, hallucination guardrails, and an evaluation suite.

> Portfolio project 2 of 5 — phase 1 (ingestion → retrieval → cited answers).
> Later phases add the eval harness, FastAPI service, chat UI, and deployment.

## Architecture

```
gov.uk Content API ──> ingest.py (30 pages, heading-aware chunking)
                            │
                            ▼
                     data/chunks.parquet ──> embed.py (BGE-small, local, free)
                            │                      │
                            ▼                      ▼
                      BM25 (keyword)      dense index (numpy)
                            └────────┬─────────────┘
                                     ▼
                      Reciprocal Rank Fusion (top-6)
                                     ▼
                   answer.py ── Claude + citation rules
                                     │
                     refuses when retrieval confidence is low;
                     every claim carries a [n] source citation
```

**Design decisions**
- **Hybrid retrieval**: dense embeddings understand paraphrase ("cost" ≈ "fee");
  BM25 nails exact terms (£769, form names). RRF fuses both without score calibration.
- **Heading-aware chunking**: chunks never span topics; each carries
  page → section → heading provenance, making citations verifiable.
- **No vector DB (yet)**: at hundreds of chunks, exact numpy search is faster
  and simpler than Qdrant/pgvector; the retriever interface is swappable.
- **Guardrails over vibes**: low retrieval confidence → refusal, not a guess.
  Citations are validated against the actual source list.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
make install
cp .env.example .env        # add your ANTHROPIC_API_KEY

make ingest                 # fetch 30 gov.uk pages -> chunked corpus
make index                  # embed chunks locally (downloads 133MB model once)
make ask Q="How much does a Skilled Worker visa cost?"
make test
```

## Roadmap

- [x] Phase 1 — ingestion, hybrid retrieval, cited answers with guardrails
- [ ] Phase 2 — evaluation suite: 100+ QA pairs, retrieval + faithfulness metrics
- [ ] Phase 3 — FastAPI service + Streamlit chat UI
- [ ] Phase 4 — deploy + retrieval experiments write-up (chunking/fusion ablations)
