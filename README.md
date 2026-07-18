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

## Evaluation

"Is the RAG any good?" is answered with numbers, not vibes:

```bash
make eval-set     # Claude generates 100 QA pairs from sampled chunks (~£0.50)
make evals        # retrieval metrics on all pairs - free, seconds
make evals-full   # + LLM-judged answer quality on 25 questions (~£1)
```

Because every generated question is traceable to its source chunk, retrieval
is scored exactly: **hit@1/3/6** (did the true chunk make the top k?) and
**MRR**. Answer quality uses Claude-as-judge for **faithfulness** (every claim
supported by sources?) and **relevance**, plus deterministic **citation
coverage** (fraction of sentences citing a source), refusal rate, latency,
and token spend. Reports are saved to `evals/results/` so retrieval
experiments (chunk sizes, fusion weights, rerankers) can be compared
regression-style.

Current results (100 retrieval questions, 25 judged answers):

| Metric | Value |
|---|---|
| Retrieval hit@1 / hit@3 / hit@6 | 0.53 / 0.75 / **0.92** |
| MRR | 0.667 |
| Faithfulness (mean /5) | **4.68** — 88% of answers scored a perfect 5 |
| Relevance (mean /5) | 4.92 |
| Citation coverage | 0.73 |
| Refusals on answerable questions | 0% |
| Latency p50 | 4.7s |

Reading: the answer layer is very strong (no hallucinated claims scored below
3), and hit@6 = 0.92 means the LLM almost always sees the right source. The
clearest improvement axis is **early precision** (hit@1 = 0.53) — a
cross-encoder reranker between fusion and the LLM is the planned experiment,
and the eval suite exists precisely to measure whether it pays for its latency.

## Roadmap

- [x] Phase 1 — ingestion, hybrid retrieval, cited answers with guardrails
- [x] Phase 2 — evaluation suite: 100+ QA pairs, retrieval + faithfulness metrics
- [ ] Phase 3 — FastAPI service + Streamlit chat UI
- [ ] Phase 4 — deploy + retrieval experiments write-up (chunking/fusion ablations)
