# UK Visa & Immigration RAG Assistant

RAG over official gov.uk visa guidance: hybrid retrieval (BM25 + dense
embeddings + reciprocal rank fusion), cross-encoder reranking, Claude-powered
answers with enforced citations, hallucination guardrails, and an evaluation
suite. Includes a FastAPI service and a Streamlit chat UI.

> Not legal advice. This is a portfolio project and it is not affiliated with gov.uk.

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
                      Reciprocal Rank Fusion (top-10)
                                     ▼
                      Cross-encoder reranker (top-6)
                                     ▼
                   answer.py ── Claude + citation rules
                                     │
                     refuses when retrieval confidence is low;
                     every claim carries a [n] source citation
```

**Design decisions**
- **Hybrid retrieval**: dense embeddings understand paraphrase ("cost" ≈ "fee");
  BM25 nails exact terms (£769, form names). RRF fuses both without score calibration.
- **Reranking changes order, not confidence**: the refusal gate still uses the
  best fused score, so moving a weakly matched chunk to the top can't cause a
  false refusal, and cross-encoder scores never need their own threshold.
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
make sweep        # compare reranker models and candidate counts - free
```

Because every generated question is traceable to its source chunk, retrieval
is scored exactly: **hit@1/3/6** (did the true chunk make the top k?) and
**MRR**. Answer quality uses Claude-as-judge for **faithfulness** (every claim
supported by sources?) and **relevance**, plus deterministic **citation
coverage** (fraction of sentences citing a source), refusal rate, latency,
and token spend. Reports are saved to `evals/results/` so retrieval
experiments (chunk sizes, fusion weights, rerankers) can be compared
regression-style.

### Reranker experiment

Plain RRF put the right chunk first only 53% of the time, so a cross-encoder
now reorders the fused results. Retrieval on the 100-question set, on CPU:

| Setup | hit@1 | hit@3 | hit@6 | MRR | Search p50 | Top result gained / lost |
|---|---|---|---|---|---|---|
| RRF only | 0.53 | 0.75 | 0.92 | 0.667 | 11 ms | - |
| **MiniLM-L-6, 10 candidates (chosen)** | **0.74** | **0.95** | **0.97** | **0.843** | **106 ms** | **26 / 5** |
| MiniLM-L-6, 20 candidates | 0.74 | 0.93 | 0.98 | 0.837 | 189 ms | 26 / 5 |
| BGE reranker base, 10 candidates | 0.71 | 0.89 | 0.98 | 0.813 | 465 ms | 24 / 6 |
| BGE reranker base, 20 candidates | 0.70 | 0.89 | 0.96 | 0.801 | 916 ms | 23 / 6 |

Reading: the reranker adds about 95 ms to an answer that takes several seconds,
and the smaller model beat the larger one on this data. Scoring more candidates
cost latency without improving hit@1.

Limits: Claude wrote the questions from individual chunks, and each counts only
one chunk as correct, even if another chunk also answers it. The settings were
chosen on these same 100 questions, so treat small differences between rows as
noise. There is no separate held-out set yet.

### Answer quality

Measured before the reranker was added (25 judged answers), and not yet re-run:

| Metric | Value |
|---|---|
| Faithfulness (mean /5) | **4.68** — 88% of answers scored a perfect 5 |
| Relevance (mean /5) | 4.92 |
| Citation coverage | 0.73 |
| Refusals on answerable questions | 0% |
| Latency p50 | 4.7s |

Answers are judged by Claude, so these scores are a useful signal rather than
an independent measurement.

## Roadmap

- [x] Phase 1 — ingestion, hybrid retrieval, cited answers with guardrails
- [x] Phase 2 — evaluation suite: 100+ QA pairs, retrieval + faithfulness metrics
- [x] Phase 3 — FastAPI service (`make serve`) + Streamlit chat UI (`make chat`)
- [x] Reranker experiment — cross-encoder reranking, measured against plain RRF
- [ ] Re-run answer-quality evaluation with the reranker
- [ ] Phase 4 — deploy + further retrieval experiments (chunking/fusion ablations)
