"""Compare reranker settings against plain RRF on the eval set.

    python -m visa_rag.evals.sweep

Retrieval only, so no API calls. Each row also counts the questions whose
top result got better or worse than the baseline, because an average can
hide a reranker that helps some questions and hurts as many others.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from visa_rag.config import load_config
from visa_rag.evals.run import load_eval_set, search_all, summarise_retrieval
from visa_rag.retrieval import HybridRetriever, load_reranker

EXPERIMENTS = [
    ("cross-encoder/ms-marco-MiniLM-L-6-v2", 10),
    ("cross-encoder/ms-marco-MiniLM-L-6-v2", 20),
    ("cross-encoder/ms-marco-MiniLM-L-6-v2", 30),
    ("cross-encoder/ms-marco-MiniLM-L-6-v2", 40),
    ("BAAI/bge-reranker-base", 10),
    ("BAAI/bge-reranker-base", 20),
    ("BAAI/bge-reranker-base", 30),
    ("BAAI/bge-reranker-base", 40),
]


def main() -> None:
    cfg = load_config()
    pairs = load_eval_set("data/eval_set.jsonl")
    chunks = pd.read_parquet(cfg.ingestion.chunks_path)
    embeddings = np.load(cfg.embedding.embeddings_path)
    model = SentenceTransformer(cfg.embedding.model_name)
    sources = [p["source_chunk_id"] for p in pairs]

    def run(reranker, candidates: int) -> tuple[dict, list[list[str]]]:
        retrieval = replace(cfg.retrieval, rerank_candidates=candidates)
        retriever = HybridRetriever(chunks, embeddings, model, retrieval, reranker=reranker)
        results, latencies = search_all(pairs, retriever)
        return summarise_retrieval(pairs, results, latencies), results

    baseline, baseline_results = run(None, 0)
    rows = [{"reranker": None, "candidates": None, **baseline}]
    reranker_cache: dict[str, object] = {}

    for name, candidates in EXPERIMENTS:
        reranker = reranker_cache.setdefault(name, load_reranker(name))
        summary, results = run(reranker, candidates)
        top = [r[:1] == [s] for r, s in zip(results, sources, strict=True)]
        base_top = [r[:1] == [s] for r, s in zip(baseline_results, sources, strict=True)]
        rows.append({
            "reranker": name, "candidates": candidates, **summary,
            "hit@1_gained": sum(t and not b for t, b in zip(top, base_top, strict=True)),
            "hit@1_lost": sum(b and not t for t, b in zip(top, base_top, strict=True)),
        })

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(":", "-")
    out = Path("evals/results") / f"rerank-sweep-{stamp}.json"
    out.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
