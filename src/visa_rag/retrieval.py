"""Hybrid retrieval: BM25 (keyword) + dense embeddings, fused with RRF.

Why hybrid: dense embeddings understand paraphrase ("cost" ~ "fee") but can
miss exact terms (form names like "SET(O)", fee amounts); BM25 nails exact
terms but not meaning. Reciprocal Rank Fusion combines both rankings without
needing score calibration:

    RRF(d) = sum over retrievers of 1 / (k + rank_r(d))

At this corpus size (hundreds of chunks) exact numpy search beats a vector
DB for both speed and simplicity; the Retriever interface is swappable for
Qdrant/pgvector when the corpus grows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from visa_rag.config import RetrievalConfig

_TOKEN_RE = re.compile(r"[a-z0-9£]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    doc_title: str
    section: str
    heading: str
    url: str
    updated_at: str
    score: float
    confidence: float = 0.0   # best fused score for the query, whatever the final order


class HybridRetriever:
    def __init__(
        self,
        chunks: pd.DataFrame,
        embeddings: np.ndarray,
        embed_model,               # SentenceTransformer (injected: easy to fake in tests)
        cfg: RetrievalConfig,
        query_prefix: str = "Represent this sentence for searching relevant passages: ",
        reranker=None,             # CrossEncoder (injected: easy to fake in tests)
    ):
        from rank_bm25 import BM25Okapi

        assert len(chunks) == len(embeddings), "chunks/embeddings misaligned"
        self.chunks = chunks.reset_index(drop=True)
        self.embeddings = embeddings
        self.model = embed_model
        self.cfg = cfg
        self.query_prefix = query_prefix  # BGE models need this for queries
        self.reranker = reranker
        corpus_tokens = [
            tokenize(f"{r.doc_title} {r.section} {r.text}")
            for r in self.chunks.itertuples()
        ]
        self.bm25 = BM25Okapi(corpus_tokens)

    # --- individual rankers (return indices, best first) ---

    def _dense_ranking(self, query: str, k: int) -> list[int]:
        q = self.model.encode([self.query_prefix + query], normalize_embeddings=True)[0]
        sims = self.embeddings @ q
        return list(np.argsort(-sims)[:k])

    def _bm25_ranking(self, query: str, k: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return list(np.argsort(-scores)[:k])

    # --- reranking ---

    def _rerank(
        self, query: str, candidates: list[tuple[int, float]]
    ) -> list[tuple[int, float]]:
        """Reorder fused candidates with a cross-encoder, keeping their RRF scores."""
        pairs = []
        for idx, _ in candidates:
            row = self.chunks.iloc[idx]
            pairs.append((query, f"{row.doc_title} - {row.section}\n{row.text}"))
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        order = np.argsort(-np.asarray(scores), kind="stable")
        return [candidates[i] for i in order]

    # --- fusion ---

    def search(self, query: str) -> list[RetrievedChunk]:
        k = self.cfg.candidates_k
        rankings = [self._dense_ranking(query, k), self._bm25_ranking(query, k)]
        rrf: dict[int, float] = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self.cfg.rrf_k + rank + 1)

        fused = sorted(rrf.items(), key=lambda kv: -kv[1])
        confidence = fused[0][1] if fused else 0.0
        if self.reranker is not None:
            fused = self._rerank(query, fused[: self.cfg.rerank_candidates])

        top = fused[: self.cfg.top_k]
        return [
            RetrievedChunk(
                chunk_id=row.chunk_id, text=row.text, doc_title=row.doc_title,
                section=row.section, heading=row.heading, url=row.url,
                updated_at=row.updated_at, score=score, confidence=confidence,
            )
            for idx, score in top
            for row in [self.chunks.iloc[idx]]
        ]


def load_retriever(config_path: str = "configs/config.yaml") -> HybridRetriever:
    """Convenience constructor used by the API/CLI (loads real model + data)."""
    from sentence_transformers import SentenceTransformer

    from visa_rag.config import load_config

    cfg = load_config(config_path)
    chunks = pd.read_parquet(cfg.ingestion.chunks_path)
    embeddings = np.load(cfg.embedding.embeddings_path)
    model = SentenceTransformer(cfg.embedding.model_name)
    return HybridRetriever(
        chunks, embeddings, model, cfg.retrieval,
        reranker=load_reranker(cfg.retrieval.reranker_model),
    )


def load_reranker(model_name: str | None):
    if model_name is None:
        return None
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device="cpu")  # CPU matches the deployment target
