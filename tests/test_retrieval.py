import zlib

import numpy as np
import pandas as pd
import pytest

from visa_rag.config import RetrievalConfig
from visa_rag.retrieval import HybridRetriever, tokenize


class FakeEmbedder:
    """Deterministic 8-dim 'embeddings' from token hashes - no model download."""

    def encode(self, texts, normalize_embeddings=True, **_):
        out = []
        for t in texts:
            v = np.zeros(8)
            for tok in tokenize(t):
                v[zlib.crc32(tok.encode()) % 8] += 1.0  # hash() differs between runs
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return np.array(out)


@pytest.fixture()
def retriever():
    rows = [
        ("skilled-worker-visa::Costs::0", "The Skilled Worker visa fee is £769 for three years.",
         "Skilled Worker visa", "Costs"),
        ("student-visa::Costs::0", "The Student visa application fee is £524.",
         "Student visa", "Costs"),
        ("graduate-visa::Overview::0", "The Graduate visa lets you stay after your studies.",
         "Graduate visa", "Overview"),
    ]
    chunks = pd.DataFrame(
        [dict(chunk_id=i, text=t, doc_title=d, section=s, heading="", url="https://x",
              updated_at="2026-01-01") for i, t, d, s in rows]
    )
    fake = FakeEmbedder()
    embeddings = fake.encode((chunks["doc_title"] + " - " + chunks["section"]
                              + "\n" + chunks["text"]).tolist())
    cfg = RetrievalConfig(top_k=2, candidates_k=3, rrf_k=60)
    return HybridRetriever(chunks, embeddings, fake, cfg, query_prefix="")


def test_exact_term_match_wins(retriever):
    results = retriever.search("£524 student fee")
    assert results[0].doc_title == "Student visa"


def test_returns_top_k_with_scores(retriever):
    results = retriever.search("visa fee")
    assert len(results) == 2
    assert results[0].score >= results[1].score


def test_tokenizer_keeps_currency():
    assert "£769" in tokenize("The fee is £769.")
