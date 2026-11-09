import numpy as np
import pandas as pd

from visa_rag.answer import answer_question
from visa_rag.config import RetrievalConfig
from visa_rag.retrieval import HybridRetriever, RetrievedChunk


class FakeEmbedder:
    def encode(self, texts, normalize_embeddings=True, **_):
        return np.ones((len(texts), 4)) / 2


class FakeReranker:
    """Scores a passage by whether it contains the favoured word."""

    def __init__(self, favoured):
        self.favoured = favoured
        self.seen = 0

    def predict(self, pairs, **_):
        self.seen = len(pairs)
        return [1.0 if self.favoured in passage else 0.0 for _, passage in pairs]


def _retriever(reranker=None, **cfg_kwargs):
    rows = [
        ("a", "The visa fee is listed here."),
        ("b", "The visa fee has a surcharge."),
        ("c", "The visa fee has a refund."),
    ]
    chunks = pd.DataFrame(
        [dict(chunk_id=i, text=t, doc_title="Doc", section="Costs", heading="", url="https://x",
              updated_at="2026-01-01") for i, t in rows]
    )
    cfg = RetrievalConfig(top_k=2, candidates_k=3, rrf_k=60, **cfg_kwargs)
    return HybridRetriever(chunks, np.ones((3, 4)) / 2, FakeEmbedder(), cfg, query_prefix="",
                           reranker=reranker)


def test_reranker_changes_the_order():
    plain = _retriever().search("visa fee surcharge")
    assert plain[0].chunk_id != "c"
    reranked = _retriever(FakeReranker("refund")).search("visa fee surcharge")
    assert reranked[0].chunk_id == "c"


def test_only_the_configured_number_of_candidates_is_scored():
    reranker = FakeReranker("refund")
    _retriever(reranker, rerank_candidates=2).search("visa fee")
    assert reranker.seen == 2


def test_confidence_stays_the_best_fused_score_after_reranking():
    plain = _retriever().search("visa fee refund")
    reranked = _retriever(FakeReranker("refund")).search("visa fee refund")
    assert reranked[0].confidence == plain[0].score
    assert reranked[1].confidence == plain[0].score


class StubRetriever:
    def __init__(self, score, confidence):
        self.chunk = RetrievedChunk(
            chunk_id="a", text="The fee is £769.", doc_title="Doc", section="Costs", heading="",
            url="https://x", updated_at="2026-01-01", score=score, confidence=confidence,
        )

    def search(self, _):
        return [self.chunk]


class FakeClient:
    class messages:  # noqa: N801 - mirrors the Anthropic client's shape
        @staticmethod
        def create(**_):
            class Usage:
                input_tokens = 1
                output_tokens = 1

            class Block:
                text = "The fee is £769 [1]."

            class Response:
                content = [Block()]
                usage = Usage()

            return Response()


def test_low_ranked_chunk_moved_to_top_does_not_trigger_a_refusal():
    ans = answer_question("How much?", StubRetriever(score=0.01, confidence=0.03),
                          min_retrieval_score=0.02, client=FakeClient())
    assert ans.refused is False


def test_low_confidence_still_refuses():
    ans = answer_question("How much?", StubRetriever(score=0.01, confidence=0.01),
                          min_retrieval_score=0.02, client=FakeClient())
    assert ans.refused is True
