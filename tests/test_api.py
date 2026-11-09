from fastapi.testclient import TestClient

from visa_rag.answer import Answer
from visa_rag.api import main as api_main
from visa_rag.retrieval import RetrievedChunk


def _fake_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="skilled-worker-visa::Costs::0",
        text="The fee is £769.",
        doc_title="Skilled Worker visa", section="Costs", heading="",
        url="https://www.gov.uk/skilled-worker-visa", updated_at="2026-01-01",
        score=0.5,
    )


def test_ask_happy_path(monkeypatch):
    api_main._state["retriever"] = object()
    from visa_rag.config import load_config
    api_main._state["cfg"] = load_config("configs/config.yaml")

    def fake_answer(question, retriever, **kw):
        return Answer(question=question, text="The fee is £769 [1].",
                      sources=[_fake_chunk()], latency_s=0.1,
                      input_tokens=100, output_tokens=20)

    monkeypatch.setattr("visa_rag.answer.answer_question", fake_answer)
    # No context manager: entering it runs lifespan, which loads the real
    # retriever. Direct calls skip startup and use the state we injected.
    client = TestClient(api_main.app, raise_server_exceptions=True)
    r = client.post("/ask", json={"question": "How much is it?"})
    assert r.status_code == 200
    body = r.json()
    assert body["refused"] is False
    assert body["sources"][0]["doc_title"] == "Skilled Worker visa"
    assert "[1]" in body["answer"]


def test_ask_validates_question_length():
    api_main._state["retriever"] = object()
    client = TestClient(api_main.app)
    assert client.post("/ask", json={"question": "hi"}).status_code == 422


def test_health_reports_retriever():
    api_main._state.pop("retriever", None)
    client = TestClient(api_main.app)
    body = client.get("/health").json()
    assert body["retriever_loaded"] is False
