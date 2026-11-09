"""FastAPI service for the visa RAG assistant.

    uvicorn visa_rag.api.main:app --reload

The retriever (embedding model + indexes) loads once at startup; each /ask
request costs one retrieval (~10ms) plus one Claude call (~3-6s).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from visa_rag.config import load_config
    from visa_rag.retrieval import load_retriever

    _state["cfg"] = load_config()
    logger.info("Loading retriever (embedding model + indexes)...")
    _state["retriever"] = load_retriever()
    logger.info("Ready.")
    yield
    _state.clear()


app = FastAPI(
    title="UK Visa RAG API",
    description="Cited answers over official gov.uk visa guidance",
    version="0.1.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class SourceOut(BaseModel):
    n: int
    doc_title: str
    section: str
    url: str
    updated_at: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    sources: list[SourceOut]
    latency_s: float
    input_tokens: int
    output_tokens: int


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "retriever_loaded": "retriever" in _state}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if "retriever" not in _state:
        raise HTTPException(status_code=503, detail="Retriever not loaded")

    from visa_rag.answer import answer_question

    cfg = _state["cfg"]
    ans = answer_question(
        req.question,
        _state["retriever"],
        model=cfg.llm.model,
        max_tokens=cfg.llm.max_tokens,
        min_retrieval_score=cfg.llm.min_retrieval_score,
    )
    return AskResponse(
        answer=ans.text,
        refused=ans.refused,
        sources=[
            SourceOut(n=i, doc_title=c.doc_title, section=c.section,
                      url=c.url, updated_at=c.updated_at)
            for i, c in enumerate(ans.sources, 1)
        ],
        latency_s=round(ans.latency_s, 2),
        input_tokens=ans.input_tokens,
        output_tokens=ans.output_tokens,
    )
