"""Answer engine: retrieval -> Claude -> cited, guarded answer.

    python -m visa_rag.answer "How much does a Skilled Worker visa cost?"

Guardrails:
- If retrieval confidence is too low, refuse instead of letting the LLM guess.
- The prompt forbids answering beyond the provided sources and requires [n]
  citations; we verify every citation index actually exists.
- Every answer carries its sources (page, section, URL, last-updated date).
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

from visa_rag.retrieval import HybridRetriever, RetrievedChunk

load_dotenv()

SYSTEM_PROMPT = """\
You are a UK visas & immigration assistant. Answer ONLY from the numbered
sources provided. Rules:
- Cite sources inline as [1], [2] etc. after each claim. Every factual claim
  needs a citation.
- If the sources do not contain the answer, say exactly what is missing and
  do not guess. Never invent fees, dates, or eligibility rules.
- Quote fees and time limits exactly as written in the sources.
- Be concise and practical. Use plain English.
- End with: "This is guidance from gov.uk, not legal advice."
"""


@dataclass
class Answer:
    question: str
    text: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    refused: bool = False
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    def cited_indices(self) -> set[int]:
        return {int(m) for m in re.findall(r"\[(\d+)\]", self.text)}


def format_sources(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        updated = f" (last updated {c.updated_at})" if c.updated_at else ""
        blocks.append(f"[{i}] {c.doc_title} > {c.section or 'Main'}{updated}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def answer_question(
    question: str,
    retriever: HybridRetriever,
    model: str = "claude-sonnet-4-5",
    max_tokens: int = 1024,
    min_retrieval_score: float = 0.02,
    client=None,
) -> Answer:
    start = time.perf_counter()
    chunks = retriever.search(question)

    # Guardrail 1: retrieval confidence gate
    if not chunks or chunks[0].confidence < min_retrieval_score:
        return Answer(
            question=question, refused=True, sources=chunks,
            text=("I couldn't find this in the gov.uk visa guidance I have "
                  "indexed, so I won't guess. Try rephrasing, or check "
                  "https://www.gov.uk/browse/visas-immigration directly."),
            latency_s=time.perf_counter() - start,
        )

    if client is None:
        import anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY not set - copy .env.example to .env")
        client = anthropic.Anthropic()

    user_msg = (
        f"Sources:\n\n{format_sources(chunks)}\n\n"
        f"Question: {question}"
    )
    resp = client.messages.create(
        model=model, max_tokens=max_tokens, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = resp.content[0].text

    ans = Answer(
        question=question, text=text, sources=chunks,
        latency_s=time.perf_counter() - start,
        input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens,
    )

    # Guardrail 2: every [n] citation must point at a real source
    bad = [i for i in ans.cited_indices() if i < 1 or i > len(chunks)]
    if bad:
        ans.text += f"\n\n(Note: citations {bad} were invalid and should be ignored.)"
    return ans


def main() -> None:
    from visa_rag.config import load_config
    from visa_rag.retrieval import load_retriever

    question = " ".join(sys.argv[1:]).strip()
    if not question:
        raise SystemExit('Usage: python -m visa_rag.answer "your question"')

    cfg = load_config()
    retriever = load_retriever()
    ans = answer_question(
        question, retriever, model=cfg.llm.model,
        max_tokens=cfg.llm.max_tokens,
        min_retrieval_score=cfg.llm.min_retrieval_score,
    )
    print(f"\n{ans.text}\n")
    if not ans.refused:
        print("Sources:")
        for i, c in enumerate(ans.sources, 1):
            print(f"  [{i}] {c.doc_title} > {c.section or 'Main'} - {c.url}")
        print(f"\n({ans.latency_s:.1f}s, {ans.input_tokens} in / {ans.output_tokens} out tokens)")


if __name__ == "__main__":
    main()
