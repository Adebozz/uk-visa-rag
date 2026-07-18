"""LLM-as-judge: score generated answers for faithfulness and relevance."""

from __future__ import annotations

from visa_rag.answer import Answer
from visa_rag.evals.metrics import parse_judge_json

JUDGE_MODEL = "claude-sonnet-4-5"

JUDGE_PROMPT = """\
You are grading a UK-visa assistant's answer.

Question: {question}

Sources the assistant was given:
{sources}

Assistant's answer:
{answer}

Score two things (1-5):
1. faithfulness - is EVERY factual claim in the answer supported by the
   sources? 5 = fully supported; 3 = minor unsupported detail; 1 = fabricated
   fees/rules/eligibility.
2. relevance - does it actually answer the user's question? 5 = directly and
   completely; 3 = partial; 1 = off-topic.

Respond with JSON only:
{{"faithfulness": <1-5>, "relevance": <1-5>, "reasoning": "<one sentence>"}}"""


def judge_answer(client, ans: Answer) -> dict | None:
    sources = "\n\n".join(
        f"[{i}] {c.doc_title} > {c.section or 'Main'}\n{c.text}"
        for i, c in enumerate(ans.sources, 1)
    )
    resp = client.messages.create(
        model=JUDGE_MODEL, max_tokens=300,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=ans.question, sources=sources, answer=ans.text)}],
    )
    return parse_judge_json(resp.content[0].text)
