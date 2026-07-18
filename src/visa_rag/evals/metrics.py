"""Pure metric computations - no API calls, fully unit-testable."""

from __future__ import annotations

import re


def hit_at_k(retrieved_ids: list[str], source_id: str, k: int) -> bool:
    """Did the true source chunk appear in the top k results?"""
    return source_id in retrieved_ids[:k]


def reciprocal_rank(retrieved_ids: list[str], source_id: str) -> float:
    """1/rank of the true source (0 if absent). Mean over queries = MRR."""
    try:
        return 1.0 / (retrieved_ids.index(source_id) + 1)
    except ValueError:
        return 0.0


def citation_coverage(answer_text: str) -> float:
    """Fraction of substantive sentences that carry a [n] citation.

    A cheap, deterministic proxy for 'every claim is cited'. Sentences under
    40 chars (headers, sign-offs) are ignored.
    """
    sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", answer_text)
        if len(s.strip()) >= 40
    ]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if re.search(r"\[\d+\]", s))
    return cited / len(sentences)


def parse_judge_json(raw: str) -> dict | None:
    """Extract the first JSON object from a judge response, tolerating prose."""
    import json

    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        data = json.loads(raw[start:end])
    except ValueError:
        return None
    if not {"faithfulness", "relevance"} <= set(data):
        return None
    try:
        f, r = int(data["faithfulness"]), int(data["relevance"])
    except (TypeError, ValueError):
        return None
    if not (1 <= f <= 5 and 1 <= r <= 5):
        return None
    return {"faithfulness": f, "relevance": r,
            "reasoning": str(data.get("reasoning", ""))}
