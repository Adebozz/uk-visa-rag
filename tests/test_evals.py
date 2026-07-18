from visa_rag.evals.metrics import (
    citation_coverage,
    hit_at_k,
    parse_judge_json,
    reciprocal_rank,
)


def test_hit_at_k():
    ids = ["a", "b", "c", "d"]
    assert hit_at_k(ids, "a", 1)
    assert not hit_at_k(ids, "c", 2)
    assert hit_at_k(ids, "c", 3)
    assert not hit_at_k(ids, "zzz", 6)


def test_reciprocal_rank():
    ids = ["a", "b", "c"]
    assert reciprocal_rank(ids, "a") == 1.0
    assert reciprocal_rank(ids, "b") == 0.5
    assert reciprocal_rank(ids, "missing") == 0.0


def test_citation_coverage_counts_substantive_sentences():
    text = ("The Skilled Worker visa fee is £769 for up to three years [1]. "
            "You must also pay the healthcare surcharge every year you stay [2]. "
            "This is guidance from gov.uk, not legal advice.")
    cov = citation_coverage(text)
    assert 0.6 < cov < 0.7  # 2 of 3 substantive sentences cited


def test_citation_coverage_empty():
    assert citation_coverage("Short.") == 0.0


def test_parse_judge_json_happy_path():
    raw = 'Here is my verdict: {"faithfulness": 5, "relevance": 4, "reasoning": "ok"}'
    v = parse_judge_json(raw)
    assert v == {"faithfulness": 5, "relevance": 4, "reasoning": "ok"}


def test_parse_judge_json_rejects_garbage():
    assert parse_judge_json("no json here") is None
    assert parse_judge_json('{"faithfulness": 9, "relevance": 4}') is None
    assert parse_judge_json('{"relevance": 4}') is None
    assert parse_judge_json('{"faithfulness": "high", "relevance": 4}') is None
