"""Run the evaluation suite.

    python -m visa_rag.evals.run                  # retrieval on all pairs (free)
    python -m visa_rag.evals.run --answers 25     # + judge 25 generated answers

Retrieval metrics need no API calls. Answer metrics call Claude twice per
question (answer + judge), so they're run on a bounded subset.

Results: printed report + evals/results/<timestamp>.json for tracking
regressions across retrieval experiments.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from visa_rag.evals.metrics import citation_coverage, hit_at_k, reciprocal_rank

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_eval_set(path: str) -> list[dict]:
    lines = Path(path).read_text().strip().splitlines()
    return [json.loads(line) for line in lines]


def eval_retrieval(pairs: list[dict], retriever) -> dict:
    hits1 = hits3 = hits6 = 0
    rranks = []
    for p in pairs:
        ids = [c.chunk_id for c in retriever.search(p["question"])]
        hits1 += hit_at_k(ids, p["source_chunk_id"], 1)
        hits3 += hit_at_k(ids, p["source_chunk_id"], 3)
        hits6 += hit_at_k(ids, p["source_chunk_id"], 6)
        rranks.append(reciprocal_rank(ids, p["source_chunk_id"]))
    n = len(pairs)
    return {
        "n_questions": n,
        "hit@1": round(hits1 / n, 3),
        "hit@3": round(hits3 / n, 3),
        "hit@6": round(hits6 / n, 3),
        "mrr": round(sum(rranks) / n, 3),
    }


def eval_answers(pairs: list[dict], retriever, cfg, n: int) -> dict:
    import anthropic

    from visa_rag.answer import answer_question
    from visa_rag.evals.judge import judge_answer

    client = anthropic.Anthropic()
    subset = pairs[:n]
    faith, rel, coverage, latencies = [], [], [], []
    refusals = judge_failures = 0
    tokens_in = tokens_out = 0

    for i, p in enumerate(subset, 1):
        ans = answer_question(
            p["question"], retriever, model=cfg.llm.model,
            max_tokens=cfg.llm.max_tokens,
            min_retrieval_score=cfg.llm.min_retrieval_score, client=client,
        )
        latencies.append(ans.latency_s)
        tokens_in += ans.input_tokens
        tokens_out += ans.output_tokens
        if ans.refused:
            refusals += 1  # refusing a grounded question = retrieval failure
            continue
        coverage.append(citation_coverage(ans.text))
        verdict = judge_answer(client, ans)
        if verdict is None:
            judge_failures += 1
            continue
        faith.append(verdict["faithfulness"])
        rel.append(verdict["relevance"])
        if verdict["faithfulness"] <= 2:
            logger.warning("LOW FAITHFULNESS on %r: %s",
                           p["question"], verdict["reasoning"])
        logger.info("[%d/%d] faith=%d rel=%d | %s",
                    i, len(subset), verdict["faithfulness"], verdict["relevance"],
                    p["question"][:60])

    return {
        "n_answered": len(subset),
        "faithfulness_mean": round(statistics.mean(faith), 2) if faith else None,
        "faithfulness_pct_5": round(sum(f == 5 for f in faith) / len(faith), 3) if faith else None,
        "relevance_mean": round(statistics.mean(rel), 2) if rel else None,
        "citation_coverage_mean": round(statistics.mean(coverage), 3) if coverage else None,
        "refusal_rate": round(refusals / len(subset), 3),
        "judge_failures": judge_failures,
        "latency_p50_s": round(statistics.median(latencies), 2),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", default="data/eval_set.jsonl")
    parser.add_argument("--answers", type=int, default=0,
                        help="Also judge N generated answers (API cost)")
    args = parser.parse_args()

    from visa_rag.config import load_config
    from visa_rag.retrieval import load_retriever

    cfg = load_config()
    retriever = load_retriever()
    pairs = load_eval_set(args.eval_set)
    logger.info("Loaded %d eval pairs", len(pairs))

    t0 = time.perf_counter()
    report = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "retrieval": eval_retrieval(pairs, retriever)}
    logger.info("Retrieval eval done in %.1fs", time.perf_counter() - t0)

    if args.answers > 0:
        report["answers"] = eval_answers(pairs, retriever, cfg, args.answers)

    out_dir = Path("evals/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{report['timestamp'].replace(':', '-')}.json"
    out.write_text(json.dumps(report, indent=2))

    print("\n===== EVALUATION REPORT =====")
    print(json.dumps(report, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
