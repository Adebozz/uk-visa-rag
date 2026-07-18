"""Generate a grounded evaluation set from the corpus.

    python -m visa_rag.evals.generate --n 100

For each sampled chunk, Claude writes a question that the chunk answers.
Because we KNOW which chunk each question came from, retrieval can be scored
exactly: did the retriever put the source chunk in the top k?

Output: data/eval_set.jsonl  {question, source_chunk_id, reference_answer}
Cost: ~£0.5 for 100 pairs with Haiku.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GENERATOR_MODEL = "claude-haiku-4-5"

PROMPT = """\
Below is an extract from official UK visa guidance.

Source: {header}
---
{text}
---

Write ONE realistic question that a visa applicant might type into a help
tool, which this extract fully answers. Then give the correct answer using
only the extract.

Rules:
- The question must stand alone (no "this visa" - name the visa).
- Prefer questions about concrete facts: fees, deadlines, eligibility, documents.
- Answer in 1-3 sentences.

Respond with JSON only: {{"question": "...", "answer": "..."}}"""


def generate_pair(client, chunk: pd.Series) -> dict | None:
    header = f"{chunk.doc_title} > {chunk.section or 'Main'}"
    resp = client.messages.create(
        model=GENERATOR_MODEL, max_tokens=400,
        messages=[{"role": "user",
                   "content": PROMPT.format(header=header, text=chunk.text)}],
    )
    raw = resp.content[0].text.strip()
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        data = json.loads(raw[start:end])
        return {
            "question": data["question"],
            "reference_answer": data["answer"],
            "source_chunk_id": chunk.chunk_id,
            "doc_title": chunk.doc_title,
        }
    except (ValueError, KeyError) as e:
        logger.warning("Unparseable generation for %s (%s)", chunk.chunk_id, e)
        return None


def main() -> None:
    import anthropic

    from visa_rag.config import load_config

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", default="data/eval_set.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    chunks = pd.read_parquet(cfg.ingestion.chunks_path)
    # Sample spread across documents: proportional but at least 1 per doc.
    eligible = chunks[chunks["text"].str.len() > 300]
    rng = random.Random(args.seed)
    sampled = (
        eligible.groupby("doc_title", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), max(1, round(args.n * len(g) / len(eligible)))),
                                  random_state=args.seed))
        .sample(frac=1.0, random_state=args.seed)
        .head(args.n)
    )
    logger.info("Generating %d QA pairs across %d documents...",
                len(sampled), sampled["doc_title"].nunique())

    client = anthropic.Anthropic()
    pairs = []
    for _, chunk in sampled.iterrows():
        pair = generate_pair(client, chunk)
        if pair:
            pairs.append(pair)
            if len(pairs) % 10 == 0:
                logger.info("%d pairs done", len(pairs))
    rng.shuffle(pairs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(p) for p in pairs) + "\n")
    logger.info("Wrote %d QA pairs -> %s", len(pairs), out)


if __name__ == "__main__":
    main()
