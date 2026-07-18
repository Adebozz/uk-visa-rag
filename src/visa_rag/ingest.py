"""Ingestion: gov.uk Content API -> cleaned, chunked corpus (parquet).

    python -m visa_rag.ingest            # fetch + chunk everything
    python -m visa_rag.ingest --no-fetch # re-chunk from cached raw JSON

Handles the three gov.uk schemas that cover the visa corpus:
- "guide":  multi-part pages (each part becomes a section)
- "answer" / "detailed_guide" / others: single-body pages
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import httpx
import pandas as pd

from visa_rag.chunking import Chunk, chunk_section
from visa_rag.config import Config, load_config, load_corpus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def fetch_page(client: httpx.Client, api_base: str, path: str) -> dict:
    r = client.get(f"{api_base}/{path}", follow_redirects=True)
    r.raise_for_status()
    return r.json()


def fetch_corpus(cfg: Config, pages: list[str]) -> list[Path]:
    raw_dir = Path(cfg.ingestion.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    with httpx.Client(timeout=30, headers={"User-Agent": "visa-rag-ingest"}) as client:
        for path in pages:
            out = raw_dir / f"{path.replace('/', '_')}.json"
            try:
                data = fetch_page(client, cfg.ingestion.api_base, path)
            except httpx.HTTPError as e:
                logger.warning("SKIP %s (%s)", path, e)
                continue
            out.write_text(json.dumps(data))
            saved.append(out)
            logger.info("fetched %s (%s)", path, data.get("schema_name"))
    return saved


def page_to_chunks(data: dict, cfg: Config) -> list[Chunk]:
    title = data.get("title", "")
    base_path = data.get("base_path", "")
    url = f"https://www.gov.uk{base_path}"
    updated = (data.get("public_updated_at") or data.get("updated_at") or "")[:10]
    details = data.get("details", {})
    kw = dict(
        doc_title=title, updated_at=updated,
        max_chars=cfg.ingestion.max_chunk_chars,
        overlap_chars=cfg.ingestion.overlap_chars,
    )

    chunks: list[Chunk] = []
    parts = details.get("parts") or []
    if parts:  # "guide" schema: one section per part
        for part in parts:
            chunks += chunk_section(
                part.get("body", ""), section=part.get("title", ""),
                url=f"{url}/{part.get('slug', '')}".rstrip("/"), **kw,
            )
    else:  # single-body schemas
        body = details.get("body", "")
        if isinstance(body, list):  # some schemas use [{content_type, content}]
            body = next(
                (b["content"] for b in body if b.get("content_type") == "text/html"), ""
            )
        chunks += chunk_section(body, section="", url=url, **kw)
    return chunks


def build_chunks(cfg: Config) -> pd.DataFrame:
    raw_dir = Path(cfg.ingestion.raw_dir)
    rows = []
    for f in sorted(raw_dir.glob("*.json")):
        data = json.loads(f.read_text())
        page_chunks = page_to_chunks(data, cfg)
        rows += [vars(c) for c in page_chunks]
        logger.info("%s -> %d chunks", f.stem, len(page_chunks))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No chunks produced - did ingestion fetch anything?")
    out = Path(cfg.ingestion.chunks_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    logger.info("Corpus: %d chunks from %d pages -> %s",
                len(df), df["doc_title"].nunique(), out)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Skip network fetch; re-chunk cached raw JSON")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not args.no_fetch:
        fetch_corpus(cfg, load_corpus())
    build_chunks(cfg)


if __name__ == "__main__":
    main()
