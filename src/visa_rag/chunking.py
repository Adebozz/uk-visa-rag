"""Heading-aware chunking of gov.uk HTML content.

Strategy: split each page section on h2/h3 headings so a chunk never spans
unrelated topics, then pack paragraphs into ~max_chars chunks with overlap.
Every chunk carries its provenance (page title, section, heading, URL) -
that metadata is what makes citations trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bs4 import BeautifulSoup

_BLOCK_TAGS = ["p", "li", "h2", "h3", "table"]


@dataclass
class Chunk:
    text: str
    doc_title: str
    section: str          # gov.uk guide part, e.g. "How much it costs"
    heading: str          # nearest h2/h3 within the section
    url: str
    updated_at: str = ""
    chunk_id: str = field(default="")

    def context_header(self) -> str:
        """Provenance line prepended when the chunk is shown to the LLM."""
        bits = [self.doc_title, self.section, self.heading]
        return " > ".join(b for b in bits if b)


def html_to_blocks(html: str) -> list[tuple[str, str]]:
    """Flatten HTML into (kind, text) blocks preserving heading boundaries."""
    soup = BeautifulSoup(html, "html.parser")
    blocks: list[tuple[str, str]] = []
    for el in soup.find_all(_BLOCK_TAGS):
        if el.find_parent("li") is not None:  # avoid duplicating nested li text
            continue
        text = " ".join(el.get_text(" ", strip=True).split())
        if not text:
            continue
        kind = "heading" if el.name in ("h2", "h3") else "text"
        blocks.append((kind, text))
    return blocks


def chunk_section(
    html: str,
    doc_title: str,
    section: str,
    url: str,
    updated_at: str = "",
    max_chars: int = 1400,
    overlap_chars: int = 200,
) -> list[Chunk]:
    """Chunk one page section, resetting at headings, packing to max_chars."""
    blocks = html_to_blocks(html)
    chunks: list[Chunk] = []
    heading = ""
    buf: list[str] = []

    def flush() -> None:
        if not buf:
            return
        text = "\n".join(buf).strip()
        if len(text) < 40:  # drop stubs (nav crumbs, empty sections)
            buf.clear()
            return
        chunks.append(
            Chunk(text=text, doc_title=doc_title, section=section,
                  heading=heading, url=url, updated_at=updated_at)
        )
        buf.clear()

    for kind, text in blocks:
        if kind == "heading":
            flush()
            heading = text
            continue
        candidate_len = sum(len(b) + 1 for b in buf) + len(text)
        if candidate_len > max_chars and buf:
            flush()
            # overlap: carry the tail of the previous chunk forward
            if chunks and overlap_chars > 0:
                tail = chunks[-1].text[-overlap_chars:]
                buf.append("…" + tail)
        buf.append(text)
    flush()

    for i, c in enumerate(chunks):
        slug = url.rstrip("/").rsplit("/", 1)[-1]
        c.chunk_id = f"{slug}::{section or 'main'}::{i}"
    return chunks
