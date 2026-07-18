from visa_rag.chunking import chunk_section, html_to_blocks

HTML = """
<h2>How much it costs</h2>
<p>The Skilled Worker visa fee depends on your circumstances.</p>
<p>You'll pay £769 if you're applying from outside the UK for up to 3 years.</p>
<h2>Healthcare surcharge</h2>
<p>You'll also pay the immigration health surcharge, usually £1,035 per year.</p>
<ul><li>Check the exact amount before you apply.</li></ul>
"""


def test_blocks_preserve_headings_and_lists():
    blocks = html_to_blocks(HTML)
    kinds = [k for k, _ in blocks]
    assert kinds.count("heading") == 2
    assert any("£1,035" in t for _, t in blocks)
    assert any("Check the exact amount" in t for _, t in blocks)


def test_chunks_split_at_headings():
    chunks = chunk_section(HTML, doc_title="Skilled Worker visa",
                           section="Costs", url="https://www.gov.uk/skilled-worker-visa")
    assert len(chunks) == 2  # one per h2
    assert chunks[0].heading == "How much it costs"
    assert chunks[1].heading == "Healthcare surcharge"
    assert "£769" in chunks[0].text and "£1,035" in chunks[1].text


def test_chunk_metadata_and_ids():
    chunks = chunk_section(HTML, doc_title="Skilled Worker visa",
                           section="Costs", url="https://www.gov.uk/skilled-worker-visa")
    for c in chunks:
        assert c.doc_title == "Skilled Worker visa"
        assert c.chunk_id.startswith("skilled-worker-visa::Costs::")
        assert "Skilled Worker visa > Costs" in c.context_header()


def test_long_sections_split_with_overlap():
    para = "<p>" + "immigration rules apply here. " * 20 + "</p>"
    html = "<h2>Rules</h2>" + para * 6
    chunks = chunk_section(html, doc_title="T", section="S", url="https://x/y",
                           max_chars=800, overlap_chars=100)
    assert len(chunks) > 1
    assert all(len(c.text) <= 1000 for c in chunks)
    # overlap: later chunks start with carried-over tail
    assert chunks[1].text.startswith("…")
