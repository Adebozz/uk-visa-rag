"""Streamlit chat UI for the visa RAG assistant.

    make chat    (API must be running: make serve)
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

st.set_page_config(page_title="UK Visa Assistant", page_icon="🛂", layout="centered")

# ---------- styling ----------

st.markdown(
    """
<style>
[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 12px;
    padding: 10px 14px;
}
.hero {
    background: linear-gradient(120deg, #1a2a6c 0%, #2a4d8f 60%, #3b6db3 100%);
    border-radius: 16px;
    padding: 22px 26px 18px;
    margin-bottom: 12px;
    color: white;
}
.hero h1 { color: white; font-size: 1.5rem; margin: 0 0 4px; }
.hero p  { color: #d6e4f5; margin: 0 0 10px; font-size: 0.95rem; }
.pill {
    display: inline-block;
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.25);
    color: #e8f1fb;
    border-radius: 999px;
    padding: 2px 11px;
    margin: 2px 5px 2px 0;
    font-size: 0.74rem;
}
</style>
<div class="hero">
  <h1>🛂 UK Visa & Immigration Assistant</h1>
  <p>Answers grounded in official gov.uk guidance — every claim cited,
  and it refuses rather than guesses.</p>
  <span class="pill">Hybrid retrieval (BM25 + dense + RRF)</span>
  <span class="pill">Claude</span>
  <span class="pill">Cited answers</span>
  <span class="pill">Evaluated: 4.68/5 faithfulness</span>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- sidebar ----------

st.sidebar.title("🛂 Visa Assistant")
base_url = st.sidebar.text_input(
    "API URL", os.environ.get("VISA_API_URL", "http://localhost:8000")
)


@st.cache_data(ttl=30)
def api_health(url: str) -> bool:
    try:
        r = httpx.get(f"{url}/health", timeout=5)
        return r.status_code == 200 and r.json().get("retriever_loaded", False)
    except httpx.HTTPError:
        return False


if api_health(base_url):
    st.sidebar.success("API connected ✅")
else:
    st.sidebar.error("API not reachable - run `make serve` first")

st.sidebar.divider()
st.sidebar.markdown("**Try asking:**")
EXAMPLES = [
    "How much does a Skilled Worker visa cost?",
    "Can I switch from a Student visa to a Skilled Worker visa?",
    "How long can I stay on a Graduate visa?",
    "Can my partner join me on a Health and Care Worker visa?",
    "What documents do I need for a Standard Visitor visa?",
]
for ex in EXAMPLES:
    if st.sidebar.button(ex, use_container_width=True):
        st.session_state["queued_question"] = ex

st.sidebar.divider()
st.sidebar.caption(
    "Indexed: 30 gov.uk visa guides (~1,000 chunks). Answers are generated "
    "from retrieved passages only; this is guidance, not legal advice."
)

# ---------- chat ----------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} gov.uk sources"):
                for s in msg["sources"]:
                    updated = f" · updated {s['updated_at']}" if s["updated_at"] else ""
                    st.markdown(
                        f"**[{s['n']}]** [{s['doc_title']} › "
                        f"{s['section'] or 'Main'}]({s['url']}){updated}"
                    )

question = st.chat_input("Ask about UK visas...")
if not question and "queued_question" in st.session_state:
    question = st.session_state.pop("queued_question")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching gov.uk guidance..."):
            try:
                r = httpx.post(f"{base_url}/ask", json={"question": question}, timeout=60)
                r.raise_for_status()
                data = r.json()
            except httpx.HTTPError as e:
                st.error(f"API call failed: {e}")
                st.stop()

        if data["refused"]:
            st.warning(data["answer"])
            sources = []
        else:
            st.markdown(data["answer"])
            sources = data["sources"]
            with st.expander(f"📚 {len(sources)} gov.uk sources"):
                for s in sources:
                    updated = f" · updated {s['updated_at']}" if s["updated_at"] else ""
                    st.markdown(
                        f"**[{s['n']}]** [{s['doc_title']} › "
                        f"{s['section'] or 'Main'}]({s['url']}){updated}"
                    )
            st.caption(f"{data['latency_s']}s · "
                       f"{data['input_tokens']}→{data['output_tokens']} tokens")

    st.session_state.messages.append(
        {"role": "assistant", "content": data["answer"], "sources": sources}
    )
