"""
app.py
Streamlit UI for the Teacher Intelligence Agent.

Pick (or type) a question about a student's tutoring session and get back either a quick
RAG summary or a full set of structured insights (+ an optional PDF report).

The backend runs on open-source models (Ollama) or Gemini. Run:  streamlit run app.py
"""
import os

import streamlit as st

# On Streamlit Community Cloud, configuration is provided via st.secrets rather than
# real environment variables. Bridge them into os.environ BEFORE importing config,
# which reads its settings from the environment at import time.
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass

import config

# Flatten an LLM response's content to plain text. Import the canonical helper, but keep a
# built-in fallback so the agent/PDF path never crashes if the deployed module is out of date.
try:
    from rag_engine import content_to_text
except Exception:  # noqa: BLE001
    def content_to_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(block.get("text") or block.get("content") or "")
                else:
                    parts.append(str(block))
            return "".join(parts).strip()
        return str(content)

st.set_page_config(page_title="Teacher Intelligence Agent", page_icon="🧑‍🏫", layout="centered")

# ----------------------------------------------------------------- styling
st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-family: -apple-system, BlinkMacSystemFont,
        "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
      .block-container { padding-top: 2rem; max-width: 820px; }
      h1, h2, h3 { font-weight: 600; letter-spacing: -0.01em; }

      .hero { background: linear-gradient(135deg, #eef1f6 0%, #f5f3fb 55%, #ffffff 100%);
        border: 1px solid #e4e6ee; border-radius: 20px; padding: 28px 32px; margin: 0 0 1.3rem; }
      .hero-eyebrow { font-size: 0.74rem; font-weight: 700; letter-spacing: 0.09em;
        text-transform: uppercase; color: #6a5acd; }
      .hero-title { font-size: 1.95rem; font-weight: 600; letter-spacing: -0.02em;
        color: #1d1d1f; margin: 8px 0 10px; }
      .hero-sub { font-size: 0.98rem; color: #3a3a3c; line-height: 1.55; max-width: 680px; }
      .hero-meta { margin-top: 15px; display: flex; flex-wrap: wrap; gap: 8px; }
      .hero-meta span { background: rgba(255,255,255,0.8); border: 1px solid #dfe1ea;
        color: #3a3a3c; border-radius: 999px; padding: 4px 12px; font-size: 0.78rem; font-weight: 500; }

      [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px;
        box-shadow: 0 1px 2px rgba(16,24,40,0.04); background: #ffffff; }

      .sect { font-size: 0.76rem; font-weight: 700; letter-spacing: 0.05em;
        text-transform: uppercase; color: #86868b; margin: 0.1rem 0 0.5rem; }
      .hint { font-size: 0.85rem; color: #6e6e73; margin: 0 0 0.4rem; }
      .answer { background: #f6f7f9; border: 1px solid #edeff2; border-radius: 12px;
        padding: 16px 20px; color: #1d1d1f; line-height: 1.6; }
      /* example chips: compact, left-aligned secondary buttons */
      div[data-testid="column"] button[kind="secondary"] { text-align: left;
        font-size: 0.85rem; font-weight: 450; white-space: normal; line-height: 1.3;
        color: #35507a; border-color: #e0e4ee; background: #f7f9fc; }
    </style>
    """,
    unsafe_allow_html=True,
)


def backend_ready() -> tuple[bool, str]:
    """Best-effort check that the configured LLM backend is reachable."""
    if config.LLM_PROVIDER == "gemini":
        key_set = bool(config.GOOGLE_API_KEY)
        return key_set, (
            "Using native Google Gemini"
            if key_set
            else "LLM_PROVIDER=gemini but no GOOGLE_API_KEY is set."
        )
    if config.LLM_PROVIDER != "ollama":
        key_set = bool(config.OPENAI_API_KEY and config.OPENAI_API_KEY != "not-needed")
        return key_set, (
            f"Using OpenAI-compatible endpoint: {config.OPENAI_BASE_URL}"
            if key_set
            else "LLM_PROVIDER=openai but no API key is set (OPENAI_API_KEY)."
        )
    try:
        import urllib.request

        with urllib.request.urlopen(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=2) as resp:
            return resp.status == 200, f"Ollama reachable at {config.OLLAMA_BASE_URL}"
    except Exception:
        return False, (
            f"Ollama not reachable at {config.OLLAMA_BASE_URL}. "
            f"Start it with `ollama serve` and pull a model: `ollama pull {config.AGENT_MODEL}`."
        )


# ----------------------------------------------------------------- hero
st.markdown(
    """
    <div class="hero">
      <div class="hero-eyebrow">RAG · Tool-using Agent</div>
      <div class="hero-title">🧑‍🏫 Teacher Intelligence Agent</div>
      <div class="hero-sub">Turns raw tutoring transcripts into actionable insight about a
        learner — what they understood, where they got stuck, and what the tutor should do
        next — using retrieval-augmented generation and a tool-using agent. Runs on
        open-source models or Gemini.</div>
      <div class="hero-meta">
        <span>Semantic retrieval (FAISS)</span>
        <span>LangChain agent</span>
        <span>Structured insights</span>
        <span>PDF report</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

ready, backend_msg = backend_ready()

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. Transcripts are chunked & embedded (SentenceTransformers, local)\n"
        "2. Stored in a FAISS vector index\n"
        "3. A LangChain agent retrieves relevant chunks (`rag_search` tool) and reasons over them\n"
        "4. Insights are returned in a structured schema, with an optional PDF export tool"
    )
    st.divider()
    st.subheader("Backend")
    (st.success if ready else st.error)(backend_msg)
    st.caption(f"Provider: `{config.LLM_PROVIDER}` · summary: `{config.SUMMARY_MODEL}` · agent: `{config.AGENT_MODEL}`")
    st.divider()
    st.subheader("Students in the demo data")
    st.caption("Willy (math, science) · Maya (fractions) · Diego (multiplication) · "
               "Aisha (reading) · Sam (geometry) · Nia (photosynthesis)")

# Example prompts — clicking one fills the box.
EXAMPLES = [
    "Did Willy understand the three states of matter, and what should the tutor do next?",
    "Where did Maya get stuck adding fractions — and did she recover by the end?",
    "Summarize Diego's multiplication session: what frustrated him and how did he respond?",
    "How engaged and confident was Aisha in the reading-comprehension session?",
    "Did Sam confuse area and perimeter? Give concrete next steps for the tutor.",
    "What misconceptions came up across students this week, and which need follow-up?",
]

if "query" not in st.session_state:
    st.session_state["query"] = EXAMPLES[0]

with st.container(border=True):
    st.markdown("<div class='sect'>Ask about a student's session</div>", unsafe_allow_html=True)
    st.markdown("<div class='hint'>Pick an example to get started, or write your own:</div>",
                unsafe_allow_html=True)
    ex_cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if ex_cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
            st.session_state["query"] = ex

    query = st.text_area(
        "Your question",
        key="query",
        placeholder=("e.g. Did Willy understand the three states of matter? What should the "
                     "tutor do next? · Where did Maya get stuck on fractions? · "
                     "Summarize Diego's session and list his misconceptions."),
        height=90,
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(2)
    run_search = col1.button("🔍 Quick RAG summary", use_container_width=True, type="primary")
    run_agent = col2.button("🧠 Full structured insights + PDF", use_container_width=True)

if (run_search or run_agent) and not ready:
    st.error(backend_msg)
elif (run_search or run_agent) and not query.strip():
    st.warning("Enter a question first.")
elif run_search:
    with st.spinner("Retrieving relevant transcript chunks and summarizing..."):
        from rag_engine import RAGSearch

        rag = RAGSearch()
        with st.container(border=True):
            st.markdown("<div class='sect'>Summary</div>", unsafe_allow_html=True)
            # Coerce to text at the app level too, so a stale rag_engine (returning raw .content)
            # still renders as prose rather than a JSON tree.
            st.markdown(f"<div class='answer'>{content_to_text(rag.search_and_summarize(query))}</div>",
                        unsafe_allow_html=True)

elif run_agent:
    with st.spinner("Running the tutoring-insights agent (retrieval → reasoning → structured output)..."):
        from agent import build_agent_executor

        executor = build_agent_executor()
        full_query = query + " Also save the insights as a PDF named 'tutoring_insights.pdf'."
        result = executor.invoke({"input": full_query})

        with st.container(border=True):
            st.markdown("<div class='sect'>Agent output</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='answer'>{content_to_text(result.get('output', ''))}</div>",
                        unsafe_allow_html=True)

            pdf_path = "tutoring_insights.pdf"
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as fh:
                    st.download_button("⬇️ Download PDF report", fh, file_name="tutoring_insights.pdf")

st.caption(
    "Demo data: sample math / science / reading tutoring transcripts in `data/`. "
    "Swap in your own JSON transcripts (role/text/timestamp turns) to try it on real sessions."
)
