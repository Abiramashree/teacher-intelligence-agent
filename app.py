"""
app.py
Streamlit UI for the Teacher Intelligence Agent.

Pick a question about a student's tutoring session and get back either a quick
RAG summary or a full set of structured insights (+ an optional PDF report).

The backend runs entirely on open-source models (Ollama by default), so the demo
needs no API key. Run:  streamlit run app.py
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

st.set_page_config(page_title="Teacher Intelligence Agent", page_icon="🧑‍🏫", layout="centered")


def backend_ready() -> tuple[bool, str]:
    """Best-effort check that the configured LLM backend is reachable."""
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


st.title("🧑‍🏫 Teacher Intelligence Agent")
st.caption("RAG + a tool-using agent that turns tutoring transcripts into actionable student insights — on open-source models.")

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

query = st.text_area(
    "Ask about a student's session",
    placeholder="e.g. Did the student understand the three states of matter? What should the tutor do next?",
    height=90,
)

col1, col2 = st.columns(2)
run_search = col1.button("🔍 Quick RAG summary", use_container_width=True)
run_agent = col2.button("🧠 Full structured insights + PDF", use_container_width=True)

if (run_search or run_agent) and not ready:
    st.error(backend_msg)
elif (run_search or run_agent) and not query.strip():
    st.warning("Enter a question first.")
elif run_search:
    with st.spinner("Retrieving relevant transcript chunks and summarizing..."):
        from rag_engine import RAGSearch

        rag = RAGSearch()
        st.markdown("### Summary")
        st.write(rag.search_and_summarize(query))

elif run_agent:
    with st.spinner("Running the tutoring-insights agent (retrieval → reasoning → structured output)..."):
        from agent import build_agent_executor

        executor = build_agent_executor()
        full_query = query + " Also save the insights as a PDF named 'tutoring_insights.pdf'."
        result = executor.invoke({"input": full_query})
        st.markdown("### Agent Output")
        st.write(result["output"])

        pdf_path = "tutoring_insights.pdf"
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as fh:
                st.download_button("⬇️ Download PDF report", fh, file_name="tutoring_insights.pdf")

st.divider()
st.caption(
    "Demo data: anonymized sample math/science tutoring transcripts in `data/`. "
    "Swap in your own JSON transcripts (role/text/timestamp turns) to try it on real sessions."
)
