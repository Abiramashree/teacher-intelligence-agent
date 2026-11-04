# 🧑‍🏫 Teacher Intelligence Agent

A Retrieval-Augmented Generation (RAG) system that reads student–tutor transcripts and
turns them into structured, actionable insights — understanding level, misconceptions,
sentiment, suggested next steps — with an optional one-click PDF report for educators.

**Runs entirely on open-source models.** By default the LLM backend is [Ollama](https://ollama.com)
serving a local model (Llama 3.1 8B), and embeddings use a local SentenceTransformers
model — so the demo needs **no API key and no third-party service**.

---

## What it does

Given a folder of tutoring session transcripts, the agent answers questions like:

> *"Did the student understand the three states of matter? What should the tutor do next?"*

…and returns either a quick semantic-search summary, or a full structured report covering:

- Topics covered & knowledge gaps
- Misconceptions identified
- Student sentiment & confidence
- Bloom's taxonomy level
- Suggested follow-up questions and activities
- A downloadable PDF report

## Architecture

```
Transcripts (JSON)
      │
      ▼
Chunking (LangChain RecursiveCharacterTextSplitter)
      │
      ▼
Embeddings (SentenceTransformers: all-MiniLM-L6-v2, local)
      │
      ▼
FAISS vector index  ──────────────┐
      │                            │
      ▼                            ▼
RAGSearch (open-source LLM)   LangChain tool-calling agent (open-source LLM)
  quick summaries                rag_search → structured Pydantic
                                 insights → save_tutoring_insights_pdf
```

The agent is a real **tool-calling LangChain agent**, not a fixed pipeline: it decides
when to call `rag_search` to ground itself in transcript evidence, and when to call
`save_tutoring_insights_pdf` to export a report — both implemented as LangChain `@tool`s.

## Tech stack

| Layer | Tool |
|---|---|
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`), local |
| Vector store | FAISS |
| LLM backend | **Ollama** (open-source models, local) — or any OpenAI-compatible server |
| Orchestration | LangChain (tools, agents, prompt templates) |
| Structured output | Pydantic |
| Report | ReportLab (PDF) |
| UI | Streamlit |

## Quick start (local, open-source)

**Prerequisites:** Python 3.10+, and [Ollama](https://ollama.com/download).

```bash
# 1. install Python deps
pip install -r requirements.txt

# 2. pull an open-source model (8B fits comfortably in ~16GB)
ollama pull llama3.1:8b

# 3. build the vector index and launch the demo
python scripts/build_index.py
streamlit run app.py
```

Or do all of it with one command:

```bash
./run_demo.sh        # installs deps, starts Ollama, pulls the model, builds the index, launches the UI
# or
make demo
```

Open http://localhost:8501, type a question, and click **Quick RAG summary** or
**Full structured insights + PDF**.

### Run it fully containerized

```bash
docker compose up --build
docker compose exec ollama ollama pull llama3.1:8b   # one time
# open http://localhost:8501
```

### Hosted demo (free, always-available)

Deploy on **Streamlit Community Cloud** with the LLM running on **Hugging Face's
serverless Inference Providers** (open-source models). Free, public URL, no backend to
manage. Full steps: [`deploy/streamlit-cloud/`](deploy/streamlit-cloud/).

Short version: push this repo to GitHub, go to <https://share.streamlit.io>, deploy
`app.py` from the repo, and add these **Secrets**:

```toml
LLM_PROVIDER = "openai"
OPENAI_BASE_URL = "https://router.huggingface.co/v1"
OPENAI_API_KEY = "hf_your_token_here"
SUMMARY_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
AGENT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
```

> Free HF serverless inference has a small monthly credit (enough for a light demo), and
> the app sleeps when idle and wakes on the next visit.

**Self-hosted alternative (Hugging Face Space, requires HF PRO).** A single-container
Docker Space that runs Ollama + the model itself — no per-call limits, fully self-hosted —
lives in [`deploy/hf-space/`](deploy/hf-space/). Note that Docker Spaces are **no longer
free**: HF now requires a PRO subscription for backend (non-static) Spaces.

## Configuration

All settings are environment variables (see [`.env.example`](.env.example)); copy it to
`.env` to override. Key ones:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` (local) or `openai` (any OpenAI-compatible server) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `AGENT_MODEL` / `SUMMARY_MODEL` | `llama3.1:8b` | Open-source models to use |
| `OPENAI_BASE_URL` | `http://localhost:4000/v1` | Used when `LLM_PROVIDER=openai` (e.g. a LiteLLM gateway / vLLM) |
| `TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP` | `5`, `1000`, `200` | Retrieval tuning |

To run against a different open-source model, just pull it and set the env var:
`ollama pull qwen2.5:7b && AGENT_MODEL=qwen2.5:7b streamlit run app.py`.

## Project structure

```
├── app.py              # Streamlit UI (entry point)
├── config.py           # Env-driven configuration
├── llm.py              # Open-source LLM factory (Ollama / OpenAI-compatible)
├── rag_engine.py       # Chunking, embeddings, FAISS store, RAG summarization
├── agent.py            # Insights schema, tools, agent executor
├── report.py           # PDF report generation
├── scripts/build_index.py  # Pre-build the FAISS index
├── tests/              # Offline tests (no LLM/network needed)
├── data/               # Sample anonymized tutoring transcripts (JSON)
├── requirements.txt
├── .env.example
├── Dockerfile / docker-compose.yml
├── deploy/streamlit-cloud/  # Free hosted demo (Streamlit Cloud + HF serverless)
├── deploy/hf-space/    # Self-hosted Docker Space (requires HF PRO)
└── Makefile / run_demo.sh
```

## Tests

```bash
make test        # or: pytest -q
```

The tests cover document loading, the FAISS build/search round-trip, and PDF
generation — none of them require a running model.

## Use cases

- **EdTech platforms** — personalized learning analytics at scale
- **Tutors & teachers** — automatic progress tracking across sessions
- **Learning-analytics research** — engagement & comprehension trend analysis
- **Student-support teams** — early identification of confusion or disengagement

## Notes

- Sample data in `data/` is small, anonymized, and included only to demo the pipeline —
  swap in your own transcripts (same `role`/`text`/`timestamp` JSON shape) to try it on
  real sessions.
- The FAISS index is rebuilt automatically the first time you run the app if `faiss_store/`
  doesn't already exist (`scripts/build_index.py` just does it ahead of time).

## License

MIT — see [LICENSE](LICENSE).
