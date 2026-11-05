# Teacher Intelligence Agent

**Live demo → https://teacher-intelligence-agent.streamlit.app/**

A retrieval-augmented, tool-using agent that reads student–tutor transcripts and turns them
into structured, actionable teaching insight — what the learner understood, where they got
stuck, how they felt, and what the tutor should do next — with an optional one-click PDF
report for educators.

It is designed to be backend-agnostic: the same code runs on fully open-source models
(Ollama, locally) or on Gemini / any OpenAI-compatible endpoint, so the demo can be hosted
for free while a production deployment can swap in whatever model it prefers.

---

## The problem it addresses

Tutoring transcripts are rich but unstructured — a teacher would have to re-read an entire
session to answer "did this student actually get it, and what should I do next time?" This
project turns that raw conversation into a concise, evidence-grounded diagnosis a tutor can
act on in seconds.

## What it does

Ask a natural-language question about a session — for example *"Where did Maya get stuck
adding fractions, and did she recover?"* — and the agent returns either:

- a **quick summary** grounded in the most relevant transcript passages, or
- a **full structured report**: topics covered, the student's understanding level,
  misconceptions, knowledge gaps, suggested questions and activities, next-best prompts,
  sentiment, Bloom's-taxonomy level, confidence, and supporting quotes — rendered as a clean
  report and exportable to PDF.

The demo ships with transcripts for six learners across different subjects (fractions,
multiplication, reading comprehension, geometry, states of matter, photosynthesis), each
written with a realistic arc — a misconception, a struggle, and a recovery — so the agent has
genuine signal to retrieve and reason over, including cross-student questions.

## How it works

```
Transcripts (JSON)
      │
      ▼  chunk  →  embed (SentenceTransformers, local)
      ▼
FAISS vector index
      │
      ├──►  Quick summary: semantic retrieval → LLM re-expresses the evidence
      │
      └──►  Agent: a tool-calling LangChain agent that decides when to call
            rag_search (to ground itself in real transcript evidence) and
            save_tutoring_insights_pdf (to export), returning a schema-validated report
```

The nuances that make it more than a prompt wrapper:

- **It's a genuine tool-calling agent, not a fixed pipeline.** The agent chooses when to
  retrieve evidence and when to export, rather than following hard-coded steps — the
  retrieval and PDF export are both real tools it can call.
- **Answers are grounded in evidence.** Insights are built from passages actually retrieved
  from the transcripts, not from the model's general knowledge, and the report surfaces the
  supporting quotes.
- **Output is schema-validated.** The structured report is defined by a Pydantic schema, so
  the result is a predictable object (consistent fields, constrained value ranges) that the UI
  and the PDF exporter can both rely on — and the UI renders it as a report rather than a raw
  blob, falling back to plain text if a response can't be parsed.
- **The backend is swappable by configuration alone.** Embeddings run locally; the LLM is
  chosen at runtime (open-source via Ollama, or Gemini / any OpenAI-compatible server) with no
  code changes — which is what lets the hosted demo stay free while keeping a production path open.

## What it demonstrates

Practical RAG and agent engineering: local embeddings and a FAISS vector store, a tool-calling
LangChain agent with evidence-grounded retrieval, schema-validated structured output, a
report/PDF presentation layer, and a provider-agnostic LLM backend — assembled around a
concrete, human problem rather than a toy demo.

## Built with

Python · LangChain (tools & agents) · FAISS · SentenceTransformers · Pydantic · ReportLab ·
Streamlit · Ollama / Gemini / OpenAI-compatible backends.

The sample transcripts are small and synthetic, included only to demonstrate the pipeline;
the same JSON shape (`role` / `text` / `timestamp` turns) works for real sessions.

## License

MIT — see [LICENSE](LICENSE).
