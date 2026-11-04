# Hosted demo — Streamlit Community Cloud + Groq (free, open-source models)

Free, always-available, public URL. The app runs on Streamlit Community Cloud; the
open-source LLM (Llama 3.x) runs on **Groq's free API** via its OpenAI-compatible
endpoint. Embeddings run locally inside the app.

> Why Groq and not Hugging Face? HF's free serverless inference credit is tiny and runs
> out fast (HTTP 402). Groq's free tier has no such micro-credit wall and serves
> open-source Llama models with fast tool-calling. The code is backend-agnostic
> (`LLM_PROVIDER=openai` + a base URL), so you can point it at any OpenAI-compatible
> provider — HF router, a LiteLLM gateway, vLLM, etc. — by changing the secrets.

## One-time setup

1. Make sure this repo is pushed to GitHub (it is):
   <https://github.com/Abiramashree/teacher-intelligence-agent>

2. Get a free Groq API key at <https://console.groq.com/keys> (starts with `gsk_`).

3. Go to <https://share.streamlit.io> → sign in with GitHub → **Create app** →
   **Deploy a public app from GitHub**:
   - Repository: `Abiramashree/teacher-intelligence-agent`
   - Branch: `main`
   - Main file path: `app.py`

4. Open **Advanced settings → Secrets** (or **Settings → Secrets** on an existing app)
   and paste:

   ```toml
   LLM_PROVIDER = "openai"
   OPENAI_BASE_URL = "https://api.groq.com/openai/v1"
   OPENAI_API_KEY = "gsk_your_groq_key_here"
   SUMMARY_MODEL = "llama-3.1-8b-instant"
   AGENT_MODEL = "llama-3.3-70b-versatile"
   ```

5. Deploy. First build takes a few minutes (installing deps, downloading the local
   embedding model, building the FAISS index). After that the app is live at a public
   `*.streamlit.app` URL.

## Notes

- **Groq free tier** has generous rate limits (requests/min, tokens/day) — fine for a
  demo. The "Quick RAG summary" button uses the small 8B model; the full agent + PDF
  button uses the 70B model for reliable tool calling.
- Streamlit Community Cloud apps sleep after inactivity and wake on the next visit (a
  short cold start).
- Keep your key in Streamlit **Secrets**, never in the repo.
- To use a different OpenAI-compatible provider instead, just change `OPENAI_BASE_URL`,
  `OPENAI_API_KEY`, and the model names in the secrets — no code change.
