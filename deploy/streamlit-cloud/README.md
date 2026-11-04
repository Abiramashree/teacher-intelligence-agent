# Hosted demo — Streamlit Community Cloud + OpenRouter (free, open-source models)

Free, always-available, public URL. The app runs on Streamlit Community Cloud; the
open-weight LLM runs on **OpenRouter's free tier** via its OpenAI-compatible endpoint.
Embeddings run locally inside the app.

> The code is backend-agnostic (`LLM_PROVIDER=openai` + a base URL), so you can point it
> at any OpenAI-compatible provider — OpenRouter, Groq, Cerebras, a LiteLLM gateway,
> vLLM, etc. — by changing the secrets, with no code change.

## One-time setup

1. Make sure this repo is pushed to GitHub (it is):
   <https://github.com/Abiramashree/teacher-intelligence-agent>

2. Create a free OpenRouter key at <https://openrouter.ai/keys> (starts with `sk-or-`).

3. **Enable free models:** at <https://openrouter.ai/settings/privacy>, turn on the
   free-model / prompt-training data policy — otherwise `:free` models return 404.

4. Go to <https://share.streamlit.io> → sign in with GitHub → **Create app** →
   **Deploy a public app from GitHub**:
   - Repository: `Abiramashree/teacher-intelligence-agent`
   - Branch: `main`
   - Main file path: `app.py`

5. Open **Advanced settings → Secrets** (or **Settings → Secrets** on an existing app)
   and paste:

   ```toml
   LLM_PROVIDER = "openai"
   OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
   OPENAI_API_KEY = "sk-or-your_key_here"
   SUMMARY_MODEL = "google/gemma-4-26b-a4b-it:free"
   AGENT_MODEL = "google/gemma-4-26b-a4b-it:free"
   ```

6. Deploy. First build takes a few minutes (installing deps, downloading the local
   embedding model, building the FAISS index). After that the app is live at a public
   `*.streamlit.app` URL.

## Alternative backend: native Google Gemini (proprietary)

Gemini's free tier is reliable and its tool-calling (the PDF/agent button) works well,
but it is **not an open-source model**. Use it instead of the OpenRouter block above:

```toml
LLM_PROVIDER = "gemini"
GOOGLE_API_KEY = "AIza_your_key"
SUMMARY_MODEL = "gemini-2.5-flash"
AGENT_MODEL = "gemini-2.5-flash"
```

Get a key at <https://aistudio.google.com/apikey>. This uses the native `langchain-
google-genai` client, not the OpenAI-compat endpoint — important because Gemini 3.x
"thinking" models return a thought signature that must be replayed through the agent's
tool loop. The OpenAI-compat shim drops it and the PDF step 400s; the native client
handles it.

## Notes

- **OpenRouter free tier** is roughly ~50 requests/day until you add a one-time $10
  (then ~1000/day). Free model IDs rotate over time; list current ones that support tool
  calling with:
  ```bash
  curl -s https://openrouter.ai/api/v1/models | python3 -c "import sys,json;[print(m['id']) for m in json.load(sys.stdin)['data'] if m['id'].endswith(':free') and 'tools' in (m.get('supported_parameters') or [])]"
  ```
- The "Quick RAG summary" button is a single call; the "full agent + PDF" button needs a
  model that supports tool calling (the one above does). If a free model stops supporting
  tools, switch `AGENT_MODEL` to `openrouter/free` (auto-selects a capable free model).
- Streamlit Community Cloud apps sleep after inactivity and wake on the next visit.
- Keep your key in Streamlit **Secrets**, never in the repo.
