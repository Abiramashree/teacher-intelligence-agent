# Hosted demo — Streamlit Community Cloud + Hugging Face serverless

Free, always-available, public URL. The app runs on Streamlit Community Cloud; the
open-source LLM runs on Hugging Face's serverless Inference Providers (called through
their OpenAI-compatible router). Embeddings run locally inside the app.

> Why not a Hugging Face Space? Free HF Spaces now only allow *Static* Spaces — Docker/
> Gradio/Streamlit Spaces that run a backend require a PRO subscription. The Docker Space
> in [`../hf-space/`](../hf-space/) still works if you have HF PRO.

## One-time setup

1. Make sure this repo is pushed to GitHub (it is):
   <https://github.com/Abiramashree/teacher-intelligence-agent>

2. Get a Hugging Face token (read access is enough) from
   <https://huggingface.co/settings/tokens>.

3. Go to <https://share.streamlit.io> → sign in with GitHub → **Create app** →
   **Deploy a public app from GitHub**:
   - Repository: `Abiramashree/teacher-intelligence-agent`
   - Branch: `main`
   - Main file path: `app.py`

4. Before (or right after) deploying, open **Advanced settings → Secrets** (or
   **Settings → Secrets** on an existing app) and paste:

   ```toml
   LLM_PROVIDER = "openai"
   OPENAI_BASE_URL = "https://router.huggingface.co/v1"
   OPENAI_API_KEY = "hf_your_token_here"
   SUMMARY_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
   AGENT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
   ```

5. Deploy. First build takes a few minutes (installing deps, downloading the local
   embedding model, building the FAISS index). After that the app is live at a public
   `*.streamlit.app` URL.

## Notes

- **Free inference credits are limited.** Hugging Face gives free accounts a small
  monthly credit for serverless inference — fine for a light demo. Heavy use needs HF
  PRO or pay-as-you-go. The "Quick RAG summary" button is the cheapest path.
- **Tool calling (the full agent + PDF button)** depends on the chosen model/provider
  supporting OpenAI-style tools. If it errors, the RAG summary still works; you can also
  switch `AGENT_MODEL` to another instruct model the router supports.
- Streamlit Community Cloud apps sleep after a period of inactivity and wake on the next
  visit (a short cold start) — the same always-available-but-sleeps behaviour as free HF.
- Keep your token in Streamlit **Secrets**, never in the repo.
