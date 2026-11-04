#!/usr/bin/env bash
# One-command local demo on open-source models.
# Prereqs: Python 3.10+, and Ollama installed (https://ollama.com/download).
set -euo pipefail

AGENT_MODEL="${AGENT_MODEL:-llama3.1:8b}"

echo "==> Installing Python dependencies"
python3 -m pip install -q -r requirements.txt

if ! command -v ollama >/dev/null 2>&1; then
  echo "!! Ollama is not installed. Install it from https://ollama.com/download, then re-run." >&2
  exit 1
fi

echo "==> Ensuring Ollama is running"
if ! curl -fsS "${OLLAMA_BASE_URL:-http://localhost:11434}/api/tags" >/dev/null 2>&1; then
  echo "   starting 'ollama serve' in the background..."
  ollama serve >/tmp/ollama.log 2>&1 &
  sleep 3
fi

echo "==> Pulling open-source model: ${AGENT_MODEL}"
ollama pull "${AGENT_MODEL}"

echo "==> Building FAISS index"
python3 scripts/build_index.py

echo "==> Launching Streamlit demo at http://localhost:8501"
streamlit run app.py
