"""
config.py
Central, environment-driven configuration for the Teacher Intelligence Agent.

Everything here can be overridden with environment variables (see .env.example),
so the same code runs against a local open-source stack (Ollama) in a demo and a
hosted OpenAI-compatible gateway in production without edits.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# --- Data & index locations -------------------------------------------------
DATA_DIR = os.getenv("DATA_DIR", "data")
PERSIST_DIR = os.getenv("PERSIST_DIR", "faiss_store")

# --- Retrieval / chunking ---------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")  # open-source, runs locally on CPU
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))

# --- LLM backend ------------------------------------------------------------
# Default backend is Ollama serving open-source models locally, so the demo needs
# no API key and no third-party service. Set LLM_PROVIDER=openai to point at any
# OpenAI-compatible endpoint instead (LiteLLM gateway, vLLM, Groq, etc.).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Model used for quick retrieval summaries (small/fast) and for the agent.
# Both default to an 8B open-source model that fits comfortably in ~16GB and
# supports tool calling.
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "llama3.1:8b")
AGENT_MODEL = os.getenv("AGENT_MODEL", "llama3.1:8b")

# Only used when LLM_PROVIDER=openai (OpenAI-compatible servers).
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:4000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "not-needed")
