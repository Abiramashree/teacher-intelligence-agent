"""
llm.py
Single factory for the chat LLM, so the rest of the codebase never imports a
specific provider directly.

Default: Ollama running an open-source model locally (no API key, no cloud).
Alternative: any OpenAI-compatible endpoint via LLM_PROVIDER=openai.
"""
from __future__ import annotations

from typing import Optional

import config


def make_llm(model: Optional[str] = None, temperature: float = 0.0):
    """Return a LangChain chat model.

    Args:
        model: Model name/tag. Defaults to config.AGENT_MODEL when not given.
        temperature: Sampling temperature.
    """
    model = model or config.AGENT_MODEL

    if config.LLM_PROVIDER == "ollama":
        # Open-source models served locally by Ollama.
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            base_url=config.OLLAMA_BASE_URL,
            temperature=temperature,
        )

    if config.LLM_PROVIDER == "openai":
        # Any OpenAI-compatible server (LiteLLM gateway, vLLM, Groq, ...).
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            base_url=config.OPENAI_BASE_URL,
            api_key=config.OPENAI_API_KEY,
            temperature=temperature,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER={config.LLM_PROVIDER!r}. Use 'ollama' or 'openai'."
    )
