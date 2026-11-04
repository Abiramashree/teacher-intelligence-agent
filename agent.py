"""
agent.py
Tool-using LangChain agent that turns retrieved transcript context into structured,
actionable tutoring insights, and can export them as a PDF report.

The LLM is an open-source model (via llm.make_llm); no proprietary API is required.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

try:
    from langchain_core.tools import tool
except ImportError:  # pragma: no cover - older LangChain fallback
    from langchain.tools import tool

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

import config
from llm import make_llm
from rag_engine import RAGSearch
from report import save_insights_pdf


class TutoringInsights(BaseModel):
    """Structured output schema for one tutoring session's analysis."""

    topics: List[str] = Field(default_factory=list)
    student_understanding: Literal["low", "medium", "high"] = "medium"
    misconceptions: List[str] = Field(default_factory=list)
    knowledge_gaps: List[str] = Field(default_factory=list)
    questions_to_ask: List[str] = Field(default_factory=list)
    suggested_activities: List[str] = Field(default_factory=list)
    next_best_prompts: List[str] = Field(default_factory=list)
    actionable_feedback: List[str] = Field(default_factory=list)
    summary: str = ""
    difficulty_rating: int = Field(3, ge=1, le=5)
    blooms_level: Literal[
        "Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"
    ] = "Understand"
    sentiment: Literal["frustrated", "neutral", "confident", "excited", "confused"] = "neutral"
    key_quotes: List[str] = Field(default_factory=list)
    confidence: float = Field(0.6, ge=0, le=1)


DEFAULT_SYSTEM_PROMPT = (
    "You are a tutoring analytics coach. Read student-tutor transcripts and produce precise, "
    "actionable insights to plan the next step. Be diagnostic, concise, and concrete. "
    "Identify misconceptions and gaps; propose targeted questions and activities. "
    "Always return output that follows the requested format."
)

_rag: Optional[RAGSearch] = None


def get_rag() -> RAGSearch:
    """Lazily initialize the shared RAGSearch instance, so importing this module does
    not build the index or load a model until an insight is actually requested."""
    global _rag
    if _rag is None:
        _rag = RAGSearch(
            persist_dir=config.PERSIST_DIR,
            embedding_model=config.EMBED_MODEL,
            llm_model=config.SUMMARY_MODEL,
        )
    return _rag


@tool("rag_search", return_direct=False)
def rag_search(query: str) -> str:
    """Retrieve the top transcript chunks for a query from the FAISS-backed vector store."""
    results = get_rag().vectorstore.query(query, top_k=config.TOP_K)
    texts = [r.get("metadata", {}).get("text", "") for r in results if r.get("metadata")]
    joined = "\n\n".join(t for t in texts if t)
    return joined if joined.strip() else "[EMPTY]"


@tool("save_tutoring_insights_pdf", return_direct=False)
def save_tutoring_insights_pdf(payload: str) -> str:
    """Save TutoringInsights to a PDF. Payload is a JSON string with keys:
    insights (dict, required), retrieved_chunks (list[str], optional), output_path (str, optional).
    """
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return "[ERROR] Payload must be valid JSON."

    insights = data.get("insights")
    if not isinstance(insights, dict):
        return "[ERROR] 'insights' must be a JSON object."

    try:
        path = save_insights_pdf(
            insights,
            data.get("retrieved_chunks", []),
            data.get("output_path", "tutoring_insights.pdf"),
        )
        return f"[INFO] PDF saved: {path}"
    except Exception as exc:  # noqa: BLE001 - report failure back to the agent
        return f"[ERROR] Failed to save PDF: {exc}"


def build_agent_executor() -> AgentExecutor:
    """Build the tool-calling agent: it calls rag_search to ground itself in real
    transcript context, then optionally calls save_tutoring_insights_pdf to export a report."""
    parser = PydanticOutputParser(pydantic_object=TutoringInsights)
    system_prompt = (
        DEFAULT_SYSTEM_PROMPT
        + "\n\nProcess:\n"
        "1) First call the `rag_search` tool with the user's query to fetch transcript excerpts.\n"
        "2) Use those excerpts to produce insights that strictly match the requested format.\n"
        "3) If the user asks to save as PDF, call `save_tutoring_insights_pdf` with a JSON payload "
        "containing the insights, retrieved_chunks (optional), and output_path (optional).\n"
        "Return only the formatted insights unless explicitly asked to save."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt + "\n{format_instructions}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    llm = make_llm(model=config.AGENT_MODEL, temperature=0)
    tools = [rag_search, save_tutoring_insights_pdf]
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)
