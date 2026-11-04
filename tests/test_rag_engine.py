"""
Tests for the retrieval layer and PDF report — no LLM/network required.

These cover the parts that don't need a running model: document loading, index
build + search round-trip, and PDF generation.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_engine import FaissVectorStore, load_all_documents  # noqa: E402
from report import save_insights_pdf  # noqa: E402


def test_load_all_documents_reads_sample_data():
    docs = load_all_documents("data")
    assert len(docs) > 0
    assert hasattr(docs[0], "page_content")


def test_faiss_build_and_search_roundtrip(tmp_path):
    store = FaissVectorStore(persist_dir=str(tmp_path / "idx"))
    store.build_from_documents(load_all_documents("data"))
    results = store.query("states of matter", top_k=3)
    assert 1 <= len(results) <= 3
    assert all("metadata" in r and "distance" in r for r in results)


def test_search_requires_index():
    store = FaissVectorStore(persist_dir="does-not-exist-xyz")
    store.index = None
    with pytest.raises(ValueError):
        store.search(np.zeros((1, 384), dtype="float32"))


def test_save_insights_pdf(tmp_path):
    out = save_insights_pdf(
        {"summary": "Test", "student_understanding": "high", "topics": ["fractions"]},
        retrieved_chunks=["chunk one"],
        output_path=str(tmp_path / "out.pdf"),
    )
    assert Path(out).exists()
    assert Path(out).stat().st_size > 0
