"""
rag_engine.py
Retrieval-Augmented Generation core for the Teacher Intelligence Agent.

Pipeline: JSON transcripts -> chunk -> embed (SentenceTransformers, local) ->
FAISS index -> semantic search -> open-source LLM summarization.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - older LangChain fallback
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from sentence_transformers import SentenceTransformer

import config
from llm import make_llm


def load_all_documents(data_dir: str = config.DATA_DIR) -> List[Any]:
    """Load tutoring transcripts (JSON) from a directory into LangChain Documents.

    Each source file is a list of {"role", "text", "timestamp"} turns.
    """
    data_path = Path(data_dir).resolve()
    documents: List[Any] = []

    for json_file in sorted(data_path.glob("**/*.json")):
        try:
            loader = JSONLoader(
                file_path=str(json_file),
                jq_schema=".[] | {role: .role, text: .text}",
                text_content=False,
            )
            documents.extend(loader.load())
        except Exception as exc:  # noqa: BLE001 - surface per-file load errors, keep going
            print(f"[ERROR] Failed to load JSON {json_file}: {exc}")

    return documents


class EmbeddingPipeline:
    """Splits transcripts into chunks and embeds them with a local SentenceTransformer."""

    def __init__(
        self,
        model_name: str = config.EMBED_MODEL,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        return splitter.split_documents(documents)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return np.asarray(embeddings, dtype="float32")


class FaissVectorStore:
    """Persistent FAISS vector store with per-vector metadata."""

    def __init__(
        self,
        persist_dir: str = config.PERSIST_DIR,
        embedding_model: str = config.EMBED_MODEL,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ):
        self.persist_dir = persist_dir
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(embedding_model)

    @property
    def _index_path(self) -> str:
        return os.path.join(self.persist_dir, "faiss.index")

    @property
    def _meta_path(self) -> str:
        return os.path.join(self.persist_dir, "metadata.pkl")

    def build_from_documents(self, documents: List[Any]) -> None:
        pipeline = EmbeddingPipeline(self.embedding_model, self.chunk_size, self.chunk_overlap)
        chunks = pipeline.chunk_documents(documents)
        embeddings = pipeline.embed_texts([c.page_content for c in chunks])
        metadatas = [{"text": c.page_content} for c in chunks]
        self.add_embeddings(embeddings, metadatas)
        self.save()

    def add_embeddings(self, embeddings: np.ndarray, metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array of shape (n_vectors, dim).")
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)

    def save(self) -> None:
        if self.index is None:
            raise ValueError("No FAISS index to save.")
        faiss.write_index(self.index, self._index_path)
        with open(self._meta_path, "wb") as fh:
            pickle.dump(self.metadata, fh)

    def load(self) -> None:
        if not (os.path.exists(self._index_path) and os.path.exists(self._meta_path)):
            raise FileNotFoundError(f"Missing index/metadata in {self.persist_dir}. Build first.")
        self.index = faiss.read_index(self._index_path)
        with open(self._meta_path, "rb") as fh:
            self.metadata = pickle.load(fh)

    def search(self, query_embedding: np.ndarray, top_k: int = config.TOP_K) -> List[Dict[str, Any]]:
        if self.index is None:
            raise ValueError("FAISS index is not loaded or built.")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_embedding.astype("float32"), top_k)
        results: List[Dict[str, Any]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx] if 0 <= idx < len(self.metadata) else None
            results.append({"index": int(idx), "distance": float(dist), "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = config.TOP_K) -> List[Dict[str, Any]]:
        query_emb = self.model.encode([query_text]).astype("float32")
        return self.search(query_emb, top_k=top_k)

    def ensure_built(self, data_dir: str = config.DATA_DIR) -> None:
        """Load an existing index, or build one from the data directory if none exists."""
        if os.path.exists(self._index_path) and os.path.exists(self._meta_path):
            self.load()
        else:
            self.build_from_documents(load_all_documents(data_dir))


def content_to_text(content) -> str:
    """Flatten an LLM response's .content to plain text.

    Thinking models (e.g. Gemini) return content as a list of blocks like
    [{"type": "text", "text": "..."}, ...] rather than a plain string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
            else:
                parts.append(str(block))
        return "".join(parts).strip()
    return str(content)


class RAGSearch:
    """Retrieves relevant transcript chunks and summarizes them with an open-source LLM."""

    def __init__(
        self,
        persist_dir: str = config.PERSIST_DIR,
        embedding_model: str = config.EMBED_MODEL,
        llm_model: str = config.SUMMARY_MODEL,
    ):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        self.vectorstore.ensure_built()
        self.llm = make_llm(model=llm_model, temperature=0)

    def search_and_summarize(self, query: str, top_k: int = config.TOP_K) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        chunks = [r["metadata"]["text"] for r in results if r.get("metadata")]
        context = "\n\n".join(chunks) if chunks else "[No relevant transcript context found.]"
        prompt = (
            "You are a tutoring analytics assistant. Using only the transcript excerpts below, "
            f"answer the question concisely and concretely.\n\nExcerpts:\n{context}\n\nQuestion: {query}"
        )
        return content_to_text(self.llm.invoke(prompt).content)
