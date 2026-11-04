"""
scripts/build_index.py
Pre-build the FAISS index from the transcripts in DATA_DIR.

Run this once before a demo so the first query in the UI is fast:
    python scripts/build_index.py
"""
import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from rag_engine import FaissVectorStore, load_all_documents  # noqa: E402


def main() -> None:
    docs = load_all_documents(config.DATA_DIR)
    if not docs:
        raise SystemExit(f"No transcripts found in {config.DATA_DIR!r}.")
    store = FaissVectorStore()
    store.build_from_documents(docs)
    print(f"Built FAISS index from {len(docs)} transcript turns -> {config.PERSIST_DIR}/")


if __name__ == "__main__":
    main()
