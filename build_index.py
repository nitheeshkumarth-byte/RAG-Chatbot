"""
build_index.py — run this once to chunk your docs, embed them, and save the index.

Usage:
    python build_index.py                  # uses sample_docs/
    python build_index.py my_docs_folder    # uses your own folder instead
"""

import os
import sys

from dotenv import load_dotenv

from rag.ingest import build_chunks
from rag.store import VectorStore

load_dotenv()  # needed now: nested-table extraction may call Gemini for vision

INDEX_PATH = os.environ.get("INDEX_PATH", "index.pkl")


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "sample_docs"

    print(f"Loading and chunking documents from '{folder}'...")
    chunks = build_chunks(folder, chunk_size=300, overlap=50)
    table_count = sum(1 for c in chunks if c.chunk_type == "table")
    text_count = len(chunks) - table_count
    print(f"Created {len(chunks)} chunks ({text_count} text, {table_count} table).")

    print("Embedding chunks (this downloads a small model on first run)...")
    store = VectorStore()
    store.add(chunks)

    store.save(INDEX_PATH)
    print(f"Saved index to '{INDEX_PATH}'. Now run: python query.py \"your question\"")


if __name__ == "__main__":
    main()
