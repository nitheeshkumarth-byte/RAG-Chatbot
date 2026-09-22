"""
query.py — ask a question, retrieve relevant chunks, and get a grounded answer.

Usage:
    python query.py "What is HNSW used for?"

Uses whichever generation backend GENERATION_BACKEND in .env points to
(gemini or bedrock) — see .env.example.
"""

import sys
import os

from dotenv import load_dotenv

from rag.generate import generate_answer
from rag.store import VectorStore

load_dotenv()

INDEX_PATH = os.environ.get("INDEX_PATH", "index.pkl")
TOP_K = 3


def build_prompt(question: str, results) -> str:
    context = "\n\n".join(
        f"[Source: {r.chunk.source}]\n{r.chunk.text}" for r in results
    )
    return f"""Answer the question using ONLY the context below. If the context
doesn't contain the answer, say so explicitly instead of guessing.

Context:
{context}

Question: {question}

Answer:"""


def main():
    if len(sys.argv) < 2:
        print('Usage: python query.py "your question here"')
        return

    question = sys.argv[1]

    print("Loading index...")
    store = VectorStore()
    store.load(INDEX_PATH)

    if not store.chunks:
        print("Index is empty — add a document first (build_index.py, or the web UI).")
        return

    print(f"Retrieving top {TOP_K} chunks for: {question!r}\n")
    results = store.search(question, top_k=TOP_K)

    for r in results:
        print(f"  [{r.score:.3f}] {r.chunk.source} (chunk {r.chunk.chunk_id})")
    print()

    prompt = build_prompt(question, results)

    try:
        answer = generate_answer(prompt)
    except Exception as e:
        print(f"ERROR generating answer: {e}")
        return

    print("--- Answer ---")
    print(answer)


if __name__ == "__main__":
    main()
