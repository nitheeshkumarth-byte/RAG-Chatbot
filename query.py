"""
query.py — ask a question, retrieve relevant chunks, and get a grounded answer.

Usage:
    python query.py "What is HNSW used for?"

Requires GEMINI_API_KEY to be set, either in your environment or in a .env
file in this folder (see .env.example).
"""

import sys
import os

from dotenv import load_dotenv
from google import genai

from rag.store import VectorStore

load_dotenv()

INDEX_PATH = "index.pkl"
TOP_K = 3
GEMINI_MODEL = "gemini-2.5-flash"


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

    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: set GEMINI_API_KEY in your environment or .env file first.")
        return

    print("Loading index...")
    store = VectorStore()
    store.load(INDEX_PATH)

    print(f"Retrieving top {TOP_K} chunks for: {question!r}\n")
    results = store.search(question, top_k=TOP_K)

    for r in results:
        print(f"  [{r.score:.3f}] {r.chunk.source} (chunk {r.chunk.chunk_id})")
    print()

    prompt = build_prompt(question, results)

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    print("--- Answer ---")
    print(response.text)


if __name__ == "__main__":
    main()
