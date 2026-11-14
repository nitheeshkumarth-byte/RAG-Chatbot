"""
query.py — ask a question, retrieve relevant chunks, and get a grounded answer.

Usage:
    python query.py "What is HNSW used for?"

Uses whichever generation backend GENERATION_BACKEND in .env points to
(gemini or bedrock) — see .env.example. Retrieval takes the same env knobs
as the web UI: RETRIEVAL_STRATEGY, RETRIEVAL_TOP_K, HYBRID_ALPHA, MMR_LAMBDA,
RETRIEVAL_MULTI_QUERY, RERANK_MODEL, RERANK_TOP_N.
"""

import os
import sys

from dotenv import load_dotenv

from rag import pipeline, retrieval
from rag.generate import generate_answer
from rag.store import VectorStore, encode_query

load_dotenv()

INDEX_PATH = os.environ.get("INDEX_PATH", "index.pkl")
STRATEGY = os.environ.get("RETRIEVAL_STRATEGY", "vector")
TOP_K = max(1, int(os.environ.get("RETRIEVAL_TOP_K", "3")))
ALPHA = float(os.environ.get("HYBRID_ALPHA", "0.5"))
LAMBDA = float(os.environ.get("MMR_LAMBDA", "0.7"))
MULTI_QUERY = min(5, max(0, int(os.environ.get("RETRIEVAL_MULTI_QUERY", "0"))))
RERANK_MODEL = os.environ.get("RERANK_MODEL") or None
RERANK_TOP_N = max(1, int(os.environ.get("RERANK_TOP_N", str(TOP_K))))


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

    strategy = STRATEGY
    if strategy not in retrieval.RETRIEVAL_STRATEGIES:
        print(f"Unknown RETRIEVAL_STRATEGY {strategy!r}; using 'vector'.")
        strategy = "vector"

    rewrites = pipeline.rewrite_queries(question, MULTI_QUERY) if MULTI_QUERY else []
    if rewrites:
        print("Rewrites: " + " | ".join(rewrites))

    query_vec = encode_query(store.model_name, question)
    if rewrites:
        query_vec = retrieval.mean_query_vector([
            query_vec,
            *[encode_query(store.model_name, q) for q in rewrites],
        ])

    indices, details, _ = retrieval.rank(
        question,
        [c.text for c in store.chunks],
        store.embeddings,
        query_vec,
        strategy=strategy,
        top_k=TOP_K,
        alpha=ALPHA,
        lam=LAMBDA,
        bm25_query=" ".join([question] + rewrites),
        reranker=RERANK_MODEL,
        rerank_top_n=RERANK_TOP_N,
    )

    print(f"Retrieved {len(indices)} chunks for: {question!r}\n")
    for i in indices:
        scores = details[i]
        score = scores["rerank"] if scores["rerank"] is not None else scores["rank"]
        chunk = store.chunks[i]
        snippet = " ".join(chunk.text.split())
        if len(snippet) > 160:
            snippet = snippet[:160].rsplit(" ", 1)[0] + "…"
        print(f"  [{score:.3f}] {chunk.source} (chunk {chunk.chunk_id})")
        print(f"      {snippet}")

    prompt = pipeline.build_prompt(question, [store.chunks[i] for i in indices])

    try:
        answer = generate_answer(prompt)
    except Exception as e:
        print(f"ERROR generating answer: {e}")
        return

    print("\n--- Answer ---")
    print(answer)


if __name__ == "__main__":
    main()
