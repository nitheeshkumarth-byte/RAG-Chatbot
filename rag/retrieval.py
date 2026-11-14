"""
retrieval.py — scoring and selection strategies used by the experiment runner.

vector  : cosine similarity over the cached embeddings
bm25     : keyword scoring, no external dependency
hybrid   : min-max normalized vector + bm25 scores, blended by alpha
mmr      : maximal marginal relevance over the vector candidates (diversity)

Reranking is optional and independent of the strategy: it re-scores the
candidate set with a cross-encoder and keeps the best N.
"""

import math
import re
import threading
import time
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

RETRIEVAL_STRATEGIES = ("vector", "bm25", "hybrid", "mmr")

_TOKEN = re.compile(r"[a-z0-9]+")
_RERANKERS = {}
_RERANK_LOCK = threading.Lock()


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    """Okapi BM25 with the usual defaults; small enough to keep in-tree."""

    def __init__(self, texts: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(t) for t in texts]
        self.n = len(self.docs)
        self.lengths = [len(d) for d in self.docs]
        self.avgdl = (sum(self.lengths) / self.n) if self.n else 0.0
        self.tfs = [Counter(d) for d in self.docs]
        df = Counter()
        for doc in self.docs:
            df.update(set(doc))
        self.idf = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n)
        if not self.n:
            return out
        for term in tokenize(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tfs):
                freq = tf.get(term, 0)
                if not freq:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avgdl or 1))
                out[i] += idf * freq * (self.k1 + 1) / (freq + norm)
        return out


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def top_indices(scores: np.ndarray, n: int) -> List[int]:
    return [int(i) for i in np.argsort(scores)[::-1][:n]]


def mean_query_vector(vectors: List[np.ndarray]) -> np.ndarray:
    """Average several query embeddings into one unit vector (multi-query)."""
    mean = np.vstack(vectors).mean(axis=0)
    return mean / (np.linalg.norm(mean) or 1.0)


def mmr(query_vec: np.ndarray, embeddings: np.ndarray, candidates: List[int],
        top_k: int, lam: float = 0.7) -> List[int]:
    """Pick top_k candidates balancing query relevance and diversity."""
    cand = list(candidates)
    if not cand:
        return []
    query_sims = embeddings[cand] @ query_vec
    selected: List[int] = []

    while cand and len(selected) < top_k:
        if not selected:
            best = int(np.argmax(query_sims))
        else:
            redundancy = (embeddings[cand] @ embeddings[selected].T).max(axis=1)
            score = lam * query_sims - (1 - lam) * redundancy
            best = int(np.argmax(score))
        selected.append(cand.pop(best))
        query_sims = np.delete(query_sims, best)
    return selected


def rerank(query: str, texts: List[str], indices: List[int],
           model_name: str, top_n: int) -> tuple[List[int], List[float]]:
    """Re-score candidate chunks with a cross-encoder. Returns (indices, scores)."""
    from sentence_transformers import CrossEncoder

    with _RERANK_LOCK:
        model = _RERANKERS.get(model_name)
        if model is None:
            model = CrossEncoder(model_name)
            _RERANKERS[model_name] = model

    pairs = [(query, texts[i]) for i in indices]
    raw = model.predict(pairs)
    scores = np.asarray(raw, dtype=float).reshape(-1)
    order = np.argsort(scores)[::-1][:top_n]
    return [indices[int(k)] for k in order], [float(scores[int(k)]) for k in order]


def rank(query: str, texts: List[str], embeddings: np.ndarray, query_vec: np.ndarray,
         *, strategy: str = "vector", top_k: int = 4, alpha: float = 0.5,
         lam: float = 0.7, candidate_pool: int = 20,
         allowed: Optional[np.ndarray] = None,
         bm25_query: Optional[str] = None,
         reranker: Optional[str] = None, rerank_top_n: int = 3
         ) -> tuple[List[int], Dict[int, Dict[str, Optional[float]]], float]:
    """Rank chunks for one query with any strategy, optionally reranking.

    Shared by the experiment runner and the live /query path so both score
    identically. `allowed` masks chunks out of the ranking (metadata filter)
    without changing the scores of the ones that remain.

    Returns (final indices, per-index raw scores, rerank milliseconds). The
    score dict has keys vector, bm25, rank and rerank for every final index.
    """
    if strategy not in RETRIEVAL_STRATEGIES:
        raise ValueError(f"Unknown retrieval strategy '{strategy}'.")
    if allowed is None:
        allowed = np.ones(len(texts), dtype=bool)

    vector_scores = embeddings @ query_vec
    bm25_scores = None
    if strategy in ("bm25", "hybrid"):
        # Multi-query keeps the original question for reranking but lets BM25
        # see the union of all query terms.
        bm25_scores = BM25(texts).scores(bm25_query or query)

    if strategy == "bm25":
        ranked = bm25_scores
    elif strategy == "hybrid":
        ranked = alpha * minmax(vector_scores) + (1 - alpha) * minmax(bm25_scores)
    else:  # vector and mmr both start from cosine similarity
        ranked = vector_scores
    if not allowed.all():
        ranked = np.where(allowed, ranked, -np.inf)

    candidates = [
        i for i in top_indices(ranked, min(candidate_pool, len(texts)))
        if allowed[i]
    ]
    if strategy == "mmr":
        final = mmr(query_vec, embeddings, candidates, top_k, lam)
        rerank_pool = final
    else:
        final = candidates[:top_k]
        rerank_pool = candidates

    started = time.perf_counter()
    rerank_values: Dict[int, float] = {}
    if reranker and rerank_pool:
        final, values = rerank(
            query, texts, rerank_pool, reranker, min(rerank_top_n, len(rerank_pool))
        )
        rerank_values = dict(zip(final, values))
    rerank_ms = (time.perf_counter() - started) * 1000

    final = final[:top_k]
    details = {
        i: {
            "vector": float(vector_scores[i]),
            "bm25": float(bm25_scores[i]) if bm25_scores is not None else None,
            "rank": float(ranked[i]),
            "rerank": rerank_values.get(i),
        }
        for i in final
    }
    return final, details, rerank_ms
