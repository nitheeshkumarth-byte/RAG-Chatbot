"""
store.py — embed chunks and do similarity search over them.

Uses sentence-transformers for local, free embeddings (no API key needed) and
plain numpy for the similarity search. No vector database yet on purpose —
at a few hundred chunks, a numpy array is plenty fast, and skipping the
database lets you see exactly what "search" means before adding a library
that hides it from you.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from .ingest import Chunk

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, good enough to start with


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        self.chunks: List[Chunk] = []
        self.embeddings: np.ndarray | None = None

    def add(self, chunks: List[Chunk]) -> None:
        """Embed a list of chunks and store them."""
        texts = [c.text for c in chunks]
        new_embeddings = self.model.encode(texts, normalize_embeddings=True)

        self.chunks.extend(chunks)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """Return the top_k chunks most similar to the query."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        query_vec = self.model.encode([query], normalize_embeddings=True)[0]

        # Embeddings are normalized, so dot product == cosine similarity.
        scores = self.embeddings @ query_vec

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [SearchResult(chunk=self.chunks[i], score=float(scores[i])) for i in top_indices]

    def sources(self) -> List[str]:
        """Return the unique, ordered list of source filenames currently indexed."""
        seen = []
        for c in self.chunks:
            if c.source not in seen:
                seen.append(c.source)
        return seen

    def remove_source(self, source: str) -> None:
        """Drop every chunk (and its embedding) that came from a given source."""
        keep_indices = [i for i, c in enumerate(self.chunks) if c.source != source]
        self.chunks = [self.chunks[i] for i in keep_indices]
        if self.embeddings is not None and keep_indices:
            self.embeddings = self.embeddings[keep_indices]
        elif not keep_indices:
            self.embeddings = None

    def save(self, path: str) -> None:
        """Persist embeddings + chunk metadata so you don't re-embed every run."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings}, f)

    def load(self, path: str) -> None:
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.embeddings = data["embeddings"]
