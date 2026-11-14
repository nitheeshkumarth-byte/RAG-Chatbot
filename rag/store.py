"""
store.py — embed chunks and do similarity search over them.

Uses sentence-transformers for local, free embeddings (no API key needed) and
plain numpy for the similarity search. No vector database yet on purpose —
at a few hundred chunks, a numpy array is plenty fast, and skipping the
database lets you see exactly what "search" means before adding a library
that hides it from you.

Models are cached per process, and each model's required query/passage
prefixes are applied automatically. E5 and BGE embed the same sentence
differently depending on whether it is a question or a document; skipping
those prefixes would make a model-comparison experiment lie about quality.
"""

import pickle
import threading
from dataclasses import dataclass
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from .ingest import Chunk

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, good enough to start with

_MODELS: dict[str, SentenceTransformer] = {}
_MODEL_LOCK = threading.Lock()


def get_model(name: str) -> SentenceTransformer:
    """Load a model once per process (downloads on first use)."""
    with _MODEL_LOCK:
        model = _MODELS.get(name)
        if model is None:
            try:
                # Try the local cache first to avoid a network round-trip.
                model = SentenceTransformer(name, local_files_only=True)
            except Exception:
                model = SentenceTransformer(name)
            _MODELS[name] = model
        return model


def _prefixes(model_name: str) -> tuple[str, str]:
    """(query_prefix, passage_prefix) required by some embedding models."""
    name = model_name.lower()
    if "e5" in name:
        return "query: ", "passage: "
    if "nomic" in name:
        return "search_query: ", "search_document: "
    if "bge" in name and "reranker" not in name:
        return "Represent this sentence for searching relevant passages: ", ""
    return "", ""


def encode_passages(model_name: str, texts: List[str]) -> np.ndarray:
    _, passage_prefix = _prefixes(model_name)
    return get_model(model_name).encode(
        [passage_prefix + t for t in texts], normalize_embeddings=True
    )


def encode_query(model_name: str, text: str) -> np.ndarray:
    query_prefix, _ = _prefixes(model_name)
    return get_model(model_name).encode([query_prefix + text], normalize_embeddings=True)[0]


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = get_model(model_name)
        self.chunks: List[Chunk] = []
        self.embeddings: np.ndarray | None = None

    def add(self, chunks: List[Chunk]) -> None:
        """Embed a list of chunks and store them."""
        if not chunks:
            return
        new_embeddings = encode_passages(self.model_name, [c.text for c in chunks])

        self.chunks.extend(chunks)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """Return the top_k chunks most similar to the query."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        query_vec = encode_query(self.model_name, query)

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
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings}, f)

    def load(self, path: str) -> None:
        """Load embeddings and chunk metadata from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data.get("chunks", [])
        self.embeddings = data.get("embeddings")
