"""
ingest.py — load documents from a folder and split them into overlapping chunks.

This is deliberately simple (no external chunking library) so you can see
exactly what's happening. Swap this out later for smarter, structure-aware
chunking once you understand the basics.
"""

import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import trafilatura
from pypdf import PdfReader


@dataclass
class Chunk:
    text: str
    source: str      # which file this chunk came from
    chunk_id: int     # position of this chunk within the file


def extract_pdf_text(path: str) -> str:
    """Pull text out of every page of a PDF and join them.

    Note: this only works for text-based PDFs. Scanned/image-only PDFs need
    OCR first (e.g. pytesseract) — extract_text() will return empty strings
    for those pages instead of raising an error, so a PDF that yields
    suspiciously little text is worth checking by eye.
    """
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len] or "page"


def extract_url_text(url: str) -> Tuple[str, str]:
    """Fetch a web page and pull out just its article text (no nav/ads/comments).

    Returns (title, text). Raises ValueError if the page can't be fetched or
    no readable article content could be found in it.
    """
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"Could not download {url}")

    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"Could not find readable article content at {url}")

    metadata = trafilatura.extract_metadata(downloaded)
    title = metadata.title if metadata and metadata.title else url
    return title, text


def load_documents(folder_path: str) -> List[tuple[str, str]]:
    """Load all .md, .txt, and .pdf files from a folder. Returns [(filename, text), ...]."""
    docs = []
    for filename in sorted(os.listdir(folder_path)):
        path = os.path.join(folder_path, filename)
        if filename.endswith((".md", ".txt")):
            with open(path, "r", encoding="utf-8") as f:
                docs.append((filename, f.read()))
        elif filename.endswith(".pdf"):
            docs.append((filename, extract_pdf_text(path)))
    return docs


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks by word count.

    chunk_size: target number of words per chunk
    overlap: number of words repeated between consecutive chunks, so a sentence
             sitting on a chunk boundary doesn't lose all its context.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start = end - overlap  # step forward, but re-include the overlap region
    return chunks


def build_chunks(folder_path: str, chunk_size: int = 500, overlap: int = 50) -> List[Chunk]:
    """Load every document in a folder and turn it into a flat list of Chunk objects."""
    all_chunks = []
    for filename, text in load_documents(folder_path):
        all_chunks.extend(chunk_single_document(filename, text, chunk_size, overlap))
    return all_chunks


def chunk_single_document(filename: str, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Chunk]:
    """Chunk one already-loaded document's text into a list of Chunk objects.

    Used both by build_chunks() above (looping over a whole folder) and by
    the API's incremental /ingest endpoints, which add one new file or URL
    to an existing index without re-processing everything else.
    """
    pieces = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    return [Chunk(text=piece, source=filename, chunk_id=i) for i, piece in enumerate(pieces)]


if __name__ == "__main__":
    # Quick sanity check when run directly: python -m rag.ingest
    chunks = build_chunks("sample_docs")
    print(f"Loaded {len(chunks)} chunks from sample_docs/")
    for c in chunks[:2]:
        print(f"\n--- {c.source} chunk {c.chunk_id} ---")
        print(c.text[:200], "...")
