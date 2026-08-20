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


@dataclass
class Chunk:
    text: str
    source: str          # which file this chunk came from
    chunk_id: int         # position of this chunk within the file
    chunk_type: str = "text"  # "text" or "table" — tables are kept whole,
                              # never split by chunk_text(), since breaking
                              # a table mid-row destroys the row/column
                              # relationships that make it useful at all


def extract_pdf_text(path: str) -> str:
    """Pull text out of every page of a PDF and join them.

    Note: this only works for text-based PDFs. Scanned/image-only PDFs need
    OCR first (e.g. pytesseract) — extract_text() will return empty strings
    for those pages instead of raising an error, so a PDF that yields
    suspiciously little text is worth checking by eye.

    pypdf is imported here rather than at module level, so that importing
    this file (e.g. for slugify()/extract_url_text() in api_kb.py, which
    never parses PDFs locally — the Knowledge Base does that during S3
    sync) doesn't require pypdf to be installed at all.
    """
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _looks_like_nested_table(rows: List[List[str]]) -> bool:
    """Heuristic for 'pdfplumber probably flattened a nested table into one
    cell's raw text instead of detecting the inner grid'. pdfplumber finds
    tables by looking for a regular grid of lines/whitespace; a table nested
    inside another cell usually doesn't form its own detectable grid, so it
    shows up as a single cell containing several newline-separated lines
    that are themselves a smaller table pdfplumber never noticed."""
    for row in rows:
        for cell in row:
            if cell and cell.count("\n") >= 2:
                return True
    return False


def _table_rows_to_markdown(rows: List[List[str]]) -> str:
    """Serialize a pdfplumber table (list of rows of cell strings) into a
    Markdown table. This preserves row/column relationships in a form both
    embeddable and human-readable — plain flattened text loses that
    structure entirely (a $ figure ends up with no indication of which row
    or column it belonged to)."""
    if not rows:
        return ""
    cleaned = [[(cell or "").replace("\n", " ").strip() for cell in row] for row in rows]
    header, *body = cleaned
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _describe_table_with_vision(image_bytes: bytes) -> str:
    """Fallback for tables pdfplumber can't cleanly parse — typically nested
    tables, where a cell contains another full table inside it. Renders the
    page as an image and asks a vision-capable model to transcribe it
    directly into Markdown, preserving the nested structure a flat grid
    parser can't represent.

    Always uses Gemini directly here regardless of GENERATION_BACKEND —
    the Bedrock path in rag/generate.py isn't wired for image input yet,
    and vision-request shape differs enough between providers that it
    isn't worth abstracting until a second provider is actually needed.
    """
    import base64

    from google import genai

    from rag.config import get_config

    api_key = get_config("GEMINI_API_KEY", param_name_env="GEMINI_API_KEY_PARAM")
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents=[
            {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": base64.b64encode(image_bytes).decode(),
                }
            },
            "Transcribe every table on this page into Markdown tables. If a "
            "table cell contains another table nested inside it, represent "
            "the nested table as its own separate Markdown table directly "
            "below the outer one, with a one-line heading noting which "
            "outer row/column it belongs to. Output only the Markdown "
            "tables and their headings — no other commentary.",
        ],
    )
    return response.text


def extract_tables_from_pdf(path: str, filename: str) -> List["Chunk"]:
    """Find every table in a PDF and return each as its own Chunk.

    Tries fast structured extraction (pdfplumber) first. Falls back to a
    vision model, page by page, only for tables that look nested — regular
    flat tables never pay the extra cost of an LLM call.

    pdfplumber and its vision-fallback imports are both local to this
    function for the same reason extract_pdf_text() keeps pypdf local:
    api_kb.py imports this module for slugify()/extract_url_text() and
    never calls this function, so it shouldn't need these packages
    installed at all.
    """
    import pdfplumber

    table_chunks: List[Chunk] = []
    table_counter = 0

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            if not tables:
                continue

            page_image_bytes = None  # lazily rendered only if a nested table is found

            for rows in tables:
                if not rows or not rows[0]:
                    continue

                if _looks_like_nested_table(rows):
                    if page_image_bytes is None:
                        from io import BytesIO

                        pil_image = page.to_image(resolution=200).original
                        buf = BytesIO()
                        pil_image.save(buf, format="PNG")
                        page_image_bytes = buf.getvalue()
                    try:
                        markdown = _describe_table_with_vision(page_image_bytes)
                    except Exception as e:
                        # Don't let one table's vision call kill the whole
                        # indexing run — fall back to the flattened table
                        # instead. It'll misrepresent the nested structure,
                        # but you still get a chunk instead of losing
                        # everything else that was about to be indexed.
                        print(
                            f"  Warning: vision transcription failed for a nested "
                            f"table on page {page_num} of {filename} ({e}). "
                            f"Falling back to flattened extraction for this table."
                        )
                        markdown = _table_rows_to_markdown(rows)
                else:
                    markdown = _table_rows_to_markdown(rows)

                if not markdown.strip():
                    continue

                header = [(c or "").strip() for c in rows[0]]
                context_line = f"Table from page {page_num} of {filename}, columns: {', '.join(header)}"
                table_counter += 1
                table_chunks.append(
                    Chunk(
                        text=f"{context_line}\n\n{markdown}",
                        source=filename,
                        chunk_id=table_counter,
                        chunk_type="table",
                    )
                )

    return table_chunks


def slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len] or "page"


def extract_url_text(url: str) -> Tuple[str, str]:
    """Fetch a web page and pull out just its article text (no nav/ads/comments).

    Returns (title, text). Raises ValueError if the page can't be fetched or
    no readable article content could be found in it.

    trafilatura is imported here rather than at module level, same reasoning
    as the pypdf/pdfplumber imports elsewhere in this file: code that only
    needs slugify() or PDF/table extraction shouldn't require trafilatura
    to be installed too.
    """
    import trafilatura

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
    """Load every document in a folder and turn it into a flat list of Chunk
    objects — prose chunks from chunk_single_document(), plus one extra
    Chunk per table found in any PDF (tables are extracted separately since
    they're kept whole rather than split by word count)."""
    all_chunks = []
    for filename, text in load_documents(folder_path):
        all_chunks.extend(chunk_single_document(filename, text, chunk_size, overlap))

        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            all_chunks.extend(extract_tables_from_pdf(path, filename))

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
