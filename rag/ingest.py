"""
ingest.py — load documents from a folder and split them into overlapping chunks.

This is deliberately simple (no external chunking library) so you can see
exactly what's happening. Swap this out later for smarter, structure-aware
chunking once you understand the basics.
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    source: str          # which file this chunk came from
    chunk_id: int         # position of this chunk within the file
    chunk_type: str = "text"  # "text" or "table" — tables are kept whole,
                              # never split by chunk_text(), since breaking
                              # a table mid-row destroys the row/column
                              # relationships that make it useful at all
    start_char: int = 0       # character offsets in the source text, so the
    end_char: int = 0         # UI can show where a chunk came from
    strategy: str = "fixed"   # which chunking strategy produced it
    parent_text: str = ""     # enclosing parent chunk, when parent-child
                              # chunking is on; retrieval uses this chunk,
                              # the prompt gets the parent
    embed_text: str = ""      # text sent to the embedding model when it
                              # differs from .text (contextual chunking)


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

    try:
        reader = PdfReader(path)
    except Exception as e:
        logger.error(f"Failed to open PDF at {path}: {e}")
        raise ValueError(f"Could not read PDF file: {e}") from e

    pages = []
    for page_idx, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception as e:
            logger.warning(f"Failed to extract text from page {page_idx + 1} of {path}: {e}")
            pages.append("")

    return "\n\n".join(pages)


def _looks_like_nested_table(rows: List[List[Any]]) -> bool:
    """Heuristic for 'pdfplumber probably flattened a nested table into one
    cell's raw text instead of detecting the inner grid'. pdfplumber finds
    tables by looking for a regular grid of lines/whitespace; a table nested
    inside another cell usually doesn't form its own detectable grid, so it
    shows up as a single cell containing several newline-separated lines
    that are themselves a smaller table pdfplumber never noticed."""
    if not rows or not isinstance(rows, (list, tuple)):
        return False
    for row in rows:
        if not isinstance(row, (list, tuple)):
            continue
        for cell in row:
            if cell is not None and isinstance(cell, str) and cell.count("\n") >= 2:
                return True
    return False


def _table_rows_to_markdown(rows: List[List[Any]]) -> str:
    """Serialize a pdfplumber table (list of rows of cell strings) into a
    Markdown table. This preserves row/column relationships in a form both
    embeddable and human-readable — plain flattened text loses that
    structure entirely.

    Defensively handles malformed tables, missing cells, empty rows, and
    unexpected cell data types.
    """
    if not rows or not isinstance(rows, (list, tuple)):
        return ""

    cleaned_rows: List[List[str]] = []
    for row in rows:
        if row is None:
            continue
        if not isinstance(row, (list, tuple)):
            row = [row]

        cleaned_row = []
        for cell in row:
            if cell is None:
                cleaned_row.append("")
            else:
                cleaned_row.append(str(cell).replace("\n", " ").strip())

        # Discard rows where every cell is completely empty
        if any(cleaned_row):
            cleaned_rows.append(cleaned_row)

    if not cleaned_rows:
        return ""

    # Normalize all rows to have the same number of columns (max columns found)
    max_cols = max(len(row) for row in cleaned_rows)
    if max_cols == 0:
        return ""

    normalized_rows = [row + [""] * (max_cols - len(row)) for row in cleaned_rows]

    # If every cell in every row is empty, there's nothing useful to emit
    if not any(cell for row in normalized_rows for cell in row):
        return ""

    header = normalized_rows[0]
    body = normalized_rows[1:]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * max_cols) + " |",
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

    Guaranteed to return a string (empty string on failure or empty response).
    """
    import base64

    from google import genai

    from rag.config import get_config

    try:
        api_key = get_config("GEMINI_API_KEY", param_name_env="GEMINI_API_KEY_PARAM")
    except Exception as e:
        logger.warning(f"Could not retrieve GEMINI_API_KEY for vision transcription: {e}")
        return ""

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    try:
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
        if response and hasattr(response, "text") and response.text:
            return response.text.strip()
        logger.warning("Gemini Vision response returned empty or None text.")
        return ""
    except Exception as e:
        logger.warning(f"Gemini Vision call failed: {e}")
        return ""


def extract_tables_from_pdf(path: str, filename: str) -> List[Chunk]:
    """Find every table in a PDF and return each as its own Chunk.

    Tries fast structured extraction (pdfplumber) first. Falls back to a
    vision model, page by page, only for tables that look nested — regular
    flat tables never pay the extra cost of an LLM call.

    Handles rendering errors, vision failures, malformed tables, and ensures
    individual table failures do not crash document ingestion.
    """
    import pdfplumber

    logger.info(f"Starting table extraction for {filename}")
    table_chunks: List[Chunk] = []
    table_counter = 0

    try:
        pdf = pdfplumber.open(path)
    except Exception as e:
        logger.error(f"Failed to open PDF for table extraction {path}: {e}")
        return []

    with pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.extract_tables()
            except Exception as e:
                logger.warning(f"Error extracting tables on page {page_num} of {filename}: {e}")
                continue

            if not tables:
                continue

            page_image_bytes = None  # lazily rendered only if a nested table is found

            for table_idx, rows in enumerate(tables, start=1):
                if not rows or not isinstance(rows, (list, tuple)):
                    continue

                markdown = ""
                is_nested = _looks_like_nested_table(rows)

                if is_nested:
                    logger.info(
                        f"Detected nested table {table_idx} on page {page_num} of {filename}. Attempting Vision extraction."
                    )
                    try:
                        if page_image_bytes is None:
                            from io import BytesIO

                            pil_image = page.to_image(resolution=200).original
                            buf = BytesIO()
                            pil_image.save(buf, format="PNG")
                            page_image_bytes = buf.getvalue()

                        markdown = _describe_table_with_vision(page_image_bytes) or ""
                    except Exception as e:
                        logger.warning(
                            f"Vision transcription failed for nested table {table_idx} on page {page_num} of {filename}: {e}"
                        )
                        markdown = ""

                    # If vision extraction failed or returned empty content, fall back to structured extraction
                    if not isinstance(markdown, str) or not markdown.strip():
                        logger.info(
                            f"Vision extraction returned empty content for table {table_idx} on page {page_num} of {filename}. "
                            "Falling back to structured table extraction."
                        )
                        try:
                            markdown = _table_rows_to_markdown(rows) or ""
                        except Exception as e:
                            logger.warning(
                                f"Structured fallback extraction failed for table {table_idx} on page {page_num} of {filename}: {e}"
                            )
                            markdown = ""
                else:
                    try:
                        markdown = _table_rows_to_markdown(rows) or ""
                    except Exception as e:
                        logger.warning(
                            f"Structured extraction failed for table {table_idx} on page {page_num} of {filename}: {e}"
                        )
                        markdown = ""

                # If both extraction methods fail or return empty content, skip this table
                if not isinstance(markdown, str) or not markdown.strip():
                    logger.warning(
                        f"Skipping table {table_idx} on page {page_num} of {filename} because extraction returned empty content."
                    )
                    continue

                # Safely construct context line
                first_row = rows[0] if (rows and isinstance(rows[0], (list, tuple))) else []
                header_cols = [str(c).strip() for c in first_row if c is not None and str(c).strip()]
                columns_label = ", ".join(header_cols) if header_cols else "unnamed"
                context_line = f"Table from page {page_num} of {filename}, columns: {columns_label}"

                table_counter += 1
                table_chunks.append(
                    Chunk(
                        text=f"{context_line}\n\n{markdown.strip()}",
                        source=filename,
                        chunk_id=table_counter,
                        chunk_type="table",
                    )
                )

    logger.info(f"Completed table extraction for {filename}: {len(table_chunks)} tables extracted")
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
    """Load all .md, .txt, and .pdf files from a folder. Returns [(filename, text), ...].

    Files that fail to load are logged and skipped rather than aborting the whole batch.
    """
    docs = []
    for filename in sorted(os.listdir(folder_path)):
        path = os.path.join(folder_path, filename)
        if filename.endswith((".md", ".txt")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    docs.append((filename, f.read()))
                logger.info(f"Loaded text file: {filename}")
            except Exception as e:
                logger.warning(f"Failed to read {filename}: {e}")
        elif filename.endswith(".pdf"):
            try:
                text = extract_pdf_text(path)
                docs.append((filename, text))
                logger.info(f"Extracted text from PDF: {filename}")
            except Exception as e:
                logger.warning(f"Failed to extract text from PDF {filename}: {e}")
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

    # Clamp so the step (chunk_size - overlap) is always positive.
    overlap = max(0, min(overlap, chunk_size - 1))
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
            try:
                table_chunks = extract_tables_from_pdf(path, filename)
                all_chunks.extend(table_chunks)
            except Exception as e:
                logger.warning(f"Table extraction failed for {filename}: {e}")
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
