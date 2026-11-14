"""
api.py — small FastAPI server that wraps the RAG pipeline for the browser UI.

Run:
    uvicorn api:app --reload

Then open http://localhost:8000 in your browser.
"""

import logging
import os
import re
import shutil

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag import pipeline, retrieval
from rag.chunking import STRATEGIES
from rag.generate import generate_answer
from rag.ingest import chunk_single_document, extract_pdf_text, extract_tables_from_pdf, extract_url_text, slugify
from rag.retrieval import RETRIEVAL_STRATEGIES
from rag.store import VectorStore, encode_query

# ---------------------------------------------------------------------------
# Logging setup — module-level, so all log calls share the same logger name.
# Configure level/handlers in your launch script or via PYTHONLOGLEVEL env var.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()  # reads .env into os.environ (local dev only; ignored if vars already set)

INDEX_PATH = os.environ.get("INDEX_PATH", "index.pkl")
DOCS_FOLDER = os.environ.get("DOCS_FOLDER", "sample_docs")
TOP_K = 3
# Live /query retrieval. The defaults reproduce the original vector/top-3
# search; set e.g. RETRIEVAL_STRATEGY=hybrid, RERANK_MODEL=... to use a
# lab-winning config without touching code.
RETRIEVAL_STRATEGY = os.environ.get("RETRIEVAL_STRATEGY", "vector")
RETRIEVAL_TOP_K = max(1, int(os.environ.get("RETRIEVAL_TOP_K", str(TOP_K))))
HYBRID_ALPHA = float(os.environ.get("HYBRID_ALPHA", "0.5"))
MMR_LAMBDA = float(os.environ.get("MMR_LAMBDA", "0.7"))
RETRIEVAL_MULTI_QUERY = min(5, max(0, int(os.environ.get("RETRIEVAL_MULTI_QUERY", "0"))))
RERANK_MODEL = os.environ.get("RERANK_MODEL") or None
RERANK_TOP_N = max(1, int(os.environ.get("RERANK_TOP_N", str(RETRIEVAL_TOP_K))))
if RETRIEVAL_STRATEGY not in RETRIEVAL_STRATEGIES:
    logger.warning(
        f"Unknown RETRIEVAL_STRATEGY {RETRIEVAL_STRATEGY!r}; falling back to 'vector'."
    )
    RETRIEVAL_STRATEGY = "vector"
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB guard (UploadFile streams, so this is a soft check)

app = FastAPI(title="RAG API")

os.makedirs(DOCS_FOLDER, exist_ok=True)

# Load the index once at startup rather than on every request — embedding
# the model itself takes a second or two, we don't want to pay that cost
# per query. If no index has been built yet, start empty: the UI can add
# the first document straight from the browser instead of requiring
# build_index.py to run first.
store = VectorStore()
try:
    store.load(INDEX_PATH)
    logger.info(f"Loaded existing index from {INDEX_PATH!r} ({len(store.chunks)} chunks)")
except FileNotFoundError:
    logger.info(f"No existing index found at {INDEX_PATH!r}. Starting with an empty store.")
except Exception as e:
    logger.warning(f"Could not load index from {INDEX_PATH!r}: {e}. Starting with an empty store.")

# Serve everything in static/ (our single-page frontend) at the root URL.
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Shared exception handler — ensures unexpected 500s always return JSON
# (not FastAPI's default plain-text "Internal Server Error"), so the
# frontend's response.json() call never blows up.
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred. Please try again."},
    )


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str


class SourceRef(BaseModel):
    source: str
    chunk_id: int
    score: float
    snippet: str
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceRef] = Field(default_factory=list)


class UrlIngestRequest(BaseModel):
    url: str


class IngestResponse(BaseModel):
    filename: str
    chunks_added: int
    total_chunks: int


class ExperimentRequest(BaseModel):
    question: str
    docs: list[str] = Field(default_factory=list)
    configs: list[dict] = Field(default_factory=list)
    generate: bool = True
    expected_keywords: list[str] = Field(default_factory=list)
    expected_snippets: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_filename(raw: str) -> str:
    """Strip path separators and control characters from an uploaded filename.

    Prevents path-traversal attacks like '../../etc/passwd'.
    Returns just the basename with only safe characters kept.
    """
    # Take only the basename (removes any directory component)
    name = os.path.basename(raw)
    # Replace anything that isn't alphanumeric, dot, dash, underscore, or space
    name = re.sub(r"[^\w.\- ]", "_", name)
    name = name.strip()
    return name or "upload"


def _add_chunks_to_index(chunks) -> None:
    """Embed a new batch of chunks, add them to the in-memory store, and
    persist the updated index to disk so it survives a server restart."""
    # Re-ingesting the same file must replace it, not append a second copy.
    for source in {c.source for c in chunks}:
        if source in store.sources():
            store.remove_source(source)
    store.add(chunks)
    store.save(INDEX_PATH)
    logger.info(f"Index updated and saved to {INDEX_PATH!r} ({len(store.chunks)} total chunks)")


def _live_search(question: str):
    """(chunk, score) pairs from the live index, ranked like the lab."""
    rewrites = (
        pipeline.rewrite_queries(question, RETRIEVAL_MULTI_QUERY)
        if RETRIEVAL_MULTI_QUERY else []
    )
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
        strategy=RETRIEVAL_STRATEGY,
        top_k=RETRIEVAL_TOP_K,
        alpha=HYBRID_ALPHA,
        lam=MMR_LAMBDA,
        bm25_query=" ".join([question] + rewrites),
        reranker=RERANK_MODEL,
        rerank_top_n=RERANK_TOP_N,
    )
    results = []
    for i in indices:
        scores = details[i]
        score = scores["rerank"] if scores["rerank"] is not None else scores["rank"]
        results.append((store.chunks[i], score))
    return results


def _snippet(text: str, limit: int = 240) -> str:
    """Short single-line excerpt for a source card."""
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[:limit].rsplit(" ", 1)[0] + "…"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


@app.get("/lab")
def serve_lab():
    return FileResponse("static/lab.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("static/favicon.svg", media_type="image/svg+xml")


@app.get("/health")
def health():
    """Health-check endpoint. Returns 200 + JSON when the server is up."""
    return {
        "status": "healthy",
        "service": "RAG API",
        "indexed_chunks": len(store.chunks),
    }


@app.get("/sources")
def list_sources():
    """Return the distinct source filenames currently in the index, so the
    UI can show the user what's already been indexed."""
    return {"sources": store.sources(), "total_chunks": len(store.chunks)}


@app.delete("/sources/{filename}")
def delete_source(filename: str):
    """Remove one indexed source (and its chunks/embeddings) from the store."""
    if filename not in store.sources():
        raise HTTPException(404, f"'{filename}' isn't currently indexed.")
    store.remove_source(filename)
    store.save(INDEX_PATH)
    logger.info(f"Removed source {filename!r} from index ({len(store.chunks)} chunks remaining)")
    return {"removed": filename, "total_chunks": len(store.chunks)}


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    # --- Validate filename & extension ----------------------------------------
    raw_name = file.filename or ""
    if not raw_name:
        raise HTTPException(400, "No filename provided.")

    safe_name = _safe_filename(raw_name)
    ext = os.path.splitext(safe_name)[1].lower()

    if not ext:
        raise HTTPException(400, "File has no extension. Use .pdf, .txt, or .md.")
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: .pdf, .txt, .md.")

    logger.info(f"File upload started: {safe_name!r} (original: {raw_name!r})")

    # --- Save to disk ----------------------------------------------------------
    dest_path = os.path.join(DOCS_FOLDER, safe_name)
    try:
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except OSError as e:
        logger.error(f"Failed to save uploaded file {safe_name!r}: {e}")
        raise HTTPException(500, "Failed to save the uploaded file.")

    # Validate the saved file is non-empty
    file_size = os.path.getsize(dest_path)
    if file_size == 0:
        os.remove(dest_path)
        raise HTTPException(400, "The uploaded file is empty.")

    logger.info(f"File saved: {dest_path!r} ({file_size} bytes)")

    # --- Extract text ---------------------------------------------------------
    text = ""
    if ext == ".pdf":
        try:
            text = extract_pdf_text(dest_path)
            logger.info(f"PDF text extraction completed for {safe_name!r}")
        except ValueError as e:
            # extract_pdf_text raises ValueError for corrupted/unreadable PDFs
            raise HTTPException(400, str(e))
        except Exception as e:
            logger.error(f"Unexpected error extracting text from {safe_name!r}: {e}", exc_info=True)
            raise HTTPException(500, "Failed to extract text from the PDF.")
    else:
        try:
            with open(dest_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            raise HTTPException(400, "File appears to be binary or uses an unsupported encoding. Use UTF-8.")
        except Exception as e:
            logger.error(f"Failed to read {safe_name!r}: {e}", exc_info=True)
            raise HTTPException(500, "Failed to read the uploaded file.")

    if not text.strip():
        raise HTTPException(
            400,
            "No extractable text found in that file. "
            "If it's a scanned PDF it needs OCR first; if it's a text file it appears empty.",
        )

    # --- Chunk & index --------------------------------------------------------
    chunks = chunk_single_document(safe_name, text)
    logger.info(f"Text chunking completed for {safe_name!r}: {len(chunks)} text chunks")

    if ext == ".pdf":
        logger.info(f"Starting table extraction for {safe_name!r}")
        try:
            table_chunks = extract_tables_from_pdf(dest_path, safe_name)
            chunks += table_chunks
            logger.info(f"Table extraction completed for {safe_name!r}: {len(table_chunks)} table chunks")
        except Exception as e:
            # Table extraction failure is non-fatal — log and continue with text chunks
            logger.warning(f"Table extraction failed for {safe_name!r}: {e}. Continuing with text-only chunks.")

    _add_chunks_to_index(chunks)

    logger.info(f"Ingestion completed for {safe_name!r}: {len(chunks)} chunks added")
    return IngestResponse(
        filename=safe_name, chunks_added=len(chunks), total_chunks=len(store.chunks)
    )


@app.post("/ingest/url", response_model=IngestResponse)
def ingest_url(req: UrlIngestRequest):
    url = req.url.strip()
    if not url:
        raise HTTPException(400, "URL cannot be empty.")

    logger.info(f"URL ingestion started: {url!r}")

    try:
        title, text = extract_url_text(url)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Unexpected error fetching URL {url!r}: {e}", exc_info=True)
        raise HTTPException(500, "Failed to fetch or parse the URL.")

    filename = slugify(title) + ".txt"
    dest_path = os.path.join(DOCS_FOLDER, filename)
    try:
        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\nSource: {url}\n\n{text}")
    except OSError as e:
        logger.error(f"Failed to save URL content to disk: {e}")
        raise HTTPException(500, "Failed to save the fetched content.")

    chunks = chunk_single_document(filename, text)
    _add_chunks_to_index(chunks)

    logger.info(f"URL ingestion completed: {filename!r} ({len(chunks)} chunks)")
    return IngestResponse(
        filename=filename, chunks_added=len(chunks), total_chunks=len(store.chunks)
    )


# ---------------------------------------------------------------------------
# Experiment endpoints — isolated from the live chatbot index above.
# ---------------------------------------------------------------------------
@app.get("/experiments/options")
def experiment_options():
    """Everything the lab UI needs to build a config."""
    return {
        "documents": pipeline.list_documents(),
        "chunking_strategies": list(STRATEGIES),
        "embedding_models": pipeline.EMBEDDING_MODELS,
        "retrieval_strategies": list(RETRIEVAL_STRATEGIES),
        "rerank_models": pipeline.RERANK_MODELS,
    }


@app.post("/experiments/run")
def run_experiment(req: ExperimentRequest):
    try:
        return pipeline.run_experiment(req.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/experiments")
def list_experiments(limit: int = 20):
    return {"experiments": pipeline.list_runs(min(max(limit, 1), 100))}


@app.get("/experiments/eval")
def experiment_eval():
    """Labelled sample questions from eval/questions.json (empty if absent)."""
    return {"questions": pipeline.list_eval_questions()}


@app.get("/experiments/{run_id}")
def get_experiment(run_id: str):
    record = pipeline.get_run(run_id)
    if record is None:
        raise HTTPException(404, f"No experiment with id '{run_id}'.")
    return record


@app.delete("/experiments/{run_id}")
def delete_experiment(run_id: str):
    if not pipeline.delete_run(run_id):
        raise HTTPException(404, f"No experiment with id '{run_id}'.")
    return {"deleted": run_id}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    question = req.question.strip()
    if not question:
        return QueryResponse(answer="Please enter a question.")

    if not store.chunks:
        return QueryResponse(answer="Nothing's been indexed yet — add a file or URL above first.")

    results = _live_search(question)
    prompt = pipeline.build_prompt(question, [chunk for chunk, _ in results])

    try:
        answer = generate_answer(prompt)
        logger.info(f"Query processed successfully: {question[:80]!r}")
    except Exception as e:
        logger.error(f"Answer generation failed for query {question[:80]!r}: {e}", exc_info=True)
        return QueryResponse(answer=f"Answer generation failed: {e}")

    return QueryResponse(
        answer=answer,
        sources=[
            SourceRef(
                source=chunk.source,
                chunk_id=chunk.chunk_id,
                score=round(float(score), 3),
                snippet=_snippet(chunk.parent_text or chunk.text),
                text=chunk.parent_text or chunk.text,
            )
            for chunk, score in results
        ],
    )
