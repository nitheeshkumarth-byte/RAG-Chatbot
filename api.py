"""
api.py — small FastAPI server that wraps the RAG pipeline for the browser UI.

Run:
    uvicorn api:app --reload

Then open http://localhost:8000 in your browser.
"""

import os
import shutil

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag.generate import generate_answer
from rag.ingest import chunk_single_document, extract_pdf_text, extract_url_text, slugify
from rag.store import VectorStore

load_dotenv()  # reads .env into os.environ

INDEX_PATH = os.environ.get("INDEX_PATH", "index.pkl")
DOCS_FOLDER = os.environ.get("DOCS_FOLDER", "sample_docs")
TOP_K = 3
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

app = FastAPI()

os.makedirs(DOCS_FOLDER, exist_ok=True)

# Serve everything in static/ (our single-page frontend) at the root URL.
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the index once at startup rather than on every request — embedding
# the model itself takes a second or two, we don't want to pay that cost
# per query. If no index has been built yet, start empty: the UI can add
# the first document straight from the browser instead of requiring
# build_index.py to run first.
store = VectorStore()
try:
    store.load(INDEX_PATH)
except FileNotFoundError:
    pass


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


class UrlIngestRequest(BaseModel):
    url: str


class IngestResponse(BaseModel):
    filename: str
    chunks_added: int
    total_chunks: int


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


def _add_chunks_to_index(chunks) -> None:
    """Embed a new batch of chunks, add them to the in-memory store, and
    persist the updated index to disk so it survives a server restart."""
    store.add(chunks)
    store.save(INDEX_PATH)


@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


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
    return {"removed": filename, "total_chunks": len(store.chunks)}


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Use .pdf, .txt, or .md.")

    dest_path = os.path.join(DOCS_FOLDER, file.filename)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    if ext == ".pdf":
        text = extract_pdf_text(dest_path)
    else:
        with open(dest_path, "r", encoding="utf-8") as f:
            text = f.read()

    if not text.strip():
        raise HTTPException(
            400,
            "No extractable text found in that file. If it's a scanned PDF, it needs OCR first.",
        )

    chunks = chunk_single_document(file.filename, text)
    _add_chunks_to_index(chunks)

    return IngestResponse(
        filename=file.filename, chunks_added=len(chunks), total_chunks=len(store.chunks)
    )


@app.post("/ingest/url", response_model=IngestResponse)
def ingest_url(req: UrlIngestRequest):
    try:
        title, text = extract_url_text(req.url)
    except ValueError as e:
        raise HTTPException(400, str(e))

    filename = slugify(title) + ".txt"
    dest_path = os.path.join(DOCS_FOLDER, filename)
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\nSource: {req.url}\n\n{text}")

    chunks = chunk_single_document(filename, text)
    _add_chunks_to_index(chunks)

    return IngestResponse(
        filename=filename, chunks_added=len(chunks), total_chunks=len(store.chunks)
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not store.chunks:
        return QueryResponse(answer="Nothing's been indexed yet — add a file or URL above first.")

    results = store.search(req.question, top_k=TOP_K)
    prompt = build_prompt(req.question, results)

    try:
        answer = generate_answer(prompt)
    except Exception as e:
        return QueryResponse(answer=f"ERROR generating answer: {e}")

    return QueryResponse(answer=answer)
