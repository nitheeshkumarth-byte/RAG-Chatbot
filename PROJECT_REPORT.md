# Mini RAG — Comprehensive Project Documentation Report

> **A from-scratch, no-framework Retrieval-Augmented Generation (RAG) pipeline**
> that chunks documents → embeds locally → retrieves by similarity → generates
> grounded answers with an LLM (Google Gemini or Amazon Bedrock/Claude).

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Overview](#2-project-overview)
3. [System Architecture](#3-system-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Directory Structure](#5-directory-structure)
6. [Pipeline in Detail](#6-pipeline-in-detail)
7. [Module Reference](#7-module-reference)
8. [API Specification](#8-api-specification)
9. [Configuration](#9-configuration)
10. [Data & Persistence](#10-data--persistence)
11. [User Interfaces](#11-user-interfaces)
12. [Docker & Containerization](#12-docker--containerization)
13. [Deployment Paths](#13-deployment-paths)
14. [Security & Privacy](#14-security--privacy)
15. [Known Limitations](#15-known-limitations)
16. [Code Quality Issues](#16-code-quality-issues)
17. [Suggested Next Steps](#17-suggested-next-steps)
18. [Troubleshooting](#18-troubleshooting)
19. [Conclusion](#19-conclusion)

---

## 1. Executive Summary

**Mini RAG** is an educational-grade, production-aware Retrieval-Augmented Generation pipeline implemented in approximately 1,000 lines of dependency-light Python. It demonstrates the complete RAG loop — **chunk → embed → store → retrieve → generate** — without relying on orchestration frameworks (LangChain, LlamaIndex) or dedicated vector database servers (Pinecone, Weaviate, Milvus).

The project supports:

- **Two deployment architectures**: a fully local pipeline (numpy + sentence-transformers) and a fully managed AWS pipeline (S3 + Bedrock Knowledge Bases + Titan embeddings + OpenSearch Serverless)
- **Two LLM backends**: Google Gemini and Amazon Bedrock (Claude), switchable at runtime via a single environment variable
- **Two interaction surfaces**: a browser-based UI (FastAPI + vanilla HTML/JS) and a command-line interface
- **Multiple document formats**: PDF (text-layer), TXT, Markdown, and arbitrary web article URLs
- **Advanced PDF handling**: table extraction via `pdfplumber` with vision-based fallback for nested tables using Gemini

The codebase is currently in a **broken state** due to unresolved git merge conflict markers across 7 files. These must be resolved before the project can run.

---

## 2. Project Overview

### What It Does

1. **Ingests** documents in three formats (`.pdf`, `.txt`, `.md`) from a folder or browser upload, plus arbitrary blog-post URLs (article text extracted automatically via `trafilatura`)
2. **Chunks** each document into overlapping, fixed-size word chunks (300 words for CLI, 500 words for API)
3. **Embeds** every chunk locally with `sentence-transformers` (`all-MiniLM-L6-v2`) — free, runs on your machine, no API key
4. **Stores** embeddings in plain numpy arrays persisted to `index.pkl` (no vector database server)
5. **Retrieves** the top-3 most similar chunks to a user's question using cosine similarity (implemented as a dot product on normalized vectors)
6. **Generates** a grounded answer from Gemini or Bedrock/Claude using a prompt that instructs the model to answer *only from the retrieved context*

### Design Philosophy

The project's core principle is **transparency**. Every stage of the pipeline is plain, readable Python so you can see exactly what "RAG" means before swapping in heavier libraries. This makes it ideal for:

- Learning RAG fundamentals
- Controlled experimentation (tune chunk size, swap models, test prompts)
- Prototyping before committing to production infrastructure

---

## 3. System Architecture

### High-Level Data Flow

```
Browser upload / URL             your_docs/*.md  (CLI path)
        │                                │
        ▼                                ▼
   POST /ingest/file ┐            build_chunks(folder)
   POST /ingest/url  ┴───────────────► ingest.py
                                              │ splits each source into
                                              ▼ overlapping ~word chunks
                                        store.py
                                              │ embeds each chunk with
                                              │ all-MiniLM-L6-v2, keeps
                                              │ vectors in numpy
                                              ▼
                              POST /query OR query.py
                                              │ embeds your question, finds
                                              │ the top-3 most similar chunks
                                              │ (cosine similarity), stuffs
                                              │ them into a prompt
                                              ▼
                                        generate.py
                                              │ generates a grounded answer
                                              │ via Gemini or Bedrock
                                              ▼
                                           answer text
```

### Two Deployment Architectures

| Aspect | Local Pipeline (`api.py`) | Managed AWS Pipeline (`api_kb.py`) |
|---|---|---|
| **Chunking** | `rag/ingest.py` (word-based) | Bedrock Knowledge Base (automatic) |
| **Embedding** | `all-MiniLM-L6-v2` (local, 384-dim) | Titan Text Embeddings V2 (managed) |
| **Vector Store** | In-memory numpy array → `index.pkl` | OpenSearch Serverless (managed) |
| **Retrieval** | Cosine similarity via dot product | Bedrock Retrieve API |
| **Generation** | Gemini or Bedrock (via `rag/generate.py`) | Gemini or Bedrock (via `rag/generate.py`) |
| **File Storage** | Local filesystem (`sample_docs/`) | Amazon S3 |
| **Docker Image** | Large (~2 GB with torch) | Small (~200 MB, no torch) |
| **Dependencies** | `requirements.txt` (11 packages) | `requirements-kb.txt` (7 packages) |

---

## 4. Technology Stack

| Concern | Technology | Notes |
|---|---|---|
| Language | Python ≥ 3.11 | Docker uses `python:3.11-slim`; workspace venv is CPython 3.12 |
| Web server | FastAPI + uvicorn | REST API + static file serving |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | Local, no API key; 384-dim vectors |
| Vector storage | numpy arrays | Persisted via `pickle` to `index.pkl` |
| PDF text | pypdf | Text-layer extraction only, no OCR |
| PDF tables | pdfplumber | Structured table extraction to Markdown |
| Vision fallback | Gemini (google-genai) | Nested table transcription via page rendering |
| URL extraction | trafilatura | Removes nav/ads/comments; returns `(title, text)` |
| Generation | google-genai (Gemini) or boto3 (Bedrock) | Selected at runtime via env var |
| Config | python-dotenv + AWS SSM Parameter Store | Dual-mode: local `.env` or cloud secrets |
| Frontend | Vanilla HTML/CSS/JS | Single file, no build step, no framework |
| Containerization | Docker | Two variants: local pipeline and KB pipeline |
| Deployment | systemd (venv) or Docker on EC2 | Three documented paths |

### Dependencies

**`requirements.txt`** (local pipeline):

| Package | Minimum Version | Purpose |
|---|---|---|
| sentence-transformers | ≥2.7.0 | Local embeddings |
| numpy | ≥1.26.0 | Vector storage + similarity search |
| google-genai | ≥0.3.0 | Gemini LLM client |
| python-dotenv | ≥1.0.0 | Load `.env` into environment |
| fastapi | ≥0.115.0 | REST API framework |
| uvicorn | ≥0.30.0 | ASGI server |
| pypdf | ≥4.0.0 | PDF text extraction |
| pdfplumber | ≥0.11.0 | PDF table extraction |
| trafilatura | ≥1.12.0 | Web article extraction |
| python-multipart | ≥0.0.9 | FastAPI file upload support |
| boto3 | ≥1.34.0 | AWS Bedrock client (optional) |

**`requirements-kb.txt`** (managed KB pipeline — lighter):

Omits `sentence-transformers`, `numpy`, `pypdf`, and `pdfplumber` — the Knowledge Base handles chunking/embedding/PDF-parsing during S3 sync. This keeps the Docker image much smaller and builds much faster.

---

## 5. Directory Structure

```
GenAi_RAG/
├── api.py                     # FastAPI server: HTTP endpoints + browser UI (local pipeline)
├── api_kb.py                  # FastAPI server: Bedrock Knowledge Bases pipeline (managed AWS)
├── build_index.py             # CLI: chunk + embed + save a whole folder
├── query.py                   # CLI: ask a question from the terminal
├── fetch_url.py               # CLI: download a URL's article text to sample_docs/
├── requirements.txt           # Python dependencies (local pipeline)
├── requirements-kb.txt        # Python dependencies (KB pipeline — lighter)
├── Dockerfile                 # Container image (local pipeline)
├── Dockerfile.apprunner       # Container image (KB pipeline — no torch/numpy)
├── .env                       # Runtime config (gitignored — never commit)
├── .gitignore                 # Files git should ignore
├── .dockerignore              # Files excluded from Docker image
├── index.pkl                  # Persisted vector index (pickle, gitignored)
├── README.md                  # Main project documentation (838 lines)
├── project_documentation.tex  # LaTeX technical documentation (679 lines)
├── DEPLOY_AWS.md              # EC2 + venv + Gemini deployment guide
├── DEPLOY_DOCKER_BEDROCK.md   # EC2 + Docker + Bedrock deployment guide
├── DEPLOY_APPRUNNER_KB.md     # App Runner + Bedrock KB guide (deprecated)
├── DEPLOY_ECS_EXPRESS_KB.md   # ECS Express Mode + Bedrock KB guide
├── rag/                       # Core RAG pipeline package
│   ├── __init__.py            # Empty package marker
│   ├── config.py              # Config loader (env vars → AWS SSM Parameter Store)
│   ├── ingest.py              # Document loading, chunking, URL extraction, PDF tables
│   ├── store.py               # Embedding + cosine similarity search + persistence
│   ├── generate.py            # LLM generation (Gemini / Bedrock backends)
│   └── kb_client.py           # Bedrock Knowledge Base client (Retrieve API)
├── static/
│   └── index.html             # Single-page frontend (vanilla HTML/CSS/JS, dark theme)
└── sample_docs/               # Sample documents for CLI path
    ├── vector_databases.md    # Explains vector DBs, ANN indexing (HNSW/IVF)
    ├── chunking_strategies.md # Fixed-size vs structure-aware vs semantic chunking
    └── New Text Document.txt  # Personal note (privacy concern — see Security)
```

---

## 6. Pipeline in Detail

### Stage 1: Ingestion (`rag/ingest.py`)

The ingestion layer is the only part of the pipeline that knows about file formats or URLs. It produces a flat list of `Chunk` objects that the rest of the pipeline processes uniformly.

**Key Functions:**

| Function | Signature | Description |
|---|---|---|
| `extract_pdf_text` | `(path: str) → str` | Opens PDF with `pypdf.PdfReader`, joins every page's text with `\n\n`. Text-layer only — scanned PDFs yield empty strings. |
| `extract_tables_from_pdf` | `(path: str, filename: str) → List[Chunk]` | Uses `pdfplumber` to detect tables, converts to Markdown. Falls back to Gemini vision for nested tables. |
| `extract_url_text` | `(url: str) → (title, text)` | Uses `trafilatura` to fetch and extract article text. Raises `ValueError` on failure. |
| `slugify` | `(text: str, max_len=60) → str` | Creates safe filename slugs from titles. |
| `load_documents` | `(folder_path) → [(filename, text)]` | Loads `.md`, `.txt`, `.pdf` files from a folder (sorted by filename). |
| `chunk_text` | `(text, chunk_size=500, overlap=50) → [str]` | Word-based chunking with overlap. |
| `chunk_single_document` | `(filename, text, ...) → [Chunk]` | Wraps chunks into `Chunk` objects. |
| `build_chunks` | `(folder_path, ...) → [Chunk]` | Loads all docs in a folder and flattens to chunk list. |

**`Chunk` dataclass:**

```python
@dataclass
class Chunk:
    text: str           # The chunk's text content
    source: str         # Filename the chunk came from
    chunk_id: int       # Position of this chunk within its file (0-based)
    chunk_type: str = "text"  # "text" or "table"
```

**PDF Table Extraction (origin/main branch):**

The `origin/main` branch adds sophisticated PDF table handling:

1. **Structured extraction**: `pdfplumber` detects tables and converts them to clean Markdown tables via `_table_rows_to_markdown()`
2. **Nested table detection**: `_looks_like_nested_table()` uses a heuristic (cells with ≥2 newlines) to detect tables that `pdfplumber` likely flattened
3. **Vision fallback**: For nested tables, the page is rendered as a PNG image and sent to Gemini with a prompt to transcribe the table structure
4. **Graceful degradation**: If vision transcription fails, falls back to flattened extraction with a warning

### Stage 2: Embedding & Storage (`rag/store.py`)

| Component | Description |
|---|---|
| `EMBEDDING_MODEL` | `"all-MiniLM-L6-v2"` — 384-dimensional, normalized vectors |
| `VectorStore.__init__` | Loads SentenceTransformer model (~90 MB download on first run) |
| `add(chunks)` | Encodes with `normalize_embeddings=True`, appends to numpy matrix via `np.vstack` |
| `search(query, top_k=3)` | Cosine similarity via dot product (valid because vectors are normalized) |
| `sources()` | Unique source filenames in insertion order (first-seen dedup) |
| `remove_source(source)` | Filters out chunks/embeddings for a given source |
| `save(path)` / `load(path)` | Pickle persistence to/from `index.pkl` |

**SearchResult dataclass:**

```python
@dataclass
class SearchResult:
    chunk: Chunk
    score: float
```

### Stage 3: Generation (`rag/generate.py`)

The generation step is swappable between Gemini and Bedrock at runtime via `GENERATION_BACKEND`:

| Backend | Function | Key Requirements | API Call |
|---|---|---|---|
| Gemini | `_generate_gemini(prompt)` | `GEMINI_API_KEY` env var | `client.models.generate_content(model, contents=prompt)` |
| Bedrock | `_generate_bedrock(prompt)` | AWS credentials (IAM role or env vars) | `client.converse(modelId, messages, inferenceConfig)` |

Both `api.py` and `query.py` call `generate_answer(prompt)` — neither knows which provider actually ran.

**Bedrock specifics:**
- Uses the **Converse API** (provider-agnostic, works for Claude/Nova/etc.)
- `inferenceConfig={"maxTokens": 500}`
- No explicit AWS keys — `boto3` auto-discovers credentials from env vars, `~/.aws/credentials`, or EC2 IAM role

### Stage 4: Configuration (`rag/config.py`)

A dual-mode config loader that enables the same code to run locally and on AWS:

1. **Local path**: If the env var is set directly (e.g., `GEMINI_API_KEY=AIza...` in `.env`), use it as-is
2. **AWS path**: If `<VAR>_PARAM` env var is set (e.g., `GEMINI_API_KEY_PARAM=/rag-project/gemini-api-key`), fetch the real value from **AWS Systems Manager Parameter Store** (SecureString, decrypted)
3. **Caching**: `lru_cache` ensures each parameter is fetched only once per process

---

## 7. Module Reference

### `api.py` — FastAPI Server (Local Pipeline)

**Module-level setup:**
- `load_dotenv()` loads `.env` into `os.environ`
- `INDEX_PATH`, `DOCS_FOLDER`, `TOP_K = 3`, `ALLOWED_EXTENSIONS`
- `DOCS_FOLDER` created with `os.makedirs(..., exist_ok=True)` at import
- `/static` mounted for frontend serving
- `VectorStore` created once at import time and loaded from `INDEX_PATH`

**Helper functions:**
- `build_prompt(question, results)` — joins top results as `[Source: <source>]\n<text>` blocks in a grounded-answer prompt
- `_add_chunks_to_index(chunks)` — calls `store.add(chunks)` then `store.save(INDEX_PATH)`

**Pydantic models:**
- `QueryRequest {question: str}`
- `QueryResponse {answer: str}`
- `UrlIngestRequest {url: str}`
- `IngestResponse {filename, chunks_added, total_chunks}`

### `api_kb.py` — FastAPI Server (Managed KB Pipeline)

A sibling to `api.py` that swaps local ingestion/retrieval for managed AWS:
- Uploads go to S3 via `boto3.client("s3")`
- Bedrock Knowledge Base handles chunking/embedding/retrieval
- Queries call `retrieve()` from `rag/kb_client.py`
- Generation unchanged — still uses `rag/generate.py`

### `build_index.py` — CLI Index Builder

One-shot CLI that chunks a folder, embeds, and saves the index:
- `python build_index.py` → chunks `sample_docs/`
- `python build_index.py my_docs_folder` → chunks your folder
- Uses `chunk_size=300, overlap=50`
- **Rebuilds from scratch** — does not merge with existing index

### `query.py` — CLI Question Asker

Terminal Q&A tool:
- `python query.py "What is HNSW used for?"`
- Loads store, retrieves top-3 chunks, prints scores, generates answer
- Uses same `build_prompt` shape as `api.py`

### `fetch_url.py` — CLI URL Fetcher

Downloads article text for the CLI path:
- `python fetch_url.py https://example.com/some-blog-post`
- Writes `# {title}\nSource: {url}\n\n{text}` to `sample_docs/<slug>.txt`

### `rag/kb_client.py` — Bedrock KB Client

Thin wrapper around Bedrock's Retrieve API:
- `retrieve(question, top_k=3)` — calls `bedrock-agent-runtime.retrieve()`
- `start_ingestion_job()` — triggers on-demand S3 sync via `bedrock-agent.start_ingestion_job()`

### `static/index.html` — Single-Page Frontend

Dark "notebook" theme with CSS custom properties. Features:
- Source panel with two tabs: Upload file (click/drag-drop) and Paste URL
- Source list with Remove buttons
- Question input + Ask button with pulsing loading indicator
- All rendering HTML-escaped via `div.textContent` trick
- `prefers-reduced-motion` media query support

---

## 8. API Specification

**Base URL:** `http://localhost:8000` (or your deployed host)

### `GET /`
Serves `static/index.html` (the single-page UI).

### `GET /sources`
Returns the distinct source filenames and total chunk count.

**Response:**
```json
{"sources": ["file1.pdf", "file2.md"], "total_chunks": 42}
```

### `POST /ingest/file`
Multipart form field `file`. Accepts `.pdf`, `.txt`, `.md` only.

**Error responses:**
- **400** — `{"detail": "Unsupported file type '.doc'. Use .pdf, .txt, or .md."}`
- **400** — `{"detail": "No extractable text found in that file. If it's a scanned PDF, it needs OCR first."}`

**Success response:**
```json
{"filename": "doc.pdf", "chunks_added": 15, "total_chunks": 57}
```

### `POST /ingest/url`
JSON body: `{"url": "https://..."}`.

**Error responses:**
- **400** — `{"detail": "Could not download https://..."}` or `{"detail": "Could not find readable article content at https://..."}`

**Success response:** Same shape as file ingest.

### `POST /query`
JSON body: `{"question": "..."}`.

**Behavior:**
- Empty store → **200** with `{"answer": "Nothing's been indexed yet — add a file or URL above first."}`
- Generation failure → **200** with `{"answer": "ERROR generating answer: {e}"}`
- Success → **200** with `{"answer": "<generated text>"}`

### `DELETE /sources/{filename}`
Removes one source and all its chunks/embeddings.

**Success response:**
```json
{"removed": "doc.pdf", "total_chunks": 42}
```

**Error response:**
- **404** — `{"detail": "'doc.pdf' isn't currently indexed."}`

---

## 9. Configuration

### Environment Variables

| Variable | Default (code) | Used by | Meaning |
|---|---|---|---|
| `INDEX_PATH` | `index.pkl` | `api.py`, `build_index.py`, `query.py` | Where the pickled index is saved/loaded |
| `DOCS_FOLDER` | `sample_docs` | `api.py` | Where uploads/URL extractions are written |
| `GENERATION_BACKEND` | `gemini` | `rag/generate.py` | `gemini` or `bedrock` |
| `GEMINI_API_KEY` | — (error if unset) | `rag/generate.py` | Google AI Studio key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | `rag/generate.py` | Gemini model name |
| `AWS_REGION` | `us-east-1` | `rag/generate.py` | AWS region for Bedrock |
| `BEDROCK_MODEL_ID` | `anthropic.claude-3-5-haiku-20241022-v1:0` | `rag/generate.py` | Bedrock model ID |
| `S3_BUCKET_NAME` | — | `api_kb.py` | S3 bucket for KB pipeline |
| `KNOWLEDGE_BASE_ID` | — | `rag/kb_client.py` | Bedrock KB ID |
| `DATA_SOURCE_ID` | — | `rag/kb_client.py` | Bedrock KB data source ID |

### Shipped `.env` Values

| Variable | Value |
|---|---|
| `GENERATION_BACKEND` | `gemini` |
| `GEMINI_MODEL` | `gemini-2.5-flash` |
| `AWS_REGION` | `ap-south-2` |
| `BEDROCK_MODEL_ID` | `anthropic.claude-haiku-4-5-20251001-v1:0` |

> **Note:** The shipped `GEMINI_API_KEY` value does not match the usual Google AI Studio format (`AIza…`) — it may be invalid or from a different provider.

---

## 10. Data & Persistence

### Index File (`index.pkl`)

- **Format**: Pickle of `{"chunks": [Chunk, ...], "embeddings": np.ndarray}`
- **Size**: ~6 KB for the sample docs
- **Saved**: After every API ingest/delete, and by `build_index.py`
- **Loaded**: Once at API startup
- **Gitignored** and **excluded from Docker**
- **Version-fragile**: Opaque pickle format, not suitable for multi-process or shared storage

### Document Folder (`sample_docs/`)

- Uploaded files stored verbatim
- URL ingests write `# <title>\nSource: <url>\n\n<text>` files
- **Not** re-ingested at startup — the index is self-contained

### Memory

- Chunks + embeddings live in RAM (numpy arrays) for the process lifetime
- Large corpora will eat memory accordingly

### Concurrency

- No locking around `store.add/save`
- Two simultaneous ingests could race on the `index.pkl` write
- Fine for single-user use

---

## 11. User Interfaces

### Browser UI

Access at `http://localhost:8000`. Single-page app with:

- **Source panel**: Two tabs (Upload file / Paste URL), status line, scrolling source list
- **File upload**: Click-to-browse and drag-and-drop zone (accepts `.pdf`, `.txt`, `.md`)
- **URL input**: Text field with Enter key or Add button submission
- **Question bar**: Input + Ask button with pulsing "Searching & thinking" indicator
- **Answer area**: Rendered with HTML escaping for safety
- **Source management**: Remove buttons for each indexed source

### Command-Line Interface

```bash
# Build index from a folder
python build_index.py path/to/your/folder

# Ask a question
python query.py "your question"

# Download an article for the CLI path
python fetch_url.py https://example.com/some-post
```

---

## 12. Docker & Containerization

### `Dockerfile` (Local Pipeline)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p sample_docs
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Key design decisions:**
- Dependencies installed first (separate cacheable layer for heavy torch/transformers)
- `sample_docs/` directory created in image
- Binds to `0.0.0.0:8000` for external access

### `Dockerfile.apprunner` (KB Pipeline)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements-kb.txt .
RUN pip install --no-cache-dir -r requirements-kb.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api_kb:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Key difference:** No `sentence-transformers`/`torch`/`numpy` — much smaller image, faster builds.

### `.dockerignore`

Excludes: `.env`, `index.pkl`, `__pycache__/`, `*.pyc`, `.venv/`, `venv/`, `.git/`, `.gitignore`, `*.md`

### Build & Run

```bash
docker build -t rag-project .
docker run -d --name rag-app --restart unless-stopped -p 8000:8000 \
  --env-file .env -v "$(pwd)/data":/app/data rag-project
```

For persistence, `.env` must set:
- `INDEX_PATH=/app/data/index.pkl`
- `DOCS_FOLDER=/app/data/sample_docs`

---

## 13. Deployment Paths

### Path 1: EC2 + venv + Gemini (`DEPLOY_AWS.md`)

- **Simplest path**: One EC2 instance (Ubuntu 22.04, t3.small), plain Python venv
- **Generation**: Google Gemini
- **Process management**: systemd service (`rag.service`)
- **Cost**: ~$15/month (t3.small), ~$0.80/month (stopped, EBS only)
- **Optional hardening**: Elastic IP, nginx reverse proxy, HTTPS via certbot, S3 for index

### Path 2: EC2 + Docker + Bedrock (`DEPLOY_DOCKER_BEDROCK.md`)

- **Containerized**: Same EC2 instance, app runs in Docker
- **Generation**: Amazon Bedrock (Claude) — no API key to manage
- **Authentication**: IAM instance role (automatic credential forwarding)
- **One-time setup**: Anthropic use-case submission (if required)

### Path 3: ECS Express Mode + Bedrock KB (`DEPLOY_ECS_EXPRESS_KB.md`)

- **Fully managed**: ECS Express Mode + Bedrock Knowledge Bases
- **Storage**: S3 + Titan embeddings + OpenSearch Serverless
- **No server management**: Auto-provisioned ALB, auto-scaling, CloudWatch logs
- **Most expensive**: OpenSearch Serverless minimum capacity billing
- **Note**: Replaces deprecated App Runner path (closed to new customers April 30, 2026)

### Deprecated: App Runner + Bedrock KB (`DEPLOY_APPRUNNER_KB.md`)

- **Status**: Deprecated — App Runner closed to new customers on April 30, 2026
- **Replacement**: ECS Express Mode (Path 3 above)

---

## 14. Security & Privacy

### Secrets Management

- `.env` and `index.pkl` are **gitignored** and **`.dockerignore`d** — never commit or bake into images
- AWS deployment uses IAM roles — no secrets in `.env` or code
- `rag/config.py` supports AWS SSM Parameter Store for cloud deployments

### API Key Concern

- The shipped `GEMINI_API_KEY` value does not match the standard `AIza…` format
- May be invalid or from a different provider
- **Recommendation**: Grab a fresh key from Google AI Studio and update `.env`

### Error Leakage

- Generation errors are returned to the browser as the answer body (`ERROR generating answer: …`)
- This can leak internal details (stack traces, provider error messages)
- **Recommendation**: Log server-side and return a generic message before public deployment

### Privacy

- `sample_docs/New Text Document.txt` contains personal/private content about a third party
- Included in any `build_index.py` run over the default folder
- **Recommendation**: Remove it (and any index built from it) before sharing or deploying

### No Authentication

- The API has no authentication, rate limiting, or multi-user separation
- Suitable for local/single-user use only

---

## 15. Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| **No OCR** | Scanned/image PDFs yield no text | Add `pytesseract` preprocessing |
| **Fixed-size word chunking** | No structure-aware or semantic chunking | Implement heading-aware or sentence-boundary chunking |
| **Top-3 retrieval only** | No hybrid search, no reranker, no metadata filtering | Add BM25 + vector hybrid, cross-encoder reranker |
| **In-memory numpy store** | No ANN index, no horizontal scaling, no transactional safety | Migrate to Chroma/Qdrant/pgvector at scale |
| **No automated tests** | Quality assessed manually | Build eval set with 15-20 questions |
| **Single-process, single-user** | No auth, no rate limiting, no locking | Add middleware for production use |
| **Pickle format** | Version-fragile, not multi-process safe | Migrate to a proper vector DB |
| **Chunk size inconsistency** | CLI uses 300 words, API uses 500 words | Standardize or make configurable |
| **Duplicated code** | `build_prompt()` and `TOP_K` duplicated between `api.py` and `query.py` | Extract to shared module |

---

## 16. Code Quality Issues

### Critical: Unresolved Git Merge Conflict Markers

**This is the most significant finding.** Multiple files contain unresolved merge conflict markers (`<<<<<<< HEAD`, `=======`, `>>>>>>> origin/main`):

| File | Conflict Locations |
|---|---|
| `README.md` | Lines 1, 444, 452, 458, 699, 787, 793, 827, 838 |
| `requirements.txt` | Lines 8-11 |
| `api.py` | Lines 20-24, 138-144 |
| `build_index.py` | Lines 12-24, 33-39 |
| `rag/ingest.py` | Lines 14-19, 24-34, 44-55, 61-211, 222-233, 285-303 |
| `rag/generate.py` | Lines 30-42, 53-57 |
| `.gitignore` | Lines 6-9 |

These conflicts are between `HEAD` (local changes) and `origin/main` (remote changes). The `origin/main` branch adds features like PDF table extraction (`pdfplumber`), vision-based nested table transcription, AWS SSM Parameter Store config, and the `api_kb.py` managed pipeline.

**The project cannot run in its current state.** These conflicts must be resolved by:
1. Deciding which version of each conflicting section to keep (or merging both)
2. Removing all `<<<<<<< HEAD`, `=======`, and `>>>>>>> origin/main` markers
3. Verifying the resolved code is syntactically valid

### Other Issues

- **No `.env.example`** exists despite being referenced by `query.py` docstring and both deployment guides
- **Duplicated code**: `build_prompt()` and `TOP_K = 3` are duplicated between `api.py` and `query.py`
- **Chunk size inconsistency**: CLI uses 300 words, API uses 500 words (module default)
- **No CI/CD configuration**: No automated build, test, or deployment pipeline
- **No automated tests**: No test suite or evaluation set

---

## 17. Suggested Next Steps

### Immediate (Fix the Code)

1. **Resolve merge conflicts** across all 7 affected files
2. **Create `.env.example`** with placeholder values
3. **Standardize chunk size** between CLI and API paths (or make it configurable)
4. **Extract shared code** (`build_prompt`, `TOP_K`) into a common module

### Short-Term (Improve Quality)

5. **Build an eval set**: Write 15-20 questions with known answers from your docs
6. **Add automated tests**: Unit tests for chunking, embedding, and retrieval
7. **Add CI/CD**: GitHub Actions for linting, testing, and Docker builds
8. **Fix the API key**: Replace with a valid Google AI Studio key

### Medium-Term (Enhance Capabilities)

9. **Add hybrid search**: Combine BM25 keyword search with vector similarity
10. **Add a reranker**: Cross-encoder model to re-rank retrieved chunks
11. **Implement structure-aware chunking**: Split on headings/paragraphs instead of fixed word count
12. **Add OCR support**: `pytesseract` for scanned PDFs
13. **Swap vector store**: Migrate from numpy/pickle to Chroma or Qdrant

### Long-Term (Production Hardening)

14. **Add authentication**: API keys or OAuth for multi-user support
15. **Add rate limiting**: Prevent abuse
16. **Add monitoring**: Structured logging, metrics, alerting
17. **Horizontal scaling**: Load balancer + shared vector store

---

## 18. Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `GEMINI_API_KEY is not set in .env` | Missing or misnamed env var | Add `GEMINI_API_KEY=AIza...` to `.env` and restart |
| `ERROR generating answer: …` | Backend config issue | Check `GENERATION_BACKEND` and provider error text |
| PDF adds but returns few/no words | Scanned PDF (no text layer) | Add OCR preprocessing |
| `Unsupported file type` | Wrong file extension | Use `.pdf`, `.txt`, or `.md` |
| `Nothing's been indexed yet` | Empty store | Add a source first |
| First run downloads slowly | Model download from Hugging Face | Wait for `all-MiniLM-L6-v2` (~90 MB) download |
| Port 8000 refused externally | Security group not open | Open Custom TCP 8000 in EC2 security group |
| CLI says empty index after `build_index.py` | Wrong `INDEX_PATH` | Confirm same directory/path for both commands |
| `AccessDeniedException` (Bedrock) | IAM role missing or misconfigured | Attach `AmazonBedrockFullAccess` to instance role |
| `ValidationException: model identifier is invalid` | Wrong model ID format | Use cross-region inference profile ID (e.g., `us.anthropic...`) |
| `ParameterNotFound` (SSM) | Wrong parameter path | Confirm exact path in Parameter Store |

---

## 19. Conclusion

**Mini RAG** is a well-architected, educational RAG pipeline that demonstrates the complete chunk → embed → retrieve → generate loop in approximately 1,000 lines of dependency-light Python. Its key strengths are:

- **Transparency**: Every stage is plain, readable Python — no black boxes
- **Flexibility**: Two LLM backends, two deployment architectures, two interaction surfaces
- **Progressive complexity**: Start local with numpy, migrate to managed AWS when ready
- **Comprehensive documentation**: 838-line README, 679-line LaTeX doc, 4 deployment guides

The codebase currently has **critical unresolved merge conflicts** that must be addressed before it can run. Once resolved, the project provides a solid foundation for learning RAG fundamentals, controlled experimentation, and gradual production hardening.

The suggested evolution path is:
1. Fix merge conflicts → 2. Add tests and eval set → 3. Tune chunk size and prompt → 4. Add hybrid search and reranker → 5. Migrate to a proper vector database → 6. Production hardening (auth, monitoring, scaling)

---

*Report generated from comprehensive analysis of all project files including source code, configuration, documentation, deployment guides, and Dockerfiles.*
