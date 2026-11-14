# Mini RAG — Complete Project Documentation

A from-scratch, no-framework **Retrieval-Augmented Generation (RAG)** pipeline:
chunk documents → embed locally → retrieve by similarity → generate a
grounded answer with an LLM (Google Gemini or Amazon Bedrock/Claude).

This project deliberately uses **no LangChain, no vector-DB server, and no
orchestration framework**. Every stage of the pipeline is plain, readable
Python so you can see exactly what "RAG" means before swapping in heavier
libraries.

---

## Table of contents

1. [What this project does](#what-this-project-does)
2. [Directory structure](#directory-structure)
3. [Tech stack](#tech-stack)
4. [How it works (pipeline)](#how-it-works-pipeline)
5. [Module-by-module reference](#module-by-module-reference)
   - [`rag/ingest.py`](#ragingestpy)
   - [Experiment layer](#experiment-layer-ragchunkingpy-ragretrievalpy-ragpipelinepy)
   - [`rag/store.py`](#ragstorepy)
   - [`rag/generate.py`](#raggeneratepy)
   - [`api.py`](#apipy)
   - [`build_index.py`](#build_indexpy)
   - [`query.py`](#querypy)
   - [`fetch_url.py`](#fetch_urlpy)
   - [`static/index.html`](#staticindexhtml)
6. [Installation](#installation)
7. [Configuration (`.env`)](#configuration-env)
8. [Running the web app](#running-the-web-app)
9. [Pipeline lab (`/lab`)](#pipeline-lab-lab)
10. [HTTP API reference](#http-api-reference)
11. [Command-line workflow](#command-line-workflow-alternative)
12. [Data & persistence](#data--persistence)
13. [Docker](#docker)
14. [Deployment](#deployment)
15. [Sample documents](#sample-documents)
16. [Important details & gotchas](#important-details--gotchas)
17. [Security & privacy notes](#security--privacy-notes)
18. [Known limitations](#known-limitations)
19. [Suggested next steps](#suggested-next-steps)
20. [Troubleshooting](#troubleshooting)

---

## What this project does

- **Ingest** documents in three formats: `.pdf`, `.txt`, `.md` (from a folder
  or an upload) plus arbitrary **blog-post URLs** (article text is extracted
  automatically).
- **Chunk** each document into overlapping, fixed-size word chunks.
- **Embed** every chunk locally with `sentence-transformers`
  (`all-MiniLM-L6-v2`) — free, runs on your machine, no API key.
- **Store** the embeddings in plain **numpy** arrays persisted to
  `index.pkl` (no vector database server).
- **Retrieve** the top chunks for a user's question with configurable ranking:
  **cosine similarity** by default (a dot product on normalized vectors), with
  optional BM25, hybrid, MMR and cross-encoder reranking.
- **Generate** a grounded answer from Gemini or Bedrock/Claude using a prompt
  that instructs the model to answer *only from the retrieved context*.

Both a **browser UI** (FastAPI + vanilla HTML/JS) and a **CLI** are available.

---

## Directory structure

```
GenAi_RAG/
├── api.py                     # FastAPI server: HTTP endpoints + browser UI
├── build_index.py             # CLI: chunk + embed + save a whole folder
├── query.py                   # CLI: ask a question from the terminal
├── fetch_url.py               # CLI: download a URL's article text to sample_docs/
├── requirements.txt           # Python dependencies (pinned minimums)
├── Dockerfile                 # Container image definition
├── .dockerignore              # What to exclude from the Docker image
├── .env                       # Runtime config (gitignored — never commit)
├── .gitignore                 # Files git should ignore
├── index.pkl                  # Persisted vector index (pickle, gitignored)
├── README.md                  # This file
├── DEPLOY_AWS.md              # EC2 + venv + Gemini deployment guide
├── DEPLOY_DOCKER_BEDROCK.md   # EC2 + Docker + Bedrock deployment guide
├── rag/                       # The RAG pipeline package (importable as "rag")
│   ├── __init__.py            # Empty package marker
│   ├── ingest.py              # Document loading + legacy chunking
│   ├── chunking.py            # Chunking strategies (fixed → semantic)
│   ├── retrieval.py           # BM25, hybrid, MMR, reranking
│   ├── pipeline.py            # Experiment runner: caches, traces, persistence
│   ├── store.py               # Embedding + similarity search + persistence
│   └── generate.py            # LLM generation (Gemini / Bedrock backends)
├── eval/
│   └── questions.json         # Labelled questions + known-answer snippets
├── static/
│   ├── index.html             # Single-page chatbot frontend (no build step)
│   └── lab.html               # Experiment UI (/lab)
└── sample_docs/               # Sample documents for the CLI path
    ├── vector_databases.md
    ├── chunking_strategies.md
    └── New Text Document.txt
```

---

## Tech stack

| Concern | Technology | Notes |
|---|---|---|
| Web server | **FastAPI** + **uvicorn** | REST API + static file serving |
| Embeddings | **sentence-transformers** (`all-MiniLM-L6-v2`) | Local, no API key; 384-dim vectors |
| Vector storage | **numpy** arrays | Persisted via `pickle` to `index.pkl` |
| PDF text | **pypdf** | Text-layer extraction only, **no OCR** |
| URL article extraction | **trafilatura** | Removes nav/ads/comments; returns `(title, text)` |
| Generation | **google-genai** (Gemini) **or** **boto3** (Bedrock) | Selected at runtime via env var |
| Config | **python-dotenv** | Reads `.env` |
| Frontend | Vanilla HTML/CSS/JS | Single file, no build step, no framework |
| Deployment | systemd (venv) or **Docker** | Two documented paths |

Python ≥ 3.11 is assumed (the shipped `Dockerfile` uses `python:3.11-slim`;
the workspace venv is CPython 3.12). Dependencies with pinned minimums (`requirements.txt`):

```
sentence-transformers>=2.7.0
numpy>=1.26.0
google-genai>=0.3.0
python-dotenv>=1.0.0
fastapi>=0.115.0
uvicorn>=0.30.0
pypdf>=4.0.0
trafilatura>=1.12.0
python-multipart>=0.0.9
boto3>=1.34.0
```

(`python-multipart` is required by FastAPI for `UploadFile`/multipart form
handling. `boto3` is only imported when the `bedrock` backend is selected, so
you can omit it if you only use Gemini.)

---

## How it works (pipeline)

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
                                              │ embeds your question, ranks
                                              │ chunks (same rank() for
                                              │ /query and query.py: vector
                                              │ by default, BM25/hybrid/MMR,
                                              │ multi-query, reranking),
                                              │ stuffs them into a prompt
                                              ▼
                                        generate.py
                                              │ generates a grounded answer
                                              │ via Gemini or Bedrock
                                              ▼
                                           answer text
```

Both the browser upload panel and the CLI funnel into the **same**
chunk → embed → store pipeline — ingestion is the only layer that knows about
file formats or URLs.

---

## Module-by-module reference

### `rag/ingest.py`

Format-agnostic loading and chunking. Exposes:

- **`Chunk`** — a `@dataclass` with three fields:
  - `text: str` — the chunk's text
  - `source: str` — filename the chunk came from
  - `chunk_id: int` — position of this chunk within its file (0-based)

- **`extract_pdf_text(path) -> str`** — opens a PDF with `pypdf.PdfReader`
  and joins every page's `extract_text()` output with `"\n\n"`. Empty pages
  become `""` (no exception). Works only on **text-layer** PDFs; scanned
  PDFs yield empty/near-empty text (see PDF note in
  [Known limitations](#known-limitations)).

- **`slugify(text, max_len=60) -> str`** — lowercases, replaces runs of
  non-`[a-z0-9]` characters with `-`, strips leading/trailing `-`, truncates
  to `max_len`. Falls back to `"page"` if the result is empty. Used to build
  filenames from URL article titles.

- **`extract_url_text(url) -> (title, text)`** — uses `trafilatura.fetch_url`
  then `trafilatura.extract` (comments and tables excluded). Raises
  `ValueError` if the page can't be downloaded **or** if no readable article
  content is found. Title comes from `trafilatura.extract_metadata`,
  falling back to the raw URL. Returns a `(title, text)` tuple.

- **`load_documents(folder_path) -> [(filename, text)]`** — lists the folder
  (sorted by filename) and reads every `.md`/`.txt` as UTF-8 and every `.pdf`
  via `extract_pdf_text`. **Non-matching files are silently skipped.**

- **`chunk_text(text, chunk_size=500, overlap=50) -> [str]`** — word-based
  chunking. Splits on whitespace, steps forward by `chunk_size` words but
  re-includes the previous `overlap` words on each step so sentences straddling
  a boundary keep context. If the whole text is ≤ `chunk_size` words it
  returns a single chunk. The last chunk may be shorter than `chunk_size`.

- **`chunk_single_document(filename, text, chunk_size=500, overlap=50) -> [Chunk]`**
  — chunks one already-loaded document and wraps each piece in a `Chunk`.
  Used both by `build_chunks()` (folder loop) and by the API's incremental
  `/ingest` endpoints (which add one file/URL without re-processing the rest).

- **`build_chunks(folder_path, chunk_size=500, overlap=50) -> [Chunk]`** —
  loads every document in a folder via `load_documents()` and flattens the
  chunk lists into one list.

- **`__main__`** — running `python -m rag.ingest` performs a sanity check:
  builds chunks from `sample_docs/`, prints the count and the first 200
  characters of the first two chunks.

> **Chunk-size note:** the module defaults are `chunk_size=500`, `overlap=50`,
> but `build_index.py` explicitly calls with `chunk_size=300, overlap=50`.
> So the **CLI path** produces ~300-word chunks while the **API/browser path**
> (which relies on the module defaults) produces ~500-word chunks. Both are
> consistent within themselves, but you'll see different counts between the
> two entry points.

### Experiment layer (`rag/chunking.py`, `rag/retrieval.py`, `rag/pipeline.py`)

The `/lab` backend. It is separate from the chatbot path on purpose: a run
builds its own chunk list and embeddings and never writes `index.pkl`.

- **`chunking.py`** — `split_text(text, strategy, size, overlap, model,
  threshold)` returns `Piece`s that are exact `text[start:end]` slices, so the
  UI can map a chunk back to the source. Strategies: `fixed`, `recursive`,
  `sentence`, `markdown`, `semantic`. Semantic embeds sentences with the
  configured model and breaks where the cosine distance between neighbours
  exceeds `threshold`.
- **`retrieval.py`** — `BM25` (no external dependency), `minmax` blending for
  hybrid search, `mmr` for diversity, and `rerank` via a `CrossEncoder`.
- **`pipeline.py`** — config validation (the browser trust boundary),
  content-hash caches for extracted text / chunks / embeddings, `run_config`
  (returns the full trace and metrics), and JSONL persistence with a retention
  cap. Parent-child mode tags each child chunk with its parent's text; the
  prompt gets the parent, retrieval ranks the child.

### `rag/store.py`

Embedding + similarity search + persistence. Uses `sentence-transformers` for
embeddings and plain numpy for search — no vector database.

- **`EMBEDDING_MODEL = "all-MiniLM-L6-v2"`** — small, fast, free, good
  enough to start with. Output vectors are **384-dimensional**, normalized.

- **`SearchResult`** — `@dataclass` with `chunk: Chunk` and `score: float`.

- **`VectorStore.__init__(model_name=EMBEDDING_MODEL)`** — loads the
  SentenceTransformer model (downloads from Hugging Face on first run, ~90 MB)
  and initializes empty `chunks` list and `embeddings = None`.

- **`add(chunks)`** — encodes the chunks' texts with
  `normalize_embeddings=True`, appends to `self.chunks`, and either sets
  `self.embeddings` (first time) or vertically stacks the new matrix with
  `np.vstack`.

- **`search(query, top_k=3) -> [SearchResult]`** — encodes the query
  (normalized), computes **cosine similarity** as a matrix dot product
  (`self.embeddings @ query_vec` — valid because the vectors are normalized),
  returns the indices with the largest `top_k` scores (descending). Returns
  `[]` if nothing is stored yet.

- **`sources() -> [str]`** — unique source filenames **in insertion order**
  (first-seen dedup, not sorted).

- **`remove_source(source)`** — filters out every chunk (and its embedding
  row) whose `source` equals the argument; resets `embeddings` to `None` if
  that empties the store.

- **`save(path)`** — pickles a dict of `{"chunks": ..., "embeddings": ...}`
  to disk. Re-opening means no re-embedding.

- **`load(path)`** — unpickles the dict back into `chunks`/`embeddings`.

> The pickle format is `{"chunks": [Chunk, ...], "embeddings": np.ndarray}`.
> It is an **opaque, version-fragile format** — fine for a local prototype,
> not a shared/multi-process store. The README's "next steps" suggest
> swapping this for Chroma when it starts to feel clunky.

### `rag/generate.py`

The generation step, swappable between **Gemini** and **Bedrock** at runtime
via `GENERATION_BACKEND`.

- **`generate_answer(prompt) -> str`** — reads `GENERATION_BACKEND`
  (default `"gemini"`), lowercases and strips it, and dispatches to either
  `_generate_gemini` or `_generate_bedrock`. Raises `RuntimeError` for any
  other value. Intentionally the **only** interface both `api.py` and
  `query.py` use — neither knows which provider actually ran.

- **`_generate_gemini(prompt)`** — lazily imports `google.genai`, requires
  `GEMINI_API_KEY` (raises `RuntimeError` if unset), uses `GEMINI_MODEL`
  (default `gemini-2.5-flash`), and calls
  `client.models.generate_content(model, contents=prompt)`. Returns
  `response.text`. The `genai` client is created fresh on every call.

- **`_generate_bedrock(prompt)`** — lazily imports `boto3`, uses
  `AWS_REGION` (default `us-east-1`) and `BEDROCK_MODEL_ID` (code default
  `anthropic.claude-3-5-haiku-20241022-v1:0`). No explicit AWS keys are
  passed on purpose: `boto3` picks up credentials from env vars,
  `~/.aws/credentials`, or — on EC2 — the **IAM instance role** automatically.
  Calls the **Converse API** (`client.converse`) with
  `inferenceConfig={"maxTokens": 500}` and returns the first text block:
  `response["output"]["message"]["content"][0]["text"]`. The Converse API is
  provider-agnostic, so the same call shape works for Claude, Nova, etc.

### `api.py`

FastAPI app wrapping the pipeline for the browser and HTTP clients.

**Module-level setup:**
- `load_dotenv()` loads `.env` into `os.environ`.
- `INDEX_PATH = os.environ.get("INDEX_PATH", "index.pkl")`
- `DOCS_FOLDER = os.environ.get("DOCS_FOLDER", "sample_docs")`
- `TOP_K = 3` — default retrieval count for `/query`.
- `RETRIEVAL_STRATEGY`, `RETRIEVAL_TOP_K`, `HYBRID_ALPHA`, `MMR_LAMBDA`,
  `RETRIEVAL_MULTI_QUERY`, `RERANK_MODEL`, `RERANK_TOP_N` — live `/query`
  retrieval config, all env-overridable (see Configuration). The defaults
  reproduce the original vector/top-3 search; an unknown `RETRIEVAL_STRATEGY`
  falls back to `vector` with a log warning.
- `ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}`
- `DOCS_FOLDER` is created with `os.makedirs(..., exist_ok=True)` at import.
- `/static` is mounted at `/static`, and `static/index.html` is the root page.
- The `VectorStore` is created once at **import time** and loaded with
  `store.load(INDEX_PATH)`, swallowing `FileNotFoundError` so a fresh setup
  starts empty (the UI can add the first document without running
  `build_index.py` first). Loading the model + index once at startup avoids
  paying the multi-second embedding-model load cost per request.

**Helpers:**
- `_live_search(question)` — optionally rewrites the question
  (`RETRIEVAL_MULTI_QUERY`), averages the query embeddings, then calls
  `retrieval.rank(...)` with the retrieval constants above, so the live
  `/query` path scores exactly like the experiment runner. Returns
  `(Chunk, score)` pairs.
- `build_prompt(question, chunks)` — joins chunks as
  `[Source: <source>]\n<text>` blocks separated by blank lines and wraps them
  in a grounded-answer prompt that instructs the model to say so explicitly
  if the context doesn't contain the answer.
- `_add_chunks_to_index(chunks)` — drops any existing chunks from the same
  source(s), then calls `store.add(chunks)` and `store.save(INDEX_PATH)` so
  re-uploading a file replaces it instead of duplicating it.

**Pydantic models:** `QueryRequest {question}`,
`QueryResponse {answer, sources: [SourceRef]}`,
`SourceRef {source, chunk_id, score, snippet, text}`,
`UrlIngestRequest {url}`, `IngestResponse {filename, chunks_added, total_chunks}`.

### `build_index.py`

One-shot CLI. Usage:
- `python build_index.py` → chunks `sample_docs/`
- `python build_index.py my_docs_folder` → chunks your folder

Calls `build_chunks(folder, chunk_size=300, overlap=50)`, prints the chunk
count, creates a fresh `VectorStore`, `store.add(chunks)`, `store.save()`.
Note it uses its own `INDEX_PATH = os.environ.get("INDEX_PATH", "index.pkl")`
(same default as the API, so both share one file). **Caution: it embeds only
the folder given — it does not merge with an existing index.**

### `query.py`

Terminal question-asking. Usage:
`python query.py "What is HNSW used for?"`

Loads the store (same `INDEX_PATH` default), retrieves `RETRIEVAL_TOP_K`
chunks through the shared `retrieval.rank(...)`, and honours the same env
knobs as the web UI (`RETRIEVAL_STRATEGY`, `HYBRID_ALPHA`, `MMR_LAMBDA`,
`RETRIEVAL_MULTI_QUERY`, `RERANK_MODEL`, `RERANK_TOP_N`). It prints each
source as `  [<score:.3f>] <source> (chunk <chunk_id>)` with a short excerpt,
builds the prompt with `pipeline.build_prompt`, calls `generate_answer`, and
prints the answer. Calls `load_dotenv()` itself. Errors print straight to
stdout — there's no HTTP layer here.

### `fetch_url.py`

Downloads a URL's article text into `sample_docs/` for `build_index.py` to
pick up. Usage: `python fetch_url.py https://example.com/some-blog-post`.

Uses `extract_url_text(url)` + `slugify(title)`, writes
`# {title}\nSource: {url}\n\n{text}` to `sample_docs/<slug>.txt`, prints the
word count, and reminds you to re-run `build_index.py`. `ValueError`s
(un-downloadable / no article content) are printed and exit cleanly.
`OUTPUT_FOLDER = "sample_docs"` is hardcoded.

### `static/index.html`

Single-page, no-build frontend. All JavaScript is vanilla; all styling is
inline `<style>` with a dark "notebook" theme (CSS custom properties: `--ink`,
`--paper`, `--accent`, etc.).

**Layout:** an eyebrow label ("Retrieval-augmented search"), the title
("Ask your notes"), a subtitle, then a **source panel** with two tabs
`Upload file` / `Paste URL`, a status line, and a scrolling source list; then
a question input + `Ask` button; then the answer area.

**Behavior:**
- Tab switching toggles `.active` classes on the two tab buttons/panels.
- File tab: a click-to-browse **and** drag-and-drop zone (accepts
  `.pdf,.txt,.md`). Drag styling via `dragover`/`dragleave` classes.
- URL tab: a URL input; Enter key or the `Add` button submits.
- `uploadFile(file)` POSTs `FormData` to `/ingest/file`, shows
  `Added <name> (<n> chunks)` on success, refreshes the source list.
- `ingestUrl(url)` POSTs JSON to `/ingest/url` similarly.
- `loadSources()` GETs `/sources` and renders each name with a `Remove`
  button (HTML-escaped via a local `escapeHtml` helper — a `div.textContent`
  trick).
- `removeSource(filename)` DELETEs `/sources/<encoded-name>`, then reloads.
- Question form: disables the button, shows a pulsing `.dot`
  "Searching & thinking" state, POSTs JSON to `/query`, and renders
  `data.answer` HTML-escaped in an `#answer-wrap` block. Failures render a
  red "Something went wrong… Is the server running?" message.
- Errors surface from FastAPI's `data.detail` when present.
- A `prefers-reduced-motion` media query disables the loading-dot pulse.

All rendering is HTML-escaped, so answers/source names are shown as text.

---

## Installation

Requires Python ≥ 3.11.

```bash
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS/Linux:    source .venv/bin/activate

pip install -r requirements.txt
```

First embedding run downloads `all-MiniLM-L6-v2` from Hugging Face
(≈90 MB). Ensure `GENERATION_BACKEND` and the matching key/model are set in
`.env` before asking questions.

---

## Configuration (`.env`)

`.env` is read by `load_dotenv()` (in both `api.py` and `query.py`) and is
**gitignored**. All variables are optional — every one has a code-level
default — but answers fail without the right generation config.

| Variable | Default (code) | Used by | Meaning |
|---|---|---|---|
| `INDEX_PATH` | `index.pkl` | `api.py`, `build_index.py`, `query.py` | Where the pickled index is saved/loaded |
| `DOCS_FOLDER` | `sample_docs` | `api.py` | Where uploads/URL extractions are written |
| `RETRIEVAL_STRATEGY` | `vector` | `api.py`, `query.py` | `/query` ranking: `vector`, `bm25`, `hybrid` or `mmr` |
| `RETRIEVAL_TOP_K` | `3` | `api.py`, `query.py` | Number of chunks sent to the prompt |
| `HYBRID_ALPHA` | `0.5` | `api.py`, `query.py` | Vector vs BM25 blend for `hybrid` (1 = vector only) |
| `MMR_LAMBDA` | `0.7` | `api.py`, `query.py` | Relevance vs diversity for `mmr` (1 = relevance only) |
| `RETRIEVAL_MULTI_QUERY` | `0` (off) | `api.py`, `query.py` | LLM query rewrites, 0–5; averaged embeddings + unioned BM25 terms |
| `RERANK_MODEL` | — (off) | `api.py`, `query.py` | Cross-encoder model, e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `RERANK_TOP_N` | `RETRIEVAL_TOP_K` | `api.py`, `query.py` | Chunks kept after reranking |
| `GENERATION_BACKEND` | `gemini` | `rag/generate.py` | `gemini` or `bedrock` |
| `GEMINI_API_KEY` | — (error if unset) | `rag/generate.py` | Google AI Studio key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | `rag/generate.py` | Gemini model name |
| `AWS_REGION` | `us-east-1` | `rag/generate.py` | AWS region for Bedrock |
| `BEDROCK_MODEL_ID` | `anthropic.claude-3-5-haiku-20241022-v1:0` | `rag/generate.py` | Bedrock model ID |

The shipped `.env` currently sets:
- `GENERATION_BACKEND=gemini`
- `GEMINI_MODEL=gemini-2.5-flash`
- `AWS_REGION=ap-south-2`
- `BEDROCK_MODEL_ID=anthropic.claude-haiku-4-5-20251001-v1:0`

with commented sample values for pointing `INDEX_PATH`/`DOCS_FOLDER` at a
Docker volume (`/app/data/...`).

> **Heads-up:** `query.py`, `DEPLOY_AWS.md`, and `DEPLOY_DOCKER_BEDROCK.md`
> all reference a `.env.example` file, but **no `.env.example` exists in this
> repository** — you'll need to create one (or copy `.env`) before following
> those guides' `cp .env.example .env` steps.

---

## Running the web app

```bash
uvicorn api:app --reload
```

Then open **http://localhost:8000**. The `--reload` flag hot-reloads on code
changes (dev only). The Docker CMD form binds `0.0.0.0:8000`, which is what
makes it reachable from other machines / the browser on deployment.

### Browser workflow

1. **Add a source** — either drag/click a `.pdf`/`.txt`/`.md` file into the
   upload panel, or paste a blog-post URL and hit **Add**. Each source is
   chunked and embedded on the spot and appears in the indexed-sources list
   with a **Remove** button.
2. **Ask a question** — it's answered using *only* the sources you've added.
   A loading dot shows while retrieval + generation run.

### Pipeline lab (`/lab`)

The chatbot above uses one fixed pipeline. `/lab` runs the same question
through up to six configurations side by side:

- **Chunking** — fixed, recursive, sentence, markdown headings, or semantic
  (splits where neighbouring sentences stop being similar; adjustable
  threshold). Optional parent-child mode retrieves small chunks but sends the
  enclosing parent to the model.
- **Embeddings** — MiniLM, BGE small, E5 small (each model's required
  query/passage prefixes are applied automatically).
- **Retrieval** — vector, BM25, hybrid (adjustable alpha), or MMR (adjustable
  lambda), with optional cross-encoder reranking and a text/table metadata
  filter.
- **Trace** — chunk map, per-chunk scores, rerank order, prompt, answer, stage
  latencies, and a PCA scatter of the chunk embeddings.
- **Evaluation** — expected keywords produce retrieval and answer hit rates;
  known-answer snippets produce precision@k and recall@k; answers get a lexical
  grounding score (share of answer content words found in the context). The
  dropdown loads labelled questions and snippets from `eval/questions.json`.

Files can be uploaded from the lab page as well; uploads also enter the
chatbot's index, but running experiments never modifies it. Runs are appended
to `experiments.jsonl` and pruned at `MAX_EXPERIMENT_RUNS` (default 100);
individual runs can be deleted from the history dropdown. Extracted text,
chunks and embeddings are cached under `cache/`, keyed by file content, so
changing a retrieval setting is instant and re-running a config never
re-parses or re-embeds anything.

### Command-line workflow (alternative)

```bash
python build_index.py path/to/your/folder   # chunks + embeds the folder
python query.py "your question"              # ask from the terminal
python fetch_url.py https://…/some-post      # save an article into sample_docs/
```

### Table extraction (including nested tables)

PDFs with tables get special handling beyond the plain text extraction
above: `rag/ingest.py` also runs `pdfplumber` over every PDF, detects tables,
and turns each one into its own dedicated chunk — a clean Markdown table
instead of the jumbled, row/column-less text that plain extraction produces
for anything tabular.

For tables that look nested (a cell containing another table inside it —
something a flat grid parser reliably mangles), it falls back to rendering
that page as an image and asking a vision-capable model (Gemini) to transcribe
the table directly, preserving the nested structure. This adds one LLM call
per nested table found, only when needed — regular flat tables never trigger
it.

Table chunks show up in `build_index.py`'s output as a separate count
(`Created N chunks (X text, Y table)`) so you can see how many were found.

---

## HTTP API reference

Base URL: `http://localhost:8000` (or your deployed host).

### `GET /`
Serves `static/index.html` (the single-page UI). All routes are served by
FastAPI; the browser hits them directly.

### `GET /lab`
Serves `static/lab.html` (the experiment UI).

### `GET /experiments/options`
Returns the documents in `DOCS_FOLDER` plus the allowed chunking, embedding,
retrieval and rerank options — everything the lab UI needs to build a config.

### `POST /experiments/run`
JSON body:

```json
{
  "question": "What is chunking?",
  "docs": ["chunking_strategies.md"],
  "generate": false,
  "expected_keywords": ["chunking", "overlap"],
  "expected_snippets": ["embed individual sentences"],
  "configs": [
    {
      "chunking":  {"strategy": "recursive", "size": 400, "overlap": 60,
                    "threshold": 0.75, "parent_size": 0},
      "embedding": {"model": "BAAI/bge-small-en-v1.5"},
      "retrieval": {"strategy": "hybrid", "top_k": 4, "alpha": 0.5, "lambda": 0.7,
                    "chunk_type": "any"},
      "rerank":    {"model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "top_n": 3}
    }
  ]
}
```

`threshold` only affects semantic chunking; `parent_size` (words, 0 = off)
turns on parent-child chunking and must exceed the chunk size; `chunk_type`
is `any`, `text` or `table`. `expected_snippets` are known-answer phrases used
for precision@k (share of retrieved chunks containing one) and recall@k
(share of snippets found anywhere in the retrieved set).

Runs each config and returns the full trace (chunks, retrieved chunks with
scores, projection, prompt, answer, metrics, latencies). A config that fails
is returned as `{"config": …, "error": "…"}` rather than failing the run.
**400** for invalid configs, unknown documents, or more than 6 configs.

### `GET /experiments/eval`
Labelled questions from `eval/questions.json` — each with `question`, `docs`,
`snippets` and `keywords`. Returns an empty list when the file is absent.

### `GET /experiments?limit=20`
Summaries of saved runs, newest first.

### `GET /experiments/{run_id}`
One saved run, including every trace.

### `DELETE /experiments/{run_id}`
Deletes one saved run. **404** if the id doesn't exist.

### `GET /sources`
- Response: `{"sources": [<filename>, ...], "total_chunks": <int>}`
- `sources` is the deduped, insertion-ordered list of indexed filenames;
  `total_chunks` is `len(store.chunks)`.
- Returns `{"sources": [], "total_chunks": 0}` when nothing is indexed.

### `POST /ingest/file`
Multipart form field `file`. Accepts `.pdf`, `.txt`, `.md` only.
- **400** — unsupported extension (`body: "Unsupported file type '<ext>'. Use .pdf, .txt, or .md."`).
- Saves the file into `DOCS_FOLDER` (overwrites same-named files), extracts
  text (PDF via `pypdf`, otherwise UTF-8 read).
- **400** — no extractable text
  (`body: "No extractable text found in that file. If it's a scanned PDF, it needs OCR first."`).
- Chunks, embeds, saves the index, and returns
  `{"filename": <name>, "chunks_added": <int>, "total_chunks": <int>}`.

### `POST /ingest/url`
JSON body: `{"url": "https://…"}`.
- **400** — `ValueError` from `extract_url_text` (un-downloadable or no
  article content); the message is passed through.
- Saves `# <title>\nSource: <url>\n\n<text>` into
  `DOCS_FOLDER/<slug>.txt` (slug derived from the article title, max 60 chars,
  fallback `"page"`), chunks (title already stripped from chunking since only
  `text` is passed), embeds, persists, and returns the same shape as file ingest.

### `POST /query`
JSON body: `{"question": "…"}`.
- If `store.chunks` is empty → **200** with
  `answer = "Nothing's been indexed yet — add a file or URL above first."`
  (the CLI prints a similar message and exits).
- Otherwise retrieves `RETRIEVAL_TOP_K` chunks with `RETRIEVAL_STRATEGY`
  (default: vector, top 3 — the original behaviour), optionally rewriting the
  query (`RETRIEVAL_MULTI_QUERY`) and reranking, builds the grounded prompt,
  and calls `generate_answer`.
- If generation raises → **200** with
  `answer = "ERROR generating answer: {e}"` (errors are returned as answers,
  not as HTTP errors).
- On success →
  `{"answer": "<generated text>", "sources": [{"source", "chunk_id", "score", "snippet", "text"}]}`.
  Each source is one retrieved chunk: `snippet` is a 240-character excerpt and
  `text` is the full chunk (or its parent when parent-child chunking is on).

### `DELETE /sources/{filename}`
Removes one source (and all its chunks/embeddings) from the store, persists,
and returns `{"removed": <filename>, "total_chunks": <int>}`.
- **404** — if `filename` isn't currently indexed
  (`body: "'{filename}' isn't currently indexed."`).

---

## Data & persistence

- **Index file (`index.pkl`)** — a pickle of
  `{"chunks": [Chunk, ...], "embeddings": np.ndarray}`. Saved after every
  ingest/delete via the API, and by `build_index.py`. Loaded once at API
  startup so everything you add survives restarts. (Currently ≈6 KB for the
  sample docs.) It is **gitignored** and **excluded from Docker**.
- **Uploaded/extracted documents (`DOCS_FOLDER` = `sample_docs/`)** — file
  uploads land here verbatim; URL ingests write
  `# <title>\nSource: <url>\n\n<text>` files here. These are *not*
  re-ingested at startup — the index is self-contained.
- **Memory** — chunks + embeddings live in RAM for the process lifetime
  (numpy arrays). Large corpora will eat memory accordingly.
- **Experiment cache (`cache/`)** — extracted text, chunks and embeddings
  keyed by file content plus chunk config (and model, for semantic chunking).
  Gitignored; delete it or bump `_CACHE_VERSION` in `rag/pipeline.py` after
  changing extraction or chunking code.
- **Run history (`experiments.jsonl`)** — one JSON line per run, including
  full traces and chunk text. Gitignored; pruned at `MAX_EXPERIMENT_RUNS`.
- **Eval set (`eval/questions.json`)** — questions, the documents they target,
  and known-answer snippets. Override the path with `EVAL_PATH`.
- **Concurrency** — there is no locking around `store.add/save`; two
  simultaneous ingests could in principle race on the `index.pkl` write.
  Fine for single-user use.

---

## Docker

**`Dockerfile`** (python:3.11-slim):
1. `WORKDIR /app`
2. Copies `requirements.txt`, runs `pip install --no-cache-dir -r requirements.txt`
   first (separate layer, so Docker layer-caches the heavy torch/transformers
   install and only re-runs it when `requirements.txt` changes).
3. Copies the rest of the project.
4. `RUN mkdir -p sample_docs` (folder that uploads are written to must exist).
5. `EXPOSE 8000` and
   `CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]`.

**`.dockerignore`** excludes `.env`, `index.pkl`, `__pycache__/`, `*.pyc`,
`.venv/`, `venv/`, `.git/`, `.gitignore`, and **`*.md`** — so the README and
deployment docs are *not* in the image, and local `.env`/index secrets aren't
baked in. You supply `.env` at runtime with `--env-file` and mount a volume
for persistence (see the Docker deployment guide).

```bash
docker build -t rag-project .
docker run -d --name rag-app --restart unless-stopped -p 8000:8000 \
  --env-file .env -v "$(pwd)/data":/app/data rag-project
```

For the volume mount to matter, `.env` must set
`INDEX_PATH=/app/data/index.pkl` and `DOCS_FOLDER=/app/data/sample_docs`.

---

## Deployment

Three documented paths — **same core app either way**; only `GENERATION_BACKEND`
in `.env` differs.

- **[`DEPLOY_AWS.md`](./DEPLOY_AWS.md)** — one EC2 instance (Ubuntu 22.04,
  t3.small), a plain Python venv, and **Gemini** for generation. Covers
  launching the instance, installing deps, adding your key, running uvicorn
  as a `systemd` service (`/etc/systemd/system/rag.service`, restart on boot/
  crash), testing, and optional next steps (Elastic IP, nginx on :80, HTTPS
  via certbot, S3 for the index). Also has cost notes and troubleshooting
  (security-group port 8000, disk-full recovery with `growpart`, and reading
  crash logs with `journalctl`).
- **[`DEPLOY_DOCKER_BEDROCK.md`](./DEPLOY_DOCKER_BEDROCK.md)** — same EC2
  idea, but the app runs in **Docker** and answers come from **Amazon
  Bedrock (Claude)**, with **no API key to manage** — an IAM role attached to
  the instance lets the container call Bedrock automatically. Covers the
  one-time Anthropic use-case submission (if required), the IAM role setup,
  installing Docker, `.env` for Bedrock + volume paths, the exact
  `docker run` flags, updates after code changes, and Bedrock-specific
  troubleshooting (`AccessDeniedException`, model-ID vs cross-region
  inference-profile ID, container crash logs, "works locally but not on EC2").
- **[`DEPLOY_ECS_EXPRESS_KB.md`](./DEPLOY_ECS_EXPRESS_KB.md)** — the
  fully-managed path: ECS Express Mode + Bedrock Knowledge Bases (S3 + Titan
  embeddings + OpenSearch Serverless) replace the local ingestion/retrieval
  pipeline entirely (`api_kb.py` instead of `api.py`). Most infrastructure,
  least code you wrote yourself — read the trade-off note in that guide
  before committing to it. (Replaces the earlier App Runner plan; App Runner
  closed to new customers on April 30, 2026.)

---

## Sample documents

`sample_docs/` ships three files for the CLI path:
- `vector_databases.md` — explains vector DBs, why they matter for RAG, and
  ANN indexing (HNSW/IVF).
- `chunking_strategies.md` — fixed-size vs structure-aware vs semantic
  chunking, with trade-offs.
- `New Text Document.txt` — a personal note in your workspace.

> **Privacy warning:** the `.txt` sample contains what appears to be private,
> personal note content about a third party. It is included in any
> `build_index.py` run over the default folder and would be queryable through
> the index. Consider removing it (and any index built from it) if you plan
> to share or deploy this repository.

---

## Important details & gotchas

- **Two different chunk sizes.** The CLI path (`build_index.py`) uses
  `chunk_size=300, overlap=50`; the browser/API path uses the module default
  `chunk_size=500, overlap=50` (`api.py` calls `chunk_single_document`
  without overriding). This is intentional in spirit ("tune chunk size") but
  easy to miss — expect different chunk counts between the two entry points.
- **Chunking is by whitespace words**, not tokens. Multi-byte/markdown-laden
  text counts differently than a tokenizer would.
- **`build_index.py` always rebuilds from scratch.** It creates a new store
  and adds only the given folder — it does *not* merge with any existing
  `index.pkl`. Use the API for incremental adds.
- **Similarity is cosine similarity** via dot product, valid only because
  embeddings are encoded with `normalize_embeddings=True`.
- **Reference integrity:** `api.py` and `query.py` both use
  `pipeline.build_prompt`; only one copy of the prompt text exists.
- **`/query` shares the lab's ranking code, not its chunking or embedding
  choices.** The live index is built by `build_index.py` / `/ingest/file` with
  fixed-size chunks and one embedding model. `RETRIEVAL_STRATEGY`,
  `RETRIEVAL_TOP_K`, `RETRIEVAL_MULTI_QUERY` and `RERANK_MODEL` only change how
  those chunks are ranked; contextual chunks, parent-child context and semantic
  chunking exist only in the lab.
- **Startup cost.** The embedding model is loaded once at API import; the
  first request after boot is fast, but *process start* takes a couple of
  seconds plus whatever it takes to unpickle the index.
- **PDFs are text-layer only** — `pypdf` returns `""` for image-only pages
  rather than erroring, so a scanned PDF silently yields nothing (the API
  catches the empty-text case with a 400; the CLI doesn't).
- **URL ingests write files** to `DOCS_FOLDER` but the source name shown is
  the `<slug>.txt` filename, not the URL.
- **Re-uploads replace, they don't merge:** uploading a file whose name is
  already indexed drops the old chunks first, so the store never holds two
  copies of the same document.
- **`sample_docs/` name collision with `DOCS_FOLDER`.** The default
  `DOCS_FOLDER` is `sample_docs` — uploads land in the same folder the CLI
  reads. That's by design (one shared folder), but be aware files there are
  *not* auto-indexed at startup; only API ingests modify the live index.
- **No `.env.example`** exists despite being referenced by the CLI docstrings
  and both deployment guides.

---

## Security & privacy notes

- `.env` and `index.pkl` are gitignored and `.dockerignore`d — **do not** add
  them to version control or bake them into images.
- The current `GEMINI_API_KEY` value in the local `.env` does not match the
  usual Google AI Studio format (`AIza…`) — it may be invalid or from a
  different provider. If `GEMINI_API_KEY is not set` / auth errors appear,
  grab a fresh key from Google AI Studio and update `.env`.
- Generation errors are returned to the browser as the answer body
  (`ERROR generating answer: …`) — this can leak internal details. Safe for a
  local prototype; consider logging server-side and returning a generic
  message before exposing it publicly.
- The sample note contains personal/private content — remove before sharing.

---

## Known limitations

- No OCR — scanned/image PDFs won't work (would need `pytesseract` +
  preprocessing).
- The live chatbot still uses fixed-size word chunking. `/query` retrieval is
  configurable (strategy, top-k, multi-query, reranking) but the lab's chunking
  modes, contextual chunks, parent-child context and text/table filter stay in
  `/lab`.
- Contextual chunking is lab-only: it changes what gets embedded, so the live
  index would have to be rebuilt to use it. Multi-query rewrites need a working
  generation backend; a failed rewrite falls back to the original question.
- The vector store is an in-memory numpy array persisted with pickle — no
  incremental `load` merge, no approximate-nearest-neighbor index (HNSW/IVF),
  no horizontal scaling, no transactional safety. At a few hundred chunks this
  is fine; beyond that, move to Chroma/Qdrant/pgvector.
- Evaluation is hint-based: keyword hit rates, snippet-based precision@k /
  recall@k, and a *lexical* grounding score. None of them judge whether a
  paraphrase is faithful.
- Experiments are single-process. Each run stores a full trace in
  `experiments.jsonl`; the oldest runs are pruned at `MAX_EXPERIMENT_RUNS`
  (default 100), and individual runs can be deleted from the lab or via
  `DELETE /experiments/{run_id}`.
- Single-process, single-user: no auth, no rate limiting, no multi-user
  separation, no locking on the index writes.

---

## Suggested next steps

1. **Break it on purpose.** Ask a question with no answer in the docs — does
   the model say "I don't know" or hallucinate? Tighten the prompt if it
   hallucinates.
2. **Tune chunk size in the lab.** Run the same question at 100, 400 and 800
   words and watch retrieval scores, answer grounding and latency move.
3. **Add more docs** and notice when 3-chunk retrieval starts missing the
   right passage — your signal to compare BM25, hybrid and reranking in `/lab`.
4. **Swap the vector store.** Once `index.pkl` feels clunky, replace
   `store.py`'s numpy array with Chroma — same interface, better persistence
   and scaling.
5. **Extend the eval set.** Add questions with known-answer snippets to
   `eval/questions.json` and watch precision@k / recall@k after every change
   instead of eyeballing answers.

---

## Troubleshooting

- **`GEMINI_API_KEY is not set in .env`** → add the key to `.env` and
  restart uvicorn (the key is read at every request, but the process must
  `load_dotenv()` at startup).
- **Answer shows `ERROR generating answer: …`** → check the backend setting
  and the provider error text; for Bedrock, see the deployment guide's
  troubleshooting (IAM role, model-ID vs cross-region profile ID, marketplace
  use-case form).
- **PDF adds but returns few/no words** → scanned PDF; needs OCR.
- **`Unsupported file type`** → use `.pdf`, `.txt`, or `.md`; URL-only and
  other formats are rejected.
- **`Nothing's been indexed yet`** → no chunks in the store; add a source
  first.
- **First run downloads slowly** → `all-MiniLM-L6-v2` is fetched from Hugging
  Face; subsequent runs are offline.
- **Port 8000 refused externally (deployed)** → check the EC2 security group
  opens Custom TCP 8000 to your IP/Anywhere.
- **CLI says empty index after `build_index.py` ran** → confirm you ran
  `query.py` in the same directory / with the same `INDEX_PATH` (default
  `index.pkl`).
