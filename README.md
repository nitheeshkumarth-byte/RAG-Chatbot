# Minimal RAG Project

A from-scratch RAG pipeline: chunk documents → embed locally → retrieve by
similarity → generate a grounded answer with Claude. No LangChain, no vector
DB server — just enough code to see every step clearly.

## Run it

```bash
uvicorn api:app --reload
```

Open **http://localhost:8000**. From there:

1. **Add a source** — either drag/click a `.pdf`/`.txt`/`.md` file into the
   upload panel, or paste a blog post URL and hit Add. Each one gets chunked
   and embedded on the spot; you'll see it appear in the indexed-sources
   list below the panel, with a "Remove" button if you want to drop it.
2. **Ask a question** in the box underneath — it's answered using only the
   sources you've added.

The index persists to `index.pkl`, so anything you add is still there next
time you start the server. No separate build step needed anymore — the old
`build_index.py`/`sample_docs/` workflow still works too (see below) if you
prefer preparing documents from the command line instead of the browser.

## Command-line workflow (optional)

If you'd rather not use the browser upload panel, you can still prep a
folder of documents ahead of time:

```bash
python build_index.py path/to/your/folder    # chunks + embeds everything in the folder
python query.py "your question"               # ask from the terminal
```

### A note on PDFs

`pypdf` extracts text directly from the PDF's internal structure — it does
**not** do OCR. That means it works well for PDFs that were generated
digitally (exported from Word, LaTeX, a web page, etc.) but will return
empty or near-empty text for scanned/photographed pages, since there's no
actual text layer to pull from. If `build_index.py` reports suspiciously
few words for a PDF you added, that's the likely cause — you'd need an OCR
step (e.g. `pytesseract`) before this pipeline can use it.

## How it works

```
Browser upload/URL       your_docs/*.md (CLI path)
       │                          │
       ▼                          ▼
  /ingest/file  ┐          ingest.py    → splits each source into
  /ingest/url   ┴────────────────┴────────overlapping ~300-word chunks
                                  │
                                  ▼
                            store.py     → embeds each chunk with
                                            sentence-transformers, stores
                                            vectors in numpy (index.pkl)
                                  │
                                  ▼
                          /query OR      → embeds your question, finds
                          query.py         the top-3 most similar chunks,
                                            stuffs them into a prompt,
                                            sends it to Gemini for a
                                            grounded answer

Both the browser upload panel and the CLI (build_index.py / fetch_url.py)
funnel into the same chunk → embed → store pipeline — ingestion is the
only layer that knows about file formats or URLs.
```

## Deploying it

Two deployment paths, same code either way (`GENERATION_BACKEND` in `.env`
switches between them):

- [`DEPLOY_AWS.md`](./DEPLOY_AWS.md) — EC2 with a plain Python venv, Gemini for generation
- [`DEPLOY_DOCKER_BEDROCK.md`](./DEPLOY_DOCKER_BEDROCK.md) — EC2 with Docker, Amazon Bedrock (Claude) for generation, no API key to manage (uses an IAM role instead)

## Things to try next, in order

1. **Break it on purpose.** Ask a question with no answer in the docs — does
   the model say "I don't know" or hallucinate? Tighten the prompt if it
   hallucinates.
2. **Tune chunk size.** Change `chunk_size` in `build_index.py` from 300 to
   100, then to 800. Re-run and see how retrieval quality (the `[score]`
   values printed) and answer quality change.
3. **Add more docs** and see when a 3-chunk retrieval starts missing the
   right passage — that's your signal to add hybrid (keyword + vector)
   search or a reranker.
4. **Swap the vector store.** Once `index.pkl` feels clunky, replace
   `store.py`'s numpy array with `chromadb` — same interface, but it scales
   further and persists more robustly.
5. **Build an eval set.** Write 15-20 questions with known answers from your
   docs, and check pass/fail after every change instead of eyeballing it.
