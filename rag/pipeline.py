"""
pipeline.py — run one configurable RAG pipeline and return its full trace.

This is the experiment layer. It does NOT touch api.py's live VectorStore /
index.pkl: every run builds its own chunk list and embeddings (cached by
content hash), so comparing configurations never overwrites the chatbot's
index.

Caches are keyed by content, not by experiment:
    cache/extracted/<sha>.json                      text + tables per document
    cache/chunks/<sha>_<chunkcfg>.json              chunks for one strategy
    cache/embeddings/<sha>_<chunkcfg>_<model>.npy   embeddings for those chunks

That means changing the retrieval strategy is instant, and embedding the same
document twice for two configs costs nothing the second time.
"""

import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np

from rag import retrieval
from rag.chunking import STRATEGIES, split_text, word_count
from rag.generate import generate_answer
from rag.ingest import Chunk, extract_pdf_text, extract_tables_from_pdf
from rag.store import encode_passages, encode_query

logger = logging.getLogger(__name__)

DOCS_FOLDER = os.environ.get("DOCS_FOLDER", "sample_docs")
CACHE_DIR = os.environ.get("CACHE_DIR", "cache")
EXPERIMENTS_PATH = os.environ.get("EXPERIMENTS_PATH", "experiments.jsonl")
EVAL_PATH = os.environ.get("EVAL_PATH", os.path.join("eval", "questions.json"))
MAX_RUNS = int(os.environ.get("MAX_EXPERIMENT_RUNS", "100"))
_WRITE_LOCK = threading.Lock()

EMBEDDING_MODELS = [
    {"id": "all-MiniLM-L6-v2", "label": "all-MiniLM-L6-v2 — 384d, fast"},
    {"id": "BAAI/bge-small-en-v1.5", "label": "BGE small — 384d, stronger retrieval"},
    {"id": "intfloat/e5-small-v2", "label": "E5 small — 384d, needs query/passage prefixes"},
]
RERANK_MODELS = [
    {"id": "cross-encoder/ms-marco-MiniLM-L-6-v2", "label": "MS MARCO MiniLM L6 — balanced"},
    {"id": "cross-encoder/ms-marco-MiniLM-L-2-v2", "label": "MS MARCO MiniLM L2 — faster"},
]
_EMBEDDING_IDS = {m["id"] for m in EMBEDDING_MODELS}
_RERANK_IDS = {m["id"] for m in RERANK_MODELS}
_DOC_EXTENSIONS = {".pdf", ".txt", ".md"}
_CANDIDATE_POOL = 20
# Contextual chunking sends this much of the document to the LLM per chunk.
# Bigger documents get truncated; the chunk itself is always included.
_CONTEXT_DOC_CHARS = 4000
# Bump when extraction/chunking code changes so old caches are not reused.
_CACHE_VERSION = 1


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def list_documents() -> List[Dict[str, Any]]:
    """Every ingestable file currently sitting in DOCS_FOLDER."""
    docs = []
    if not os.path.isdir(DOCS_FOLDER):
        return docs
    for name in sorted(os.listdir(DOCS_FOLDER)):
        path = os.path.join(DOCS_FOLDER, name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(name)[1].lower() not in _DOC_EXTENSIONS:
            continue
        docs.append({"name": name, "size": os.path.getsize(path), "sha256": _sha256(path)})
    return docs


def _cache_file(kind: str, key: str, ext: str) -> str:
    return os.path.join(CACHE_DIR, kind, f"{key}.{ext}")


def _read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def _extract_document(name: str, sha: str) -> Dict[str, Any]:
    """Text + table text for a document, cached by content hash."""
    path = _cache_file("extracted", f"v{_CACHE_VERSION}_{sha}", "json")
    cached = _read_json(path)
    if cached is not None:
        return cached

    file_path = os.path.join(DOCS_FOLDER, name)
    ext = os.path.splitext(name)[1].lower()
    tables: List[str] = []

    if ext == ".pdf":
        text = extract_pdf_text(file_path)
        try:
            tables = [c.text for c in extract_tables_from_pdf(file_path, name)]
        except Exception as e:
            logger.warning(f"Table extraction failed for {name!r}: {e}")
    else:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError as e:
            raise ValueError(f"{name} is not valid UTF-8 text.") from e

    if not text.strip() and not tables:
        raise ValueError(f"No extractable text in {name}. Scanned PDFs need OCR first.")

    data = {"text": text, "tables": tables}
    _write_json(path, data)
    return data


def _chunk_key(sha: str, chunking: Dict[str, Any]) -> str:
    key = (
        f"v{_CACHE_VERSION}_{sha}_"
        f"{chunking['strategy']}-{chunking['size']}-{chunking['overlap']}"
    )
    if chunking["strategy"] == "semantic":
        key += f"-s{chunking['threshold']}-{_model_slug(chunking['model'])}"
    if chunking.get("parent_size"):
        key += f"-p{chunking['parent_size']}"
    return key


def _assign_parents(items: List[Dict[str, Any]], parent_size: int, text: str) -> None:
    """Tag each child chunk with the text of the parent span that contains it."""
    group: List[Dict[str, Any]] = []
    words = 0

    def flush() -> None:
        if not group:
            return
        start, end = group[0]["start"], group[-1]["end"]
        parent = text[start:end] if end > start else "".join(i["text"] for i in group)
        for item in group:
            item["parent"] = parent

    for item in items:
        w = word_count(item["text"])
        if group and words + w > parent_size:
            flush()
            group, words = [], 0
        group.append(item)
        words += w
    flush()


def _get_chunks(name: str, sha: str, chunking: Dict[str, Any]) -> List[Chunk]:
    """Chunk a document with one strategy, cached by (content, strategy)."""
    key = _chunk_key(sha, chunking)
    path = _cache_file("chunks", key, "json")
    cached = _read_json(path)
    if cached is None:
        doc = _extract_document(name, sha)
        pieces = split_text(
            doc["text"], chunking["strategy"], chunking["size"], chunking["overlap"],
            chunking.get("model"), chunking.get("threshold", 0.75),
        )

        cached = [
            {"text": p.text, "start": p.start, "end": p.end, "type": "text"}
            for p in pieces
        ]
        if chunking.get("parent_size"):
            _assign_parents(cached, chunking["parent_size"], doc["text"])
        cached += [
            {"text": t, "start": 0, "end": 0, "type": "table"}
            for t in doc["tables"]
        ]
        _write_json(path, cached)

    if chunking.get("contextual"):
        _attach_context(cached, name, sha, key)

    strategy = chunking["strategy"]
    return [
        Chunk(
            text=item["text"],
            source=name,
            chunk_id=i,
            chunk_type=item.get("type", "text"),
            start_char=item.get("start", 0),
            end_char=item.get("end", 0),
            strategy=strategy,
            parent_text=item.get("parent", ""),
            embed_text=(
                f"{item['context']}\n\n{item['text']}"
                if item.get("context") else ""
            ),
        )
        for i, item in enumerate(cached)
    ]


_CONTEXT_PROMPT = """Here is a document:
<document>
{doc}
</document>

Here is one chunk taken from it:
<chunk>
{chunk}
</chunk>

Write a single short sentence (at most 30 words) that situates this chunk in
the document, so it can be found by search. Reply with the sentence only."""


def _attach_context(cached: List[Dict[str, Any]], name: str, sha: str,
                    chunk_key: str) -> None:
    """LLM context sentence per chunk, cached by document + chunk config.

    The sentence is embedded with the chunk but not shown or sent to the
    answer prompt, so chunk text and offsets stay untouched.
    """
    path = _cache_file("contextual", f"v{_CACHE_VERSION}_{chunk_key}", "json")
    contexts = _read_json(path)
    if contexts is None:
        doc_text = _extract_document(name, sha)["text"][:_CONTEXT_DOC_CHARS]
        contexts = []
        for item in cached:
            if item.get("type", "text") != "text":
                contexts.append("")
                continue
            prompt = _CONTEXT_PROMPT.format(doc=doc_text, chunk=item["text"])
            contexts.append(" ".join(generate_answer(prompt).split())[:300])
        _write_json(path, contexts)
    for item, context in zip(cached, contexts):
        item["context"] = context


def _model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_")


def _get_embeddings(sha: str, chunking: Dict[str, Any], model_name: str,
                    chunks: List[Chunk]) -> np.ndarray:
    key = f"{_chunk_key(sha, chunking)}_{_model_slug(model_name)}"
    if chunking.get("contextual"):
        key += "-ctx"
    path = _cache_file("embeddings", key, "npy")
    try:
        return np.load(path)
    except (OSError, ValueError):
        pass

    matrix = encode_passages(model_name, [c.embed_text or c.text for c in chunks])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npy"
    np.save(tmp, matrix)
    os.replace(tmp, path)
    return matrix


# ---------------------------------------------------------------------------
# Config handling
# ---------------------------------------------------------------------------
def normalize_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Validate one experiment config and fill in defaults.

    Raises ValueError with a user-facing message on bad input, since this is
    the trust boundary between the browser and the pipeline.
    """
    if not isinstance(raw, dict):
        raise ValueError("Each config must be an object.")

    ch = raw.get("chunking") or {}
    em = raw.get("embedding") or {}
    rt = raw.get("retrieval") or {}
    rr = raw.get("rerank")

    strategy = str(ch.get("strategy", "fixed"))
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown chunking strategy '{strategy}'.")
    size = int(ch.get("size", 500))
    overlap = int(ch.get("overlap", 50))
    if not 20 <= size <= 2000:
        raise ValueError("Chunk size must be between 20 and 2000 words.")
    if not 0 <= overlap < size:
        raise ValueError("Chunk overlap must be >= 0 and smaller than the chunk size.")

    threshold = float(ch.get("threshold", 0.75))
    if not 0 <= threshold <= 1:
        raise ValueError("Semantic break threshold must be between 0 and 1.")
    parent_size = int(ch.get("parent_size") or 0)
    if parent_size and not size < parent_size <= 4000:
        raise ValueError("Parent size must be larger than the chunk size (max 4000 words).")
    contextual = bool(ch.get("contextual", False))

    model = str(em.get("model", EMBEDDING_MODELS[0]["id"]))
    if model not in _EMBEDDING_IDS:
        raise ValueError(f"Unknown embedding model '{model}'.")

    r_strategy = str(rt.get("strategy", "vector"))
    if r_strategy not in retrieval.RETRIEVAL_STRATEGIES:
        raise ValueError(f"Unknown retrieval strategy '{r_strategy}'.")
    top_k = int(rt.get("top_k", 4))
    if not 1 <= top_k <= _CANDIDATE_POOL:
        raise ValueError(f"Top K must be between 1 and {_CANDIDATE_POOL}.")
    alpha = float(rt.get("alpha", 0.5))
    lam = float(rt.get("lambda", 0.7))
    if not 0 <= alpha <= 1 or not 0 <= lam <= 1:
        raise ValueError("Hybrid alpha and MMR lambda must be between 0 and 1.")
    chunk_type = str(rt.get("chunk_type", "any"))
    if chunk_type not in ("any", "text", "table"):
        raise ValueError("Chunk type filter must be 'any', 'text' or 'table'.")
    multi_query = int(rt.get("multi_query", 0))
    if not 0 <= multi_query <= 5:
        raise ValueError("Multi-query rewrites must be between 0 (off) and 5.")

    config = {
        "chunking": {
            "strategy": strategy,
            "size": size,
            "overlap": overlap,
            "threshold": threshold,
            "parent_size": parent_size,
            "contextual": contextual,
            "model": model,  # semantic chunking borrows the embedding model
        },
        "embedding": {"model": model},
        "retrieval": {
            "strategy": r_strategy,
            "top_k": top_k,
            "alpha": alpha,
            "lambda": lam,
            "chunk_type": chunk_type,
            "multi_query": multi_query,
        },
        "rerank": None,
    }

    if isinstance(rr, dict) and rr.get("model"):
        r_model = str(rr["model"])
        if r_model not in _RERANK_IDS:
            raise ValueError(f"Unknown rerank model '{r_model}'.")
        top_n = int(rr.get("top_n", 3))
        if top_n < 1:
            raise ValueError("Rerank top N must be at least 1.")
        # Never ask for more reranked chunks than retrieval returned.
        config["rerank"] = {"model": r_model, "top_n": min(top_n, top_k)}

    return config


# ---------------------------------------------------------------------------
# Prompt / metrics
# ---------------------------------------------------------------------------
def build_prompt(question: str, chunks: List[Chunk]) -> str:
    context = "\n\n".join(
        f"[Source: {c.source}]\n{c.parent_text or c.text}" for c in chunks
    )
    return f"""Answer the question using ONLY the context below. If the context
doesn't contain the answer, say so explicitly instead of guessing.

Context:
{context}

Question: {question}

Answer:"""


def _hit_rate(keywords: List[str], texts: List[str]) -> float | None:
    if not keywords:
        return None
    blob = "\n".join(texts).lower()
    found = sum(1 for k in keywords if k.lower() in blob)
    return round(found / len(keywords), 3)


_STOPWORDS = frozenset(
    """a an the and or but if then than that this these those of to in on at
    for from by with as is are was were be been being it its their there they
    we you i not no do does did can could should would will may might must
    about into over under more most other some such only also very""".split()
)


def _precision_at_k(snippets: List[str], retrieved: List[Dict[str, Any]]) -> float | None:
    """Share of retrieved chunks that contain at least one known answer snippet."""
    if not snippets:
        return None
    if not retrieved:
        return 0.0
    hits = sum(
        1 for r in retrieved
        if any(s.lower() in r["text"].lower() for s in snippets)
    )
    return round(hits / len(retrieved), 3)


def _recall_at_k(snippets: List[str], retrieved: List[Dict[str, Any]]) -> float | None:
    """Share of known answer snippets found anywhere in the retrieved chunks."""
    if not snippets:
        return None
    blob = "\n".join(r["text"] for r in retrieved).lower()
    return round(sum(1 for s in snippets if s.lower() in blob) / len(snippets), 3)


def _grounding(answer: str, context: str) -> float | None:
    """Share of the answer's content words that also occur in the context.

    Lexical, not semantic: a faithful paraphrase still scores low.
    """
    if not answer.strip():
        return None
    words = [w for w in retrieval.tokenize(answer) if w not in _STOPWORDS]
    if not words:
        return None
    available = set(retrieval.tokenize(context))
    return round(sum(1 for w in words if w in available) / len(words), 3)


def _project(matrix: np.ndarray) -> List[List[float]]:
    """2D PCA with numpy's SVD — enough for a scatter plot, no extra dependency."""
    if len(matrix) == 0:
        return []
    if len(matrix) == 1:
        return [[0.0, 0.0]]
    centered = matrix - matrix.mean(axis=0)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        xy = centered @ vt[:2].T
    except np.linalg.LinAlgError:
        return [[0.0, 0.0] for _ in matrix]
    return [[round(float(x), 4), round(float(y), 4)] for x, y in xy]


_REWRITE_PROMPT = """Rewrite this search question in {count} different ways.
One rewrite per line, no numbering, no explanation. Keep the meaning and the
key terms; vary the wording.

Question: {question}

Rewrites:"""


def rewrite_queries(question: str, count: int) -> List[str]:
    """LLM rewrites for multi-query retrieval, cached by question + count.

    Returns [] when generation fails, so a rate-limited or offline run still
    retrieves with the original question instead of failing.
    """
    digest = hashlib.sha256(question.strip().lower().encode()).hexdigest()[:24]
    path = _cache_file("rewrites", f"v{_CACHE_VERSION}_{digest}_n{count}", "json")
    cached = _read_json(path)
    if cached is not None:
        return cached
    try:
        raw = generate_answer(_REWRITE_PROMPT.format(count=count, question=question))
    except Exception as e:
        logger.warning(f"Multi-query rewrite failed, using the question only: {e}")
        return []
    rewrites = []
    for line in raw.splitlines():
        text = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip("\"'")
        if text and text.lower() != question.strip().lower():
            rewrites.append(text)
    rewrites = rewrites[:count]
    _write_json(path, rewrites)
    return rewrites


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run_config(config: Dict[str, Any], question: str, docs: List[Dict[str, str]],
               generate: bool = True, keywords: List[str] | None = None,
               snippets: List[str] | None = None) -> Dict[str, Any]:
    """Execute one config against one question and return its trace."""
    keywords = keywords or []
    snippets = snippets or []
    started = time.perf_counter()
    latencies = {
        "prepare_ms": 0, "embed_ms": 0, "rewrite_ms": 0,
        "retrieve_ms": 0, "rerank_ms": 0, "generate_ms": 0,
    }

    all_chunks: List[Chunk] = []
    matrices = []
    for doc in docs:
        t0 = time.perf_counter()
        chunks = _get_chunks(doc["name"], doc["sha256"], config["chunking"])
        latencies["prepare_ms"] += (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        embedding = _get_embeddings(doc["sha256"], config["chunking"], config["embedding"]["model"], chunks)
        latencies["embed_ms"] += (time.perf_counter() - t0) * 1000
        all_chunks.extend(chunks)
        matrices.append(embedding)

    if not all_chunks:
        raise ValueError("No chunks were produced from the selected documents.")

    embeddings = np.vstack(matrices) if len(matrices) > 1 else matrices[0]
    texts = [c.text for c in all_chunks]

    wanted_type = config["retrieval"].get("chunk_type", "any")
    allowed = np.ones(len(all_chunks), dtype=bool)
    if wanted_type != "any":
        allowed = np.array([c.chunk_type == wanted_type for c in all_chunks])
        if not allowed.any():
            raise ValueError(f"No '{wanted_type}' chunks in the selected documents.")

    t0 = time.perf_counter()
    rewrites: List[str] = []
    if config["retrieval"]["multi_query"]:
        rewrites = rewrite_queries(question, config["retrieval"]["multi_query"])
        latencies["rewrite_ms"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()

    query_vec = encode_query(config["embedding"]["model"], question)
    if rewrites:
        query_vec = retrieval.mean_query_vector([
            query_vec,
            *[encode_query(config["embedding"]["model"], q) for q in rewrites],
        ])
    final, details, rerank_ms = retrieval.rank(
        question, texts, embeddings, query_vec,
        strategy=config["retrieval"]["strategy"],
        top_k=config["retrieval"]["top_k"],
        alpha=config["retrieval"]["alpha"],
        lam=config["retrieval"]["lambda"],
        candidate_pool=_CANDIDATE_POOL,
        allowed=allowed,
        bm25_query=" ".join([question] + rewrites),
        reranker=config["rerank"]["model"] if config["rerank"] else None,
        rerank_top_n=config["rerank"]["top_n"] if config["rerank"] else 0,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    latencies["retrieve_ms"] += elapsed - rerank_ms
    latencies["rerank_ms"] += rerank_ms

    retrieved = []
    for i in final:
        scores = details[i]
        retrieved.append({
            "id": i,
            "source": all_chunks[i].source,
            "chunk_type": all_chunks[i].chunk_type,
            "text": texts[i],
            "parent_text": all_chunks[i].parent_text or None,
            "words": word_count(texts[i]),
            "vector_score": round(scores["vector"], 4),
            "bm25_score": round(scores["bm25"], 4) if scores["bm25"] is not None else None,
            "rank_score": round(scores["rank"], 4),
            "rerank_score": round(scores["rerank"], 4) if scores["rerank"] is not None else None,
        })

    # Parent-child chunking: send each parent once, even if two of its
    # children were retrieved.
    context_chunks: List[Chunk] = []
    seen_context = set()
    for i in final:
        chunk = all_chunks[i]
        marker = chunk.parent_text or chunk.text
        if marker in seen_context:
            continue
        seen_context.add(marker)
        context_chunks.append(chunk)

    prompt = build_prompt(question, context_chunks)
    answer = ""
    if generate:
        t0 = time.perf_counter()
        try:
            answer = generate_answer(prompt)
        except Exception as e:
            answer = f"Answer generation failed: {e}"
        latencies["generate_ms"] = (time.perf_counter() - t0) * 1000

    latencies["total_ms"] = (time.perf_counter() - started) * 1000
    latencies = {k: int(v) for k, v in latencies.items()}

    return {
        "config": config,
        "rewrites": rewrites,
        "chunks_total": len(all_chunks),
        "chunks": [
            {
                "id": i,
                "source": c.source,
                "chunk_type": c.chunk_type,
                "words": word_count(c.text),
                "start": c.start_char,
                "end": c.end_char,
                "text": c.text,
                "context": c.embed_text.split("\n\n", 1)[0] if c.embed_text else None,
            }
            for i, c in enumerate(all_chunks)
        ],
        "retrieved": retrieved,
        "projection": _project(embeddings) if len(all_chunks) <= 800 else [],
        "prompt": prompt,
        "answer": answer,
        "metrics": {
            "context_words": sum(word_count(c.parent_text or c.text) for c in context_chunks),
            "retrieval_keyword_hit_rate": _hit_rate(keywords, [r["text"] for r in retrieved]),
            "retrieval_precision_at_k": _precision_at_k(snippets, retrieved),
            "retrieval_recall_at_k": _recall_at_k(snippets, retrieved),
            "answer_keyword_hit_rate": _hit_rate(keywords, [answer]),
            "answer_grounding": _grounding(
                answer, "\n".join(c.parent_text or c.text for c in context_chunks)
            ),
        },
        "latency_ms": latencies,
    }


def run_experiment(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate, execute every config, save the record, return it."""
    question = str(payload.get("question") or "").strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    doc_names = payload.get("docs") or []
    if not isinstance(doc_names, list) or not doc_names:
        raise ValueError("Select at least one document.")
    doc_names = list(dict.fromkeys(doc_names))
    if len(doc_names) > 10:
        raise ValueError("Select at most 10 documents per experiment.")

    known = {d["name"]: d for d in list_documents()}
    missing = [n for n in doc_names if n not in known]
    if missing:
        raise ValueError(f"Document(s) not found: {', '.join(missing)}")

    raw_configs = payload.get("configs") or []
    if not isinstance(raw_configs, list) or not raw_configs:
        raise ValueError("Add at least one config.")
    if len(raw_configs) > 6:
        raise ValueError("Run at most 6 configs at once.")

    configs = [normalize_config(c) for c in raw_configs]
    keywords = [str(k).strip() for k in (payload.get("expected_keywords") or []) if str(k).strip()][:20]
    snippets = [str(s).strip() for s in (payload.get("expected_snippets") or []) if str(s).strip()][:20]
    generate = bool(payload.get("generate", True))

    docs = [{"name": n, "sha256": known[n]["sha256"]} for n in doc_names]
    results = []
    for config in configs:
        try:
            results.append(run_config(config, question, docs, generate, keywords, snippets))
        except Exception as e:
            logger.warning(f"Config failed: {e}", exc_info=True)
            results.append({"config": config, "error": str(e)})

    record = {
        "id": uuid.uuid4().hex[:12],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "docs": doc_names,
        "expected_keywords": keywords,
        "expected_snippets": snippets,
        "results": results,
    }

    _append_record(record)
    return record


def _write_records(records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(EXPERIMENTS_PATH) or ".", exist_ok=True)
    tmp = EXPERIMENTS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    os.replace(tmp, EXPERIMENTS_PATH)


def _append_record(record: Dict[str, Any]) -> None:
    """Append a run; drop the oldest records when the retention cap is hit."""
    with _WRITE_LOCK:
        os.makedirs(os.path.dirname(EXPERIMENTS_PATH) or ".", exist_ok=True)
        with open(EXPERIMENTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if MAX_RUNS > 0:
            records = _read_records()
            if len(records) > MAX_RUNS:
                _write_records(records[-MAX_RUNS:])


def delete_run(run_id: str) -> bool:
    """Remove one saved run. Returns False when the id isn't found."""
    with _WRITE_LOCK:
        records = _read_records()
        kept = [r for r in records if r.get("id") != run_id]
        if len(kept) == len(records):
            return False
        _write_records(kept)
        return True


def _read_records() -> List[Dict[str, Any]]:
    records = []
    try:
        with open(EXPERIMENTS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except ValueError:
                    continue  # skip a partially written line
    except FileNotFoundError:
        pass
    return records


def list_runs(limit: int = 20) -> List[Dict[str, Any]]:
    records = _read_records()
    return [
        {
            "id": r["id"],
            "created_at": r["created_at"],
            "question": r["question"],
            "docs": r.get("docs", []),
            "configs": [res.get("config") for res in r.get("results", [])],
        }
        for r in reversed(records[-limit:])
    ]


def get_run(run_id: str) -> Dict[str, Any] | None:
    for record in reversed(_read_records()):
        if record.get("id") == run_id:
            return record
    return None


def list_eval_questions() -> List[Dict[str, Any]]:
    """Labelled questions from EVAL_PATH: question, docs, snippets, keywords.

    Returns [] when the file is missing or malformed, so the lab still works
    without an eval set.
    """
    try:
        with open(EVAL_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [
        {
            "question": str(item.get("question", "")).strip(),
            "docs": [str(d) for d in item.get("docs", [])],
            "snippets": [str(s) for s in item.get("snippets", [])],
            "keywords": [str(k) for k in item.get("keywords", [])],
        }
        for item in data
        if isinstance(item, dict) and str(item.get("question", "")).strip()
    ]
