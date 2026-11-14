"""Assert-based checks for the experiment layer (no network, no LLM calls).

Run:  python test_experiments.py
"""

import os
import shutil
import sys
import tempfile

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["EXPERIMENTS_PATH"] = os.path.join(
    os.environ.get("TEMP", "."), "rag_test_experiments.jsonl"
)

from fastapi.testclient import TestClient

from rag import chunking, retrieval
import api as api_module

client = TestClient(api_module.app, raise_server_exceptions=False)
failures = []


def check(label, condition, detail=""):
    if condition:
        print(f"[PASS] {label}")
    else:
        print(f"[FAIL] {label} — {detail}")
        failures.append(label)


# ---- Chunking strategies -------------------------------------------------
text = ("# Heading one\n\n" + "alpha beta gamma delta. " * 120 + "\n\n"
        + "# Heading two\n\n" + "epsilon zeta eta theta. " * 120)

for strategy in chunking.STRATEGIES:
    if strategy == "semantic":
        # Needs the embedding model; skip (not fail) if it can't be loaded.
        try:
            pieces = chunking.split_text(text, strategy, size=80, overlap=20,
                                         model="all-MiniLM-L6-v2")
        except Exception as e:
            print(f"[SKIP] semantic chunking unavailable: {e}")
            continue
    else:
        pieces = chunking.split_text(text, strategy, size=80, overlap=20)
    check(f"{strategy}: produces chunks", len(pieces) > 1, len(pieces))
    check(f"{strategy}: every chunk <= size words",
          all(chunking.word_count(p.text) <= 80 for p in pieces),
          max(chunking.word_count(p.text) for p in pieces))
    check(f"{strategy}: offsets move forward",
          all(a.start < b.start and a.end <= b.end for a, b in zip(pieces, pieces[1:])))
    check(f"{strategy}: text matches offsets",
          all(p.text == text[p.start:p.end] for p in pieces))

try:
    chunking.split_text(text, "semantic")
    check("semantic: needs an embedding model", False)
except ValueError:
    check("semantic: needs an embedding model", True)

fixed = chunking.split_text(text, "fixed", size=50, overlap=10)
overlap_ok = (len(fixed) > 2 and
              fixed[0].text.split()[-10:] == fixed[1].text.split()[:10])
check("fixed: overlap is carried", overlap_ok)

check("fixed: overlap near size still progresses",
      len(chunking.split_text(text, "fixed", size=50, overlap=49)) > 1)
check("legacy chunk_text: overlap >= size does not hang",
      len(api_module.chunk_single_document("x", text, 50, 50)) > 1)

try:
    chunking.split_text(text, "nope")
    check("unknown strategy rejected", False)
except ValueError:
    check("unknown strategy rejected", True)

# ---- Retrieval -----------------------------------------------------------
bm25 = retrieval.BM25(["apple banana", "banana banana", "car"])
scores = bm25.scores("banana")
check("bm25: repeated term ranks first", scores[1] > scores[0] > scores[2], scores.tolist())
check("minmax: scales to 0..1", abs(retrieval.minmax(scores).max() - 1) < 1e-9)

import numpy as np
mean = retrieval.mean_query_vector([np.array([1.0, 0.0]), np.array([0.0, 1.0])])
check("mean query vector: unit length and averaged",
      abs(float(np.linalg.norm(mean)) - 1.0) < 1e-9 and abs(mean[0] - mean[1]) < 1e-9,
      mean.tolist())
emb = np.eye(3)
picked = retrieval.mmr(np.array([1.0, 0.9, 0.0]), emb, [0, 1, 2], 2, 0.5)
check("mmr: returns unique top_k", len(picked) == len(set(picked)) == 2, picked)

rank_texts = ["apple banana", "banana banana", "car"]
rank_vec = np.array([1.0, 0.0, 0.0])
order, details, rerank_ms = retrieval.rank(
    "banana", rank_texts, emb, rank_vec, strategy="bm25", top_k=2
)
check("rank: bm25 puts the repeated term first", order == [1, 0], order)
check("rank: details carry every score",
      set(details[order[0]]) == {"vector", "bm25", "rank", "rerank"} and rerank_ms >= 0)
order, details, _ = retrieval.rank(
    "banana", rank_texts, emb, rank_vec, strategy="vector", top_k=2,
    allowed=np.array([False, True, True]),
)
check("rank: filter hides disallowed chunks", 0 not in order and len(order) == 2, order)

# The cross-encoder download is optional; skip (not fail) when it isn't cached.
try:
    order, values = retrieval.rerank(
        "what is chunking",
        ["chunking splits a document into passages", "banana bread recipe"],
        [0, 1],
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        2,
    )
    check("rerank: relevant chunk first", order[0] == 0, order)
    check("rerank: returns ordered scores", len(values) == 2 and values[0] > values[1], values)
except Exception as e:
    print(f"[SKIP] reranker unavailable: {e}")

# ---- Metrics -------------------------------------------------------------
retrieved_fake = [{"text": "fixed-size chunking uses overlap"}, {"text": "bananas"}]
check("precision@k: half the retrieved chunks are relevant",
      api_module.pipeline._precision_at_k(["overlap"], retrieved_fake) == 0.5)
check("recall@k: one of two snippets found",
      api_module.pipeline._recall_at_k(["overlap", "missing"], retrieved_fake) == 0.5)
check("precision/recall: no snippets => None",
      api_module.pipeline._precision_at_k([], retrieved_fake) is None)
check("grounding: copied answer scores 1.0",
      api_module.pipeline._grounding("chunking uses overlap", "chunking uses overlap") == 1.0)
check("grounding: invented answer scores 0.0",
      api_module.pipeline._grounding("bananas", "chunking uses overlap") == 0.0)
check("grounding: empty answer => None",
      api_module.pipeline._grounding("", "chunking uses overlap") is None)

# ---- API -----------------------------------------------------------------
r = client.get("/experiments/options")
check("GET /experiments/options => 200", r.status_code == 200, r.text)
options = r.json() if r.status_code == 200 else {}
doc_names = [d["name"] for d in options.get("documents", [])]
check("options list documents", len(doc_names) > 0, doc_names)
check("options list strategies", options.get("chunking_strategies") == list(chunking.STRATEGIES))

r = client.get("/lab")
check("GET /lab => 200 HTML", r.status_code == 200 and "<html" in r.text.lower(), r.status_code)

if api_module.store.chunks:
    live_hits = api_module._live_search("what is this about")
    check("live search: returns configured chunk count",
          1 <= len(live_hits) <= api_module.RETRIEVAL_TOP_K, len(live_hits))

    original = (api_module.RETRIEVAL_MULTI_QUERY, api_module.pipeline.rewrite_queries)
    calls = []
    api_module.RETRIEVAL_MULTI_QUERY = 2
    api_module.pipeline.rewrite_queries = (
        lambda q, n: calls.append((q, n)) or ["rewrite one", "rewrite two"]
    )
    try:
        multi_hits = api_module._live_search("what is this about")
    finally:
        api_module.RETRIEVAL_MULTI_QUERY, api_module.pipeline.rewrite_queries = original
    check("live search: multi-query uses rewrites and still returns chunks",
          calls == [("what is this about", 2)] and 1 <= len(multi_hits) <= api_module.RETRIEVAL_TOP_K,
          {"calls": calls, "hits": len(multi_hits)})

r = client.get("/experiments/eval")
check("GET /experiments/eval => 200", r.status_code == 200, r.text[:200])
eval_questions = r.json().get("questions", []) if r.status_code == 200 else []
check("eval set loads questions", len(eval_questions) > 0, eval_questions)
check("eval questions carry snippets",
      all(q.get("snippets") for q in eval_questions), eval_questions)

if doc_names:
    doc = next((n for n in doc_names if n.lower().endswith((".txt", ".md"))), doc_names[0])
    doc_text = open(os.path.join("sample_docs", doc), encoding="utf-8", errors="ignore").read()
    snippet = " ".join(doc_text.split()[:8])
    payload = {
        "question": "What is chunking?",
        "docs": [doc],
        "generate": False,
        "expected_keywords": ["chunk"],
        "expected_snippets": [snippet],
        "configs": [
            {"chunking": {"strategy": "fixed", "size": 80, "overlap": 20},
             "embedding": {"model": "all-MiniLM-L6-v2"},
             "retrieval": {"strategy": "vector", "top_k": 3}},
            {"chunking": {"strategy": "recursive", "size": 80, "overlap": 20},
             "embedding": {"model": "all-MiniLM-L6-v2"},
             "retrieval": {"strategy": "bm25", "top_k": 3}},
            {"chunking": {"strategy": "recursive", "size": 60, "overlap": 10,
                          "parent_size": 200},
             "embedding": {"model": "all-MiniLM-L6-v2"},
             "retrieval": {"strategy": "vector", "top_k": 3, "chunk_type": "text"}},
        ],
    }
    r = client.post("/experiments/run", json=payload)
    check("POST /experiments/run => 200", r.status_code == 200, r.text[:300])
    if r.status_code == 200:
        record = r.json()
        results = record["results"]
        check("run returns one trace per config", len(results) == 3)
        check("run has no config errors", all("error" not in x for x in results),
              [x.get("error") for x in results])
        first = results[0]
        check("trace has chunks", first.get("chunks_total", 0) > 0)
        check("trace has retrieved chunks", len(first.get("retrieved", [])) == 3)
        check("trace has scores", all("vector_score" in x for x in first["retrieved"]))
        check("trace has latency", "total_ms" in first.get("latency_ms", {}))
        check("trace has projection",
              len(first.get("projection", [])) == first.get("chunks_total"))
        check("keyword metric computed",
              first["metrics"]["retrieval_keyword_hit_rate"] is not None)
        check("snippet metrics computed",
              first["metrics"]["retrieval_precision_at_k"] is not None
              and first["metrics"]["retrieval_recall_at_k"] is not None,
              first["metrics"])
        check("generation off => empty answer", first["answer"] == "")

        parented = results[2]
        check("parent-child: children retrieved",
              len(parented.get("retrieved", [])) > 0, parented.get("error"))
        check("parent-child: parents attached",
              any(r.get("parent_text") for r in parented.get("retrieved", [])))
        check("parent-child: prompt carries the parent",
              any(r["parent_text"] in parented.get("prompt", "")
                  for r in parented.get("retrieved", []) if r.get("parent_text")))
        check("metadata filter: only text chunks",
              all(r["chunk_type"] == "text" for r in parented.get("retrieved", [])))

        r = client.get("/experiments")
        check("GET /experiments => 200", r.status_code == 200, r.text[:200])
        ids = [x["id"] for x in r.json().get("experiments", [])]
        check("run appears in history", record["id"] in ids, ids)

        r = client.get(f"/experiments/{record['id']}")
        check("GET /experiments/{id} => 200", r.status_code == 200, r.text[:200])
        check("stored run round-trips", r.json().get("id") == record["id"])
        check("GET /experiments/missing => 404",
              client.get("/experiments/does-not-exist").status_code == 404)

        check("DELETE /experiments/{id} => 200",
              client.delete(f"/experiments/{record['id']}").status_code == 200)
        check("deleted run is gone",
              client.get(f"/experiments/{record['id']}").status_code == 404)
        check("DELETE /experiments/missing => 404",
              client.delete("/experiments/does-not-exist").status_code == 404)

    bad = dict(payload, configs=[{"chunking": {"strategy": "nope"}}])
    r = client.post("/experiments/run", json=bad)
    check("invalid config => 400", r.status_code == 400, r.text[:200])
    r = client.post("/experiments/run", json=dict(payload, docs=[]))
    check("empty docs => 400", r.status_code == 400, r.text[:200])
    r = client.post("/experiments/run", json=dict(payload, docs=["nope.txt"]))
    check("missing doc => 400", r.status_code == 400, r.text[:200])

    # A .txt document has no table chunks, so the filter fails the config only.
    tables_only = dict(payload, configs=[{
        "chunking": {"strategy": "recursive", "size": 80, "overlap": 20},
        "embedding": {"model": "all-MiniLM-L6-v2"},
        "retrieval": {"strategy": "vector", "top_k": 3, "chunk_type": "table"},
    }])
    r = client.post("/experiments/run", json=tables_only)
    error = r.json()["results"][0].get("error", "") if r.status_code == 200 else ""
    check("metadata filter: empty type fails with a clear message",
          "table" in error, error[:200])

# ---- Multi-query + contextual (LLM calls stubbed) ------------------------
pipeline = api_module.pipeline
tmp = tempfile.mkdtemp(prefix="rag_test_llm_")
tmp_docs = os.path.join(tmp, "docs")
os.makedirs(tmp_docs)
with open(os.path.join(tmp_docs, "note.txt"), "w", encoding="utf-8") as f:
    f.write("Alpha beta gamma. " * 80)

original = (pipeline.CACHE_DIR, pipeline.DOCS_FOLDER, pipeline.generate_answer)
calls = []


def fake_generate(prompt):
    calls.append(prompt)
    if "Rewrite this search question" in prompt:
        return "alpha beta\ngamma delta\n"
    return "This chunk is about alpha and beta."


pipeline.CACHE_DIR = os.path.join(tmp, "cache")
pipeline.DOCS_FOLDER = tmp_docs
pipeline.generate_answer = fake_generate
try:
    sha = pipeline._sha256(os.path.join(tmp_docs, "note.txt"))
    rewrites = pipeline.rewrite_queries("What is alpha?", 2)
    check("multi-query: parses one rewrite per line",
          rewrites == ["alpha beta", "gamma delta"], rewrites)
    again = pipeline.rewrite_queries("What is alpha?", 2)
    check("multi-query: rewrites are cached", again == rewrites and len(calls) == 1)

    config = pipeline.normalize_config({
        "chunking": {"strategy": "fixed", "size": 40, "overlap": 10, "contextual": True},
        "embedding": {"model": "all-MiniLM-L6-v2"},
        "retrieval": {"strategy": "vector", "top_k": 2, "multi_query": 2},
    })
    trace = pipeline.run_config(
        config, "What is alpha?", [{"name": "note.txt", "sha256": sha}], generate=False
    )
    check("multi-query: trace records the rewrites",
          trace["rewrites"] == rewrites, trace["rewrites"])
    check("multi-query: rewrite latency recorded",
          "rewrite_ms" in trace["latency_ms"], trace["latency_ms"])
    text_chunks = [c for c in trace["chunks"] if c["chunk_type"] == "text"]
    check("contextual: every text chunk gets a context sentence",
          text_chunks and all(c["context"] for c in text_chunks), trace["chunks"][:2])
    embedding_files = os.listdir(os.path.join(pipeline.CACHE_DIR, "embeddings"))
    check("contextual: embeddings use a separate cache",
          any(name.endswith("-ctx.npy") for name in embedding_files), embedding_files)
finally:
    pipeline.CACHE_DIR, pipeline.DOCS_FOLDER, pipeline.generate_answer = original
    shutil.rmtree(tmp, ignore_errors=True)

try:
    pipeline.normalize_config({"retrieval": {"multi_query": 9}})
    check("multi-query: > 5 rewrites rejected", False)
except ValueError:
    check("multi-query: > 5 rewrites rejected", True)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("All experiment checks passed.")
