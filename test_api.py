"""Quick API tests using FastAPI TestClient (no live server needed)."""
import io
import os
import sys

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from dotenv import load_dotenv
load_dotenv()

from fastapi.testclient import TestClient
import api as api_module

app = api_module.app
client = TestClient(app, raise_server_exceptions=False)

results = []

def check(label, condition, detail=""):
    if condition:
        print(f"[PASS] {label}")
        results.append(("PASS", label))
    else:
        print(f"[FAIL] {label} — {detail}")
        results.append(("FAIL", label))


# ---- 1. Health endpoint --------------------------------------------------
r = client.get("/health")
check("GET /health => 200", r.status_code == 200, f"status={r.status_code}")
if r.status_code == 200:
    data = r.json()
    check(
        'GET /health response has status=healthy',
        data.get("status") == "healthy",
        str(data),
    )
    check(
        "GET /health response has indexed_chunks",
        "indexed_chunks" in data,
        str(data),
    )
print(f"  /health response: {r.json() if r.status_code == 200 else r.text}")

# ---- 2. Sources (empty or existing) --------------------------------------
r = client.get("/sources")
check("GET /sources => 200", r.status_code == 200, r.text)
print(f"  /sources: {r.json()}")

# ---- 3. Upload an empty file - should 400 --------------------------------
r = client.post(
    "/ingest/file",
    files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
)
check("POST /ingest/file (empty file) => 400", r.status_code == 400, r.text)
print(f"  empty file detail: {r.json().get('detail', r.text)}")

# ---- 4. Upload unsupported extension - should 400 ------------------------
r = client.post(
    "/ingest/file",
    files={"file": ("test.exe", io.BytesIO(b"data"), "application/octet-stream")},
)
check("POST /ingest/file (.exe) => 400", r.status_code == 400, r.text)
print(f"  .exe detail: {r.json().get('detail', r.text)}")

# ---- 5. Path traversal safety -------------------------------------------
r = client.post(
    "/ingest/file",
    files={"file": ("../../etc/passwd", io.BytesIO(b"data"), "text/plain")},
)
# After sanitization '../../etc/passwd' has no recognised extension -> 400
check(
    "POST /ingest/file (path traversal) does not crash",
    r.status_code in (200, 400, 422),
    f"status={r.status_code}",
)
print(f"  path traversal response: {r.status_code} {r.text[:80]}")

# ---- 6. Valid TXT upload -------------------------------------------------
content = b"This document explains chunking strategies in retrieval-augmented generation systems. Chunking divides long documents into smaller overlapping passages."
r = client.post(
    "/ingest/file",
    files={"file": ("test_doc.txt", io.BytesIO(content), "text/plain")},
)
check("POST /ingest/file (txt) => 200", r.status_code == 200, r.text)
if r.status_code == 200:
    data = r.json()
    check("TXT ingest returns chunks_added > 0", data["chunks_added"] > 0, str(data))
    check("TXT ingest returns filename", data["filename"] == "test_doc.txt", str(data))
    print(f"  txt ingest: {data}")

# ---- 7. Sources list updated after ingest --------------------------------
r = client.get("/sources")
check(
    "GET /sources includes test_doc.txt after ingest",
    "test_doc.txt" in r.json().get("sources", []),
    str(r.json()),
)

# ---- 8. Valid MD upload --------------------------------------------------
r = client.post(
    "/ingest/file",
    files={
        "file": (
            "test.md",
            io.BytesIO(b"# Vector Databases\nVector databases store embeddings and enable fast similarity search."),
            "text/markdown",
        )
    },
)
check("POST /ingest/file (md) => 200", r.status_code == 200, r.text)
print(f"  md ingest: {r.json() if r.status_code == 200 else r.text}")

# ---- 9. Query (Gemini not available in sandbox so we expect graceful error)
r = client.post("/query", json={"question": "What is chunking?"})
check("POST /query => 200 (even if generation fails)", r.status_code == 200, r.text)
answer = r.json().get("answer", "")
check(
    "POST /query answer is a non-empty string",
    isinstance(answer, str) and len(answer) > 0,
    repr(answer),
)
print(f"  query answer (first 100): {answer[:100]!r}")

# ---- 10. Delete source ---------------------------------------------------
r = client.delete("/sources/test_doc.txt")
check("DELETE /sources/test_doc.txt => 200", r.status_code == 200, r.text)
print(f"  delete: {r.json() if r.status_code == 200 else r.text}")

# ---- 11. Delete non-existent source -> 404 --------------------------------
r = client.delete("/sources/does_not_exist.txt")
check("DELETE /sources/nonexistent => 404", r.status_code == 404, r.text)

# ---- 12. PDF upload (grade.pdf is image-based, should still not crash) ----
with open("sample_docs/grade.pdf", "rb") as f:
    pdf_bytes = f.read()
r = client.post(
    "/ingest/file",
    files={"file": ("grade.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
)
# grade.pdf is image-only so extract_pdf_text returns empty => 400 (correct behaviour)
check(
    "POST /ingest/file (image PDF) => 400 (no text) or 200",
    r.status_code in (200, 400),
    f"status={r.status_code} detail={r.text[:100]}",
)
print(f"  grade.pdf response: {r.status_code} {r.text[:120]}")

# ---- 13. PDF upload (DA_GATE2027_Syllabus.pdf has text) ------------------
with open("sample_docs/DA_GATE2027_Syllabus.pdf", "rb") as f:
    pdf_bytes = f.read()
r = client.post(
    "/ingest/file",
    files={"file": ("syllabus.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
)
check(
    "POST /ingest/file (text PDF with tables) => 200",
    r.status_code == 200,
    f"status={r.status_code} {r.text[:120]}",
)
if r.status_code == 200:
    data = r.json()
    print(f"  syllabus.pdf: {data}")

# ---- Summary -------------------------------------------------------------
print()
passed = sum(1 for s, _ in results if s == "PASS")
failed = sum(1 for s, _ in results if s == "FAIL")
print(f"Results: {passed} passed, {failed} failed out of {len(results)} tests")
if failed:
    print("FAILED tests:")
    for s, label in results:
        if s == "FAIL":
            print(f"  - {label}")
    sys.exit(1)
else:
    print("All tests passed.")

