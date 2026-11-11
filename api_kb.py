"""
api_kb.py — FastAPI app for the App Runner + Knowledge Bases architecture.

This is a sibling to api.py, not a replacement — api.py's local numpy
pipeline still works exactly as before. This file swaps ingestion/retrieval
for the managed AWS path: uploads go to S3, a Bedrock Knowledge Base
chunks + embeds (Titan) + indexes (OpenSearch Serverless) them, and
queries call the Knowledge Base's Retrieve API instead of searching an
in-memory array. Generation is unchanged — still rag/generate.py, Gemini
by default.

Run locally against real AWS resources for testing:
    uvicorn api_kb:app --reload
Deployed on App Runner via Docker — see DEPLOY_APPRUNNER_KB.md.
"""

import os

import boto3
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag.generate import generate_answer
from rag.ingest import extract_url_text, slugify
from rag.kb_client import retrieve, start_ingestion_job

load_dotenv()  # only matters for local testing; App Runner sets real env vars

AWS_REGION = os.environ.get("AWS_REGION", "ap-south-2")
S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
TOP_K = 3
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

s3 = boto3.client("s3", region_name=AWS_REGION)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


class UrlRequest(BaseModel):
    url: str


def build_prompt(question: str, results) -> str:
    context = "\n\n".join(f"[Source: {r.source}]\n{r.text}" for r in results)
    return f"""Answer the question using ONLY the context below. If the context
doesn't contain the answer, say so explicitly instead of guessing.

Context:
{context}

Question: {question}

Answer:"""


@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


@app.get("/sources")
def list_sources():
    """Lists what's in the S3 bucket. Note this reflects the bucket, not
    necessarily what the Knowledge Base has finished indexing yet — a sync
    can take a minute or two after upload."""
    if not S3_BUCKET:
        raise HTTPException(500, "S3_BUCKET_NAME is not configured.")
    response = s3.list_objects_v2(Bucket=S3_BUCKET)
    keys = [obj["Key"] for obj in response.get("Contents", [])]
    return {"sources": keys}


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Use PDF, TXT, or MD.")

    contents = await file.read()
    s3.put_object(Bucket=S3_BUCKET, Key=file.filename, Body=contents)

    job_id = None
    try:
        job_id = start_ingestion_job()
    except Exception:
        pass  # not fatal — the KB will still pick it up on its next scheduled sync

    return {"filename": file.filename, "uploaded_to_s3": True, "ingestion_job": job_id}


@app.post("/ingest/url")
def ingest_url(req: UrlRequest):
    try:
        title, text = extract_url_text(req.url)
    except ValueError as e:
        raise HTTPException(400, str(e))

    filename = slugify(title) + ".txt"
    body = f"# {title}\nSource: {req.url}\n\n{text}"
    s3.put_object(Bucket=S3_BUCKET, Key=filename, Body=body.encode("utf-8"))

    job_id = None
    try:
        job_id = start_ingestion_job()
    except Exception:
        pass

    return {"filename": filename, "uploaded_to_s3": True, "ingestion_job": job_id}


@app.delete("/sources/{filename}")
def delete_source(filename: str):
    s3.delete_object(Bucket=S3_BUCKET, Key=filename)
    try:
        start_ingestion_job()
    except Exception:
        pass
    return {"removed": filename}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    results = retrieve(req.question, top_k=TOP_K)
    if not results:
        return QueryResponse(
            answer="Nothing relevant found. If you just uploaded a document, "
            "the Knowledge Base sync can take a minute or two — try again shortly."
        )

    prompt = build_prompt(req.question, results)
    try:
        answer = generate_answer(prompt)
    except Exception as e:
        return QueryResponse(answer=f"ERROR generating answer: {e}")

    return QueryResponse(answer=answer)
