"""
kb_client.py — thin wrapper around Bedrock's Retrieve API.

This plays the same role rag/store.py's VectorStore.search() plays in the
local pipeline, just delegated entirely to AWS: the Knowledge Base already
did the chunking (when the S3 data source synced), the embedding (via
Titan), and now does the similarity search (against OpenSearch Serverless)
— this function just calls it and gets matching passages back.
"""

import os
from dataclasses import dataclass
from typing import List

import boto3


@dataclass
class KBResult:
    text: str
    source: str
    score: float


def retrieve(question: str, top_k: int = 3) -> List[KBResult]:
    region = os.environ.get("AWS_REGION", "us-east-1")
    kb_id = os.environ["KNOWLEDGE_BASE_ID"]

    client = boto3.client("bedrock-agent-runtime", region_name=region)
    response = client.retrieve(
        knowledgeBaseId=kb_id,
        retrievalQuery={"text": question},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": top_k}},
    )

    results = []
    for item in response.get("retrievalResults", []):
        text = item.get("content", {}).get("text", "")
        score = item.get("score", 0.0)
        source = item.get("location", {}).get("s3Location", {}).get("uri", "unknown")
        results.append(KBResult(text=text, source=source, score=score))
    return results


def start_ingestion_job() -> str:
    """Trigger an on-demand sync so a just-uploaded S3 file gets chunked and
    embedded promptly, instead of waiting for the Knowledge Base's next
    scheduled sync. Called after every upload in api_kb.py."""
    region = os.environ.get("AWS_REGION", "us-east-1")
    kb_id = os.environ["KNOWLEDGE_BASE_ID"]
    data_source_id = os.environ["DATA_SOURCE_ID"]

    client = boto3.client("bedrock-agent", region_name=region)
    response = client.start_ingestion_job(knowledgeBaseId=kb_id, dataSourceId=data_source_id)
    return response["ingestionJob"]["ingestionJobId"]
