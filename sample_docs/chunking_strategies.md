# Chunking Strategies for RAG

Chunking is the process of splitting long documents into smaller pieces before
embedding them. It's one of the most underrated parts of a RAG pipeline —
poor chunking often hurts retrieval quality more than the choice of embedding
model or vector database.

## Fixed-size chunking

The simplest approach: split text every N tokens or characters, often with some
overlap between consecutive chunks (for example, 500 tokens per chunk with 50
tokens of overlap). Overlap helps prevent an important sentence from being
awkwardly cut in half and losing context in both resulting chunks.

## Structure-aware chunking

Instead of blindly cutting at a fixed length, split along natural document
boundaries — paragraphs, markdown headers, or sections. This tends to produce
chunks that are more semantically coherent, since each chunk stays within a
single topic rather than spanning two unrelated ideas.

## Semantic chunking

A more advanced approach: embed individual sentences, then group consecutive
sentences together only while their embeddings stay similar to each other,
starting a new chunk when the topic shifts significantly. This adapts chunk
boundaries to the actual content rather than an arbitrary length.

## Trade-offs

Smaller chunks give more precise retrieval (less irrelevant text mixed in) but
can lose broader context. Larger chunks preserve context but may dilute the
relevant part of a chunk with unrelated surrounding text, which can confuse
both the retriever and the generator. Most practical systems land somewhere
between 200 and 800 tokens per chunk, then tune from there based on evaluation
results.
