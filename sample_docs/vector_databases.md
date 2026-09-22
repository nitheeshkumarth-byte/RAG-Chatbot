# Vector Databases

A vector database stores data as high-dimensional vectors, which are mathematical
representations of features or attributes. Each vector has a certain number of
dimensions, ranging from tens to thousands, depending on the complexity of the
data being represented.

## Why they matter for RAG

Traditional databases are built for exact matches — you search for a row where
`id = 5` or `name = "Alice"`. Vector databases are built for *similarity* search:
given a query vector, find the stored vectors that are closest to it in the
embedding space. This is exactly what retrieval-augmented generation needs,
since we want to find text passages that are semantically similar to a user's
question, not just passages that share the same keywords.

## Popular options

- **FAISS** (Facebook AI Similarity Search) — a library, not a server. Great for
  local prototyping since it runs in-process and needs no setup.
- **Chroma** — an open-source embedding database designed to be simple to run
  locally, with a lightweight persistent storage layer.
- **Qdrant** — a vector search engine written in Rust, often used in production
  because it supports filtering, payloads, and horizontal scaling.
- **pgvector** — a Postgres extension that adds vector similarity search to a
  database you may already be running, useful if you don't want a separate
  system to maintain.

## Indexing strategies

Most vector databases don't do a brute-force comparison against every stored
vector (that would be too slow at scale). Instead they use approximate nearest
neighbor (ANN) algorithms like HNSW (Hierarchical Navigable Small World graphs)
or IVF (Inverted File Index) to trade a small amount of accuracy for a large
speedup in search time.
