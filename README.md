# askpdf

A Retrieval-Augmented Generation (RAG) system built from scratch in Python. Upload a PDF, ask questions, get answers grounded in the document's content.

No frameworks like LangChain or LlamaIndex — just the core building blocks to understand how RAG actually works.

## How It Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   PDF       │ ──▶ │  Chunker    │ ──▶ │  Embedder   │ ──▶ │  Vector DB  │
│             │     │  (split)    │     │  (OpenAI)   │     │ (PostgreSQL)│
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                   │
                                                                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Answer    │ ◀── │  Generator  │ ◀── │  Retriever  │ ◀── │   Query     │
│             │     │  (GPT-4o)   │     │  (search)   │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

**Ingestion Pipeline:**
1. PDF is parsed and text is extracted
2. Text is split into overlapping chunks (preserves context at boundaries)
3. Each chunk is converted to a 1536-dimensional vector using OpenAI embeddings
4. Chunks and vectors are stored in PostgreSQL with pgvector

**Query Pipeline:**
1. User's question is converted to a vector
2. Cosine similarity search finds the most relevant chunks
3. Retrieved chunks are passed as context to GPT-4o-mini
4. LLM generates an answer grounded in the retrieved context

## Features

- **PDF ingestion** with text extraction and chunking
- **Overlapping chunks** to preserve context across boundaries
- **Vector similarity search** using PostgreSQL + pgvector
- **Cosine distance** for semantic matching
- **GPT-4o-mini** for answer generation
- **Clean architecture** — each component is a separate module

## Project Structure

```
askpdf/
├── chunker.py      # PDF parsing and text chunking
├── embedder.py     # OpenAI embeddings API wrapper
├── vectordb.py     # PostgreSQL/pgvector operations
├── retriever.py    # Query → relevant chunks
├── generator.py    # Context + query → answer
├── ingest.py       # PDF ingestion pipeline
├── main.py         # CLI entry point
├── tests/          # Unit tests with mocks
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.9+
- Docker (for PostgreSQL)
- OpenAI API key

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/muditbuilds/askpdf.git
   cd askpdf
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start PostgreSQL with pgvector**
   ```bash
   docker run -d \
     --name askpdf-db \
     -e POSTGRES_USER=postgres \
     -e POSTGRES_PASSWORD=postgres \
     -e POSTGRES_DB=askpdf \
     -p 5432:5432 \
     pgvector/pgvector:pg16
   ```

4. **Create the database schema**
   ```bash
   docker exec -it askpdf-db psql -U postgres -d askpdf -c "
     CREATE EXTENSION IF NOT EXISTS vector;
     CREATE TABLE IF NOT EXISTS documents (
       id SERIAL PRIMARY KEY,
       content TEXT,
       embedding vector(1536),
       source TEXT,
       chunk_index INTEGER
     );
   "
   ```

5. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

   Your `.env` should contain:
   ```
   OPENAI_API_KEY=sk-your-key-here
   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/askpdf
   ```

### Usage

```bash
python main.py
```

Then:
1. Enter the path to a PDF file
2. Wait for ingestion to complete
3. Ask questions about the document
4. Type `\q` to quit

## Running Tests

```bash
python -m pytest tests/ -v
```

## What I Learned

Building this from scratch taught me several things that using a framework would have hidden:

### Chunking Strategy Matters
Naive chunking (split every N words) breaks sentences mid-thought. **Overlapping chunks** solve this — each chunk shares some words with its neighbors, so context isn't lost at boundaries. The overlap parameter is a tradeoff: more overlap = better context preservation but more storage.

### Embeddings Are Just Vectors
An embedding is a list of floats (1536 for OpenAI's `text-embedding-3-small`). Similar meanings → similar vectors. The "magic" is just high-dimensional geometry — cosine distance measures the angle between vectors.

### Vector Search Needs Type Casting
PostgreSQL's pgvector uses a custom `vector` type. When passing Python lists to queries, explicit casting (`::vector`) is sometimes needed, even with adapters like `register_vector()`. Debugging this taught me to understand what's happening at the database protocol level.

### Connection Management Is Easy to Get Wrong
Creating a new database connection per request works but is wasteful. Centralizing connection creation in one place prevents bugs (like forgetting to register the vector type) and makes future improvements (connection pooling) easier.

### RAG Is Retrieval + Generation
The "retrieval" part is just search. The "augmented generation" part is just prompting an LLM with context. There's no magic — it's search + prompting, composed well.

### Mocking External Services
Unit tests for code that calls OpenAI or PostgreSQL need mocks. This forced me to think about interfaces and dependencies — what does my code *actually* need from these services?

## Limitations

- **No chunking by semantic boundaries** — splits by word count, not paragraphs or sections
- **No hybrid search** — uses only vector similarity, no keyword matching
- **No reranking** — returns top-k chunks without scoring refinement
- **Single PDF** — ingesting a new PDF doesn't clear old chunks
- **No streaming** — waits for full LLM response before displaying

These are all solvable — and now I understand *why* frameworks like LangChain include these features.

## License

MIT
