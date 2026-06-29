# DocMind — RAG Document Q&A

[![CI](https://github.com/you/docmind/actions/workflows/ci.yml/badge.svg)](https://github.com/you/docmind/actions)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-0.4.0-gray)

Ask questions over any document corpus. Retrieval-augmented generation pipeline with semantic chunking, hybrid search, cross-encoder re-ranking, and streaming answers grounded in your sources.

---

## Architecture

```
Indexing:
  Documents → SemanticChunker → Embedder → VectorStore + BM25 index

Retrieval:
  Query → embed → dense search ─┐
          BM25 keyword search  ─┴─ RRF fusion → CrossEncoder re-rank → Claude → cited answer
```

---

## Quickstart

**Requirements:** Python 3.11+, an Anthropic API key, and either a Qdrant instance or no extra setup (Chroma works locally).

```bash
# Install
pip install "docmind[qdrant,rerank]"

# Set keys
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...        # skip if using local embeddings

# Index your documents
docmind ingest ./docs/

# Ask a question
docmind query "What is the refund policy for annual plans?"

# Stream the answer
docmind query "Summarise the onboarding guide" --stream
```

Or as a Python library:

```python
from docmind import Pipeline

pipeline = Pipeline.from_config("config.yaml")
pipeline.ingest("./docs/")

answer = pipeline.query("What are the payment terms?")
print(answer.text)      # grounded, cited answer
print(answer.sources)   # [{doc, page, chunk_id, score}, ...]
print(answer.latency_ms)

# Streaming
for token in pipeline.stream("Explain the onboarding process"):
    print(token, end="", flush=True)
```

---

## Docker (recommended for production)

```bash
cp .env.example .env          # fill in your API keys
docker compose up -d

# Ingest
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"path": "/app/docs"}'

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?", "stream": false}'
```

---

## Configuration

Copy `config.yaml` and adjust to your needs:

```yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-6
  max_tokens: 1024

embeddings:
  provider: openai          # or: local (no API key needed)
  model: text-embedding-3-large
  dimensions: 3072

vector_store:
  backend: qdrant           # or: chroma (zero config, local files)
  collection: docmind
  url: http://localhost:6333

chunker:
  strategy: semantic
  chunk_size: 512
  overlap: 0.1

retrieval:
  mode: hybrid
  top_k: 20
  rerank_top_n: 5
```

**No API key? Use local embeddings:**

```yaml
embeddings:
  provider: local
  model: BAAI/bge-small-en-v1.5   # downloads ~120 MB on first run
```

**No Docker / Qdrant? Use Chroma:**

```yaml
vector_store:
  backend: chroma
  collection: docmind
  path: .chroma             # persisted to disk automatically
```

---

## Stack

| Layer | Default | Alternatives |
|---|---|---|
| LLM | Claude Sonnet 4.6 | GPT-4o, Gemini, Ollama |
| Embeddings | text-embedding-3-large | BAAI/bge-small (local), Cohere |
| Vector DB | Qdrant | Chroma, pgvector |
| Re-ranker | cross-encoder/ms-marco | Cohere Rerank, disabled |
| API server | FastAPI + SSE | — |

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check + pipeline status |
| `POST` | `/ingest` | Index a file or directory |
| `POST` | `/query` | Ask a question (streaming or batch) |

**POST /query**
```json
{ "question": "What is the SLA?", "stream": false }
```
Response:
```json
{
  "text": "The SLA guarantees 99.9% uptime [1].",
  "sources": [{"doc": "sla.pdf", "page": 2, "chunk_id": "...", "score": 0.94}],
  "latency_ms": 312,
  "tokens_used": 487
}
```

---

## Supported file types

`.pdf` · `.md` · `.txt` · `.html` · `.rst` · `.docx`

---

## Development

```bash
git clone https://github.com/you/docmind
cd docmind
pip install -e ".[dev,qdrant,rerank]"
pytest
ruff check docmind/
mypy docmind/
```

---

## License

MIT
