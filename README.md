# Project-16
DocMind — RAG Document Q&amp;A
[README.md](https://github.com/user-attachments/files/29451377/README.md)
# DocMind — RAG Document Q&A

Ask questions over any document corpus. Retrieval-augmented generation pipeline with semantic chunking, hybrid search, and streaming answers grounded in your sources.

![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Version](https://img.shields.io/badge/version-0.4.0-gray)

---

## How it works

DocMind separates **indexing** (ingest once) from **retrieval** (query fast):

```
Documents → Chunker → Embedder → Vector store
                                      ↓
Query → Hybrid search (vector + BM25) → Re-ranker → Claude → Cited answer
```

## Quickstart

```bash
pip install "docmind[qdrant,rerank]"
```

```python
from docmind import Pipeline

pipeline = Pipeline.from_config("config.yaml")

# Index your documents
pipeline.ingest("./docs/")

# Ask questions
answer = pipeline.query("What is the refund policy for annual plans?")
print(answer.text)     # grounded, cited answer
print(answer.sources)  # [{doc, page, chunk_id, score}, ...]
```

Or via the CLI:

```bash
docmind ingest ./docs/ --config config.yaml
docmind query "What is the refund policy?" --stream
```

## Configuration

Copy `config.yaml` and set your API keys as environment variables:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
```

See `config.yaml` for all options (LLM model, chunk size, top-k, re-rank, vector store backend).

## Stack

| Layer | Default | Alternatives |
|---|---|---|
| LLM | Claude Sonnet 4.6 | GPT-4o, Gemini, local via Ollama |
| Embeddings | text-embedding-3-large | BGE-M3, Cohere |
| Vector DB | Qdrant | Chroma, pgvector, Pinecone |
| Re-ranker | cross-encoder/ms-marco | Cohere Rerank, none |
| API server | FastAPI + SSE | — |

## API server

```bash
docmind serve --port 8000

# POST /query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the payment terms?", "stream": false}'

# POST /ingest
curl -X POST http://localhost:8000/ingest \
  -d '{"path": "./docs/"}'
```

## Development

```bash
git clone https://github.com/you/docmind
cd docmind
pip install -e ".[dev,qdrant,rerank]"
pytest
```

## License

MIT
