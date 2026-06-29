[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "docmind"
version = "0.4.0"
description = "RAG pipeline for document Q&A — semantic chunking, hybrid search, streaming answers"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.25.0",
    "openai>=1.20.0",
    "pyyaml>=6.0",
    "fastapi>=0.111.0",
    "uvicorn[standard]>=0.29.0",
    "pydantic>=2.0",
    "pypdf>=4.0",
    "beautifulsoup4>=4.12",
    "python-docx>=1.1",
    "rank-bm25>=0.2",
]

[project.optional-dependencies]
qdrant = ["qdrant-client>=1.9.0"]
chroma = ["chromadb>=0.5.0"]
rerank = ["sentence-transformers>=3.0"]
local-embed = ["sentence-transformers>=3.0"]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "httpx>=0.27",
    "ruff>=0.4",
    "mypy>=1.10",
]

[project.scripts]
docmind = "docmind.cli:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["docmind*"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
