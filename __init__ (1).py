from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, call

from docmind.chunker.semantic import SemanticChunker
from docmind.retrieval.hybrid import HybridRetriever
from docmind.pipeline import Pipeline


# ------------------------------------------------------------------
# Chunker
# ------------------------------------------------------------------

class TestSemanticChunker:
    def setup_method(self):
        self.chunker = SemanticChunker(chunk_size=50, overlap=0.1)

    def test_basic_chunking(self):
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = self.chunker.chunk(text)
        assert len(chunks) >= 1
        for c in chunks:
            assert "id" in c and "text" in c and "metadata" in c

    def test_metadata_preserved(self):
        chunks = self.chunker.chunk("Hello world.", metadata={"source": "test.pdf"})
        assert all(c["metadata"]["source"] == "test.pdf" for c in chunks)

    def test_chunk_index_increments(self):
        text = ". ".join([f"Sentence {i}" for i in range(100)])
        chunks = self.chunker.chunk(text)
        assert [c["metadata"]["chunk_index"] for c in chunks] == list(range(len(chunks)))

    def test_empty_text_returns_empty(self):
        assert self.chunker.chunk("") == []

    def test_unique_ids(self):
        chunks = self.chunker.chunk("One. Two. Three.")
        ids = [c["id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_overlap_produces_shared_sentences(self):
        long = ". ".join([f"Sentence number {i} is here" for i in range(40)])
        chunker = SemanticChunker(chunk_size=30, overlap=0.2)
        chunks = chunker.chunk(long)
        if len(chunks) > 1:
            # Last sentence(s) of chunk N should appear at start of chunk N+1
            last_words = chunks[0]["text"].split()[-3:]
            assert any(w in chunks[1]["text"] for w in last_words)


# ------------------------------------------------------------------
# HybridRetriever / RRF
# ------------------------------------------------------------------

class TestHybridRetriever:
    def setup_method(self):
        self.store = MagicMock()
        self.embedder = MagicMock()
        self.embedder.embed_one.return_value = [0.1] * 10
        self.retriever = HybridRetriever(
            store=self.store,
            embedder=self.embedder,
            top_k=5,
            rerank_top_n=3,
        )

    def _doc(self, doc_id: str, score: float = 1.0) -> dict:
        return {"id": doc_id, "text": f"Text {doc_id}", "metadata": {}, "score": score}

    def test_rrf_shared_docs_rank_higher(self):
        list_a = [self._doc("a"), self._doc("b"), self._doc("c")]
        list_b = [self._doc("b"), self._doc("c"), self._doc("d")]
        fused = self.retriever._rrf(list_a, list_b)
        ids = [d["id"] for d in fused]
        assert ids.index("b") < ids.index("a")
        assert ids.index("c") < ids.index("a")

    def test_rrf_all_docs_present(self):
        list_a = [self._doc("x"), self._doc("y")]
        list_b = [self._doc("y"), self._doc("z")]
        fused = self.retriever._rrf(list_a, list_b)
        ids = {d["id"] for d in fused}
        assert ids == {"x", "y", "z"}

    def test_rrf_scores_are_floats(self):
        docs = [self._doc(f"d{i}") for i in range(3)]
        for doc in self.retriever._rrf(docs, []):
            assert isinstance(doc["score"], float)

    def test_retrieve_calls_embedder_once(self):
        self.store.search.return_value = []
        self.store.bm25_search.return_value = []
        self.retriever.retrieve("test query")
        self.embedder.embed_one.assert_called_once_with("test query")

    def test_retrieve_respects_rerank_top_n(self):
        docs = [self._doc(f"d{i}", score=float(i)) for i in range(10)]
        self.store.search.return_value = docs
        self.store.bm25_search.return_value = []
        results = self.retriever.retrieve("test")
        assert len(results) <= self.retriever.rerank_top_n

    def test_retrieve_empty_store(self):
        self.store.search.return_value = []
        self.store.bm25_search.return_value = []
        results = self.retriever.retrieve("anything")
        assert results == []


# ------------------------------------------------------------------
# Pipeline (unit — all I/O mocked)
# ------------------------------------------------------------------

class TestPipeline:
    def _make_pipeline(self):
        config = {
            "llm": {"model": "claude-sonnet-4-6", "max_tokens": 512},
            "chunker": {"chunk_size": 100, "overlap": 0.1},
        }
        p = Pipeline.__new__(Pipeline)
        p.config = config
        p._chunker = SemanticChunker(chunk_size=100, overlap=0.1)
        p._embedder = MagicMock()
        p._embedder.embed.return_value = [[0.1] * 10]
        p._embedder.embed_one.return_value = [0.1] * 10
        p._store = MagicMock()
        p._store.doc_exists.return_value = False
        p._retriever = MagicMock()
        p._retriever.retrieve.return_value = [
            {"id": "c1", "text": "The refund period is 14 days.",
             "metadata": {"source": "policy.pdf", "filename": "policy.pdf", "page": 1},
             "score": 0.95}
        ]
        p._llm = MagicMock()
        p._llm.generate.return_value = {
            "text": "The refund period is 14 days [1].",
            "tokens_used": 42,
        }
        return p

    def test_query_returns_answer(self):
        p = self._make_pipeline()
        answer = p.query("What is the refund period?")
        assert "14 days" in answer.text
        assert len(answer.sources) == 1
        assert answer.sources[0].doc == "policy.pdf"
        assert answer.latency_ms >= 0

    def test_query_empty_raises(self):
        p = self._make_pipeline()
        with pytest.raises(ValueError, match="empty"):
            p.query("   ")

    def test_ingest_text_stores_chunks(self):
        p = self._make_pipeline()
        result = p.ingest_text("This is a test document. It has two sentences.")
        assert result.files_processed == 1
        assert result.chunks_stored >= 1
        p._store.upsert.assert_called()

    def test_ingest_skips_unchanged_file(self, tmp_path):
        """ingest() skips files whose hash is already in the store."""
        p = self._make_pipeline()
        p._store.doc_exists.return_value = True
        doc = tmp_path / "doc.txt"
        doc.write_text("Same content.")
        result = p.ingest(tmp_path)
        assert result.chunks_stored == 0
        assert result.skipped == 1
        p._store.upsert.assert_not_called()

    def test_build_context_format(self):
        p = self._make_pipeline()
        chunks = [
            {"text": "Hello world.", "metadata": {"filename": "doc.pdf"}, "id": "1", "score": 0.9},
        ]
        ctx = p._build_context(chunks)
        assert "[1]" in ctx
        assert "doc.pdf" in ctx
        assert "Hello world." in ctx
