from __future__ import annotations

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi


class ChromaStore:
    """
    Vector store backed by ChromaDB (local, no server required).
    Good for development and small corpora.
    """

    BM25_PATH = Path(".docmind_bm25_chroma.pkl")

    def __init__(self, path: str = ".chroma", collection: str = "docmind"):
        self.path = path
        self.collection_name = collection
        self._client = None
        self._collection = None
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[dict] = []
        self._load_bm25()

    @property
    def collection(self):
        if self._collection is None:
            try:
                import chromadb
            except ImportError:
                raise ImportError("Install chromadb: pip install chromadb")
            self._client = chromadb.PersistentClient(path=self.path)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def upsert(self, chunk: dict, embedding: list[float]) -> None:
        self.collection.upsert(
            ids=[chunk["id"]],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[chunk["metadata"]],
        )
        self._bm25_docs.append(chunk)
        self._bm25 = None
        self._save_bm25()

    def search(self, embedding: list[float], top_k: int = 20) -> list[dict]:
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.collection.count() or 1),
        )
        out = []
        for i, doc_id in enumerate(results["ids"][0]):
            out.append({
                "id": doc_id,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] or {},
                "score": 1 - results["distances"][0][i],
            })
        return out

    def bm25_search(self, query: str, top_k: int = 20) -> list[dict]:
        if not self._bm25_docs:
            return []
        if self._bm25 is None:
            tokenised = [doc["text"].lower().split() for doc in self._bm25_docs]
            self._bm25 = BM25Okapi(tokenised)
        scores = self._bm25.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {**self._bm25_docs[i], "score": float(scores[i])}
            for i in top_indices
            if scores[i] > 0
        ]

    def doc_exists(self, doc_hash: str) -> bool:
        try:
            results = self.collection.get(where={"hash": doc_hash}, limit=1)
            return len(results["ids"]) > 0
        except Exception:
            return False

    def _save_bm25(self) -> None:
        with open(self.BM25_PATH, "wb") as f:
            pickle.dump(self._bm25_docs, f)

    def _load_bm25(self) -> None:
        if self.BM25_PATH.exists():
            try:
                with open(self.BM25_PATH, "rb") as f:
                    self._bm25_docs = pickle.load(f)
            except Exception:
                self._bm25_docs = []
