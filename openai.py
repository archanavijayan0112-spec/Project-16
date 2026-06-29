from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


class QdrantStore:
    """
    Vector store backed by Qdrant with a persisted BM25 index for hybrid retrieval.
    BM25 index is serialized to disk so it survives server restarts.
    """

    BM25_PATH = Path(".docmind_bm25.pkl")

    def __init__(self, url: str = "http://localhost:6333", collection: str = "docmind"):
        self.url = url
        self.collection = collection
        self._client = None
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[dict] = []
        self._load_bm25()

    @property
    def client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.models import Distance, VectorParams
            except ImportError:
                raise ImportError("Install qdrant-client: pip install qdrant-client")

            self._client = QdrantClient(url=self.url)
            existing = [c.name for c in self._client.get_collections().collections]
            if self.collection not in existing:
                self._client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
                )
        return self._client

    def upsert(self, chunk: dict, embedding: list[float]) -> None:
        from qdrant_client.models import PointStruct

        point = PointStruct(
            id=chunk["id"],
            vector=embedding,
            payload={"text": chunk["text"], **chunk["metadata"]},
        )
        self.client.upsert(collection_name=self.collection, points=[point])
        self._bm25_docs.append(chunk)
        self._bm25 = None
        self._save_bm25()

    def search(self, embedding: list[float], top_k: int = 20) -> list[dict]:
        results = self.client.search(
            collection_name=self.collection,
            query_vector=embedding,
            limit=top_k,
        )
        return [
            {
                "id": str(r.id),
                "text": r.payload.get("text", ""),
                "metadata": {k: v for k, v in r.payload.items() if k != "text"},
                "score": r.score,
            }
            for r in results
        ]

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
        """Check if a document hash has already been indexed."""
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue
            results, _ = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="hash", match=MatchValue(value=doc_hash))]
                ),
                limit=1,
            )
            return len(results) > 0
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
