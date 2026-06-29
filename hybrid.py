from __future__ import annotations

import re
import uuid
from typing import Any


class SemanticChunker:
    """
    Splits text into semantically coherent chunks.

    Uses sentence boundaries as split points, then merges sentences
    until the target chunk_size (in tokens) is reached. Overlap is
    expressed as a fraction of chunk_size.
    """

    def __init__(self, chunk_size: int = 512, overlap: float = 0.1):
        self.chunk_size = chunk_size
        self.overlap_tokens = int(chunk_size * overlap)

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict]:
        metadata = metadata or {}
        sentences = self._split_sentences(text)
        chunks: list[dict] = []
        buf: list[str] = []
        buf_tokens = 0

        for sentence in sentences:
            s_tokens = self._count_tokens(sentence)

            if buf_tokens + s_tokens > self.chunk_size and buf:
                chunk_text = " ".join(buf)
                chunks.append(self._make_chunk(chunk_text, metadata, len(chunks)))

                # keep overlap: retain last N tokens worth of sentences
                overlap_buf: list[str] = []
                overlap_count = 0
                for s in reversed(buf):
                    t = self._count_tokens(s)
                    if overlap_count + t > self.overlap_tokens:
                        break
                    overlap_buf.insert(0, s)
                    overlap_count += t
                buf = overlap_buf
                buf_tokens = overlap_count

            buf.append(sentence)
            buf_tokens += s_tokens

        if buf:
            chunks.append(self._make_chunk(" ".join(buf), metadata, len(chunks)))

        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_sentences(self, text: str) -> list[str]:
        # Simple rule-based sentence splitter; swap for spacy/nltk if needed
        text = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _count_tokens(self, text: str) -> int:
        # Rough approximation: 1 token ≈ 4 characters
        return max(1, len(text) // 4)

    def _make_chunk(self, text: str, metadata: dict, index: int) -> dict:
        return {
            "id": str(uuid.uuid4()),
            "text": text,
            "metadata": {**metadata, "chunk_index": index},
        }
