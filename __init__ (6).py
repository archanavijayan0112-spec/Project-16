from docmind.stores.qdrant import QdrantStore
from docmind.stores.chroma import ChromaStore

__all__ = ["QdrantStore", "ChromaStore"]


def get_store(config: dict):
    """Factory — returns the right store based on config."""
    backend = config.get("vector_store", {}).get("backend", "qdrant")
    if backend == "qdrant":
        return QdrantStore(
            url=config["vector_store"].get("url", "http://localhost:6333"),
            collection=config["vector_store"].get("collection", "docmind"),
        )
    elif backend == "chroma":
        return ChromaStore(
            path=config["vector_store"].get("path", ".chroma"),
            collection=config["vector_store"].get("collection", "docmind"),
        )
    else:
        raise ValueError(f"Unknown vector store backend: {backend!r}. Choose 'qdrant' or 'chroma'.")
