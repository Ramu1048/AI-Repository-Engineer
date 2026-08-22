from .vector_retriever import retrieve, Chunk, RetrievedChunk
from vectorstore.chroma_store import store_chunks, chunk_count
from .exceptions import (
    RetrievalBaseError,
    EmbeddingError,
    EmbeddingProviderConfigError,
    EmbeddingProviderUnreachableError,
    VectorStoreError,
    VectorStoreWriteError,
    VectorStoreQueryError,
    RetrievalError,
    RepositoryNotIndexedError,
    EmptyCollectionError
)

__all__ = [
    "retrieve",
    "Chunk",
    "RetrievedChunk",
    "store_chunks",
    "chunk_count",
    "RetrievalBaseError",
    "EmbeddingError",
    "EmbeddingProviderConfigError",
    "EmbeddingProviderUnreachableError",
    "VectorStoreError",
    "VectorStoreWriteError",
    "VectorStoreQueryError",
    "RetrievalError",
    "RepositoryNotIndexedError",
    "EmptyCollectionError"
]
