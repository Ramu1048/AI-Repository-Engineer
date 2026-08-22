class RetrievalBaseError(Exception):
    """Base exception for all retrieval and embedding errors."""
    pass

# --- Embedding Errors ---

class EmbeddingError(RetrievalBaseError):
    """Base exception for embedding failures."""
    pass

class EmbeddingProviderConfigError(EmbeddingError):
    """Raised when an embedding provider is misconfigured (e.g. missing environment variables)."""
    pass

class EmbeddingProviderUnreachableError(EmbeddingError):
    """Raised when an embedding provider cannot be reached (e.g. network timeout, service down)."""
    pass

# --- Vector Store Errors ---

class VectorStoreError(RetrievalBaseError):
    """Base exception for vector store failures."""
    pass

class VectorStoreWriteError(VectorStoreError):
    """Raised when writing/indexing chunks to ChromaDB fails."""
    pass

class VectorStoreQueryError(VectorStoreError):
    """Raised when querying ChromaDB fails."""
    pass

# --- Retrieval/Indexing Errors ---

class RetrievalError(RetrievalBaseError):
    """Base exception for query retrieval failures."""
    pass

class RepositoryNotIndexedError(RetrievalError):
    """Raised when querying a repository that has not been indexed yet."""
    pass

class EmptyCollectionError(RetrievalError):
    """Raised when the repository has been indexed but has no stored chunks (empty)."""
    pass
