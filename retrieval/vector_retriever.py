from dataclasses import dataclass
import logging
from embeddings import get_embedding_provider
from vectorstore import chroma_store
from retrieval.exceptions import (
    RepositoryNotIndexedError,
    EmptyCollectionError
)

logger = logging.getLogger("code_intelligence.retrieval")

@dataclass
class Chunk:
    repository_id: str
    file_path: str
    language: str
    symbol_type: str      # "function" | "class" | "method" | "doc_section" | "config_section"
    symbol_name: str or None
    class_name: str or None
    start_line: int
    end_line: int
    content: str           # the actual text to embed


@dataclass
class RetrievedChunk:
    chunk: Chunk        # full metadata, including content
    score: float          # similarity score from Chroma
    retrieval_method: str   # always "vector" in this build


def retrieve(repository_id: str, question: str, top_k: int = 8) -> list[RetrievedChunk]:
    """
    Retrieves the top-K chunks matching a question within a repository.
    Enforces repository isolation, resolves embeddings dynamically, and queries ChromaDB.
    """
    logger.info(f"Retrieving top {top_k} chunks for repository '{repository_id}' and question: '{question[:50]}'")
    
    # Check index existence
    if not chroma_store.collection_exists(repository_id):
        raise RepositoryNotIndexedError(f"Repository '{repository_id}' has not been indexed yet. Please run indexing first.")
        
    # Check if empty
    count = chroma_store.chunk_count(repository_id)
    if count == 0:
        raise EmptyCollectionError(f"Repository '{repository_id}' is indexed but contains 0 chunks.")
        
    # Initialize embedding provider
    provider = get_embedding_provider()
    
    # Generate query embedding
    query_embeddings = provider.embed([question])
    if not query_embeddings:
        raise RuntimeError("Failed to generate embedding for the retrieval query.")
    query_embedding = query_embeddings[0]
    
    # Query ChromaDB
    results = chroma_store.query(repository_id, query_embedding, top_k)
    return results
