import os
import hashlib
import logging
try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

from retrieval.exceptions import VectorStoreWriteError, VectorStoreQueryError

logger = logging.getLogger("code_intelligence.vectorstore")

COLLECTION_NAME = "repository_chunks"

# In-memory mock classes for fallback when chromadb is missing
class MockCollection:
    def __init__(self):
        self.data = {}
        self.deleted_repos = set()
        
    def delete(self, where=None):
        repo_id = where.get("repository_id") if where else None
        if repo_id:
            self.deleted_repos.add(repo_id)
            self.data = {k: v for k, v in self.data.items() if v[2].get("repository_id") != repo_id}
            
    def add(self, ids, embeddings, documents, metadatas):
        for idx, c_id in enumerate(ids):
            self.data[c_id] = (embeddings[idx], documents[idx], metadatas[idx])
            
    def query(self, query_embeddings, n_results, where=None):
        repo_id = where.get("repository_id") if where else None
        q_emb = query_embeddings[0]
        
        # Filter by repository_id
        filtered = {k: v for k, v in self.data.items() if v[2].get("repository_id") == repo_id}
        
        # Helper for Euclidean distance
        def euclidean_distance(v1, v2):
            import math
            return math.sqrt(sum((x - y) ** 2 for x, y in zip(v1, v2)))
            
        scored = []
        for k, (emb, doc, meta) in filtered.items():
            dist = euclidean_distance(q_emb, emb)
            scored.append((dist, k, doc, meta))
            
        scored.sort(key=lambda x: x[0]) # distance ascending
        top = scored[:n_results]
        
        if not top:
            return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}
            
        return {
            "ids": [[x[1] for x in top]],
            "distances": [[x[0] for x in top]],
            "documents": [[x[2] for x in top]],
            "metadatas": [[x[3] for x in top]]
        }
        
    def get(self, where=None, include=None):
        repo_id = where.get("repository_id") if where else None
        filtered_ids = [k for k, v in self.data.items() if v[2].get("repository_id") == repo_id]
        return {"ids": filtered_ids}

class MockClient:
    def __init__(self):
        self.collection = MockCollection()
        
    def get_or_create_collection(self, name):
        return self.collection

_client = None
_mock_client = None
_force_mock = False
_in_memory_registry: set = set()  # tracks registered repos when using mock client

def get_chroma_client():
    global _client, _mock_client, _force_mock
    if not HAS_CHROMA or _force_mock:
        if _mock_client is None:
            logger.info("Initializing in-memory fallback client.")
            _mock_client = MockClient()
        return _mock_client

    if _client is None:
        chroma_host = os.getenv("CHROMA_HOST")
        if chroma_host:
            chroma_api_key = os.getenv("CHROMA_API_KEY")
            chroma_tenant = os.getenv("CHROMA_TENANT")
            chroma_database = os.getenv("CHROMA_DATABASE")
            headers = {"x-chroma-token": chroma_api_key} if chroma_api_key else None
            
            logger.info(f"Connecting to hosted ChromaDB client at {chroma_host} (database: {chroma_database})")
            try:
                _client = chromadb.HttpClient(
                    host=chroma_host,
                    ssl=True,
                    tenant=chroma_tenant,
                    database=chroma_database,
                    headers=headers,
                    settings=Settings(anonymized_telemetry=False)
                )
            except Exception as e:
                logger.error(f"Failed to connect to hosted ChromaDB client at {chroma_host}: {e}")
                raise RuntimeError(f"ChromaDB cloud connection failure: {e}") from e
        else:
            db_dir = os.getenv("CHROMA_DB_DIR", "./db/chroma")
            try:
                os.makedirs(db_dir, exist_ok=True)
                _client = chromadb.PersistentClient(
                    path=db_dir,
                    settings=Settings(anonymized_telemetry=False)
                )
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB client at {db_dir}: {e}")
                raise RuntimeError(f"ChromaDB initialization failure: {e}") from e
    return _client


def get_collection():
    client = get_chroma_client()
    try:
        return client.get_or_create_collection(name=COLLECTION_NAME)
    except Exception as e:
        logger.error(f"Failed to get or create ChromaDB collection '{COLLECTION_NAME}': {e}")
        raise RuntimeError(f"ChromaDB collection access failure: {e}") from e


def _get_registry_file():
    db_dir = os.getenv("CHROMA_DB_DIR", "./db/chroma")
    return os.path.join(db_dir, "indexed_repositories.txt")


def _get_registered_repositories() -> set[str]:
    reg_file = _get_registry_file()
    if not os.path.exists(reg_file):
        return set()
    try:
        with open(reg_file, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    except Exception as e:
        logger.error(f"Failed to read repository registry: {e}")
        return set()


def _register_repository(repository_id: str):
    global _in_memory_registry
    # Always register in-memory (works for both real and mock client)
    _in_memory_registry.add(repository_id)
    
    # Also persist to file if using real ChromaDB
    if not HAS_CHROMA:
        return
    db_dir = os.getenv("CHROMA_DB_DIR", "./db/chroma")
    reg_file = _get_registry_file()
    try:
        os.makedirs(db_dir, exist_ok=True)
        repos = _get_registered_repositories()
        if repository_id not in repos:
            repos.add(repository_id)
            with open(reg_file, "w", encoding="utf-8") as f:
                f.write("\n".join(repos))
    except Exception as e:
        logger.error(f"Failed to register repository '{repository_id}' to disk: {e}")


def store_chunks(repository_id: str, chunks: list, embeddings: list[list[float]]) -> None:
    """
    Stores a list of Chunk objects and their embeddings in ChromaDB.
    Enforces repository isolation and overwrites existing chunks for the same repository_id.
    """
    global _client, _force_mock
    if len(chunks) != len(embeddings):
        raise ValueError("The number of chunks and embeddings must match.")
        
    try:
        collection = get_collection()
        # Overwrite strategy: clear any existing chunks for this repository_id first
        logger.info(f"Clearing existing chunks for repository '{repository_id}' in ChromaDB.")
        collection.delete(where={"repository_id": repository_id})
        _register_repository(repository_id)
    except Exception as e:
        logger.error(f"Failed to clear existing chunks for repository '{repository_id}': {e}")
        if os.getenv("CHROMA_HOST") and not _force_mock:
            logger.warning(f"Chroma Cloud write failed during clear ({e}). Falling back globally to in-memory mock client.")
            _force_mock = True
            _client = None
            return store_chunks(repository_id, chunks, embeddings)
        raise VectorStoreWriteError(f"ChromaDB delete failure during re-indexing overwrite: {e}") from e

    if not chunks:
        return

    ids = []
    documents = []
    metadatas = []
    
    for idx, chunk in enumerate(chunks):
        # Generate deterministic chunk ID: md5 hash of repository_id + file_path + start_line + end_line
        unique_key = f"{repository_id}:{chunk.file_path}:{chunk.start_line}:{chunk.end_line}"
        chunk_id = hashlib.md5(unique_key.encode("utf-8")).hexdigest()
        
        ids.append(chunk_id)
        documents.append(chunk.content)
        
        metadatas.append({
            "repository_id": repository_id,
            "file_path": chunk.file_path,
            "language": chunk.language or "",
            "symbol_type": chunk.symbol_type or "",
            "symbol_name": chunk.symbol_name or "",
            "class_name": chunk.class_name or "",
            "start_line": chunk.start_line,
            "end_line": chunk.end_line
        })
        
    try:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Successfully stored {len(chunks)} chunks for repository '{repository_id}' in ChromaDB.")
    except Exception as e:
        logger.error(f"Failed to store chunks in ChromaDB: {e}")
        if os.getenv("CHROMA_HOST") and not _force_mock:
            logger.warning(f"Chroma Cloud write failed during add ({e}). Falling back globally to in-memory mock client.")
            _force_mock = True
            _client = None
            return store_chunks(repository_id, chunks, embeddings)
        raise VectorStoreWriteError(f"ChromaDB write failure: {e}") from e


def query(repository_id: str, query_embedding: list[float], top_k: int):
    """
    Queries ChromaDB for the top_k most similar chunks, hard-filtered by repository_id.
    """
    collection = get_collection()
    
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"repository_id": repository_id}
        )
    except Exception as e:
        logger.error(f"ChromaDB query failed for repository '{repository_id}': {e}")
        raise VectorStoreQueryError(f"ChromaDB query failure: {e}") from e

    retrieved_chunks = []
    
    if not results or not results.get("ids") or not results["ids"][0]:
        return []
        
    ids = results["ids"][0]
    distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(ids)
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    
    from retrieval.vector_retriever import Chunk, RetrievedChunk
    
    for idx in range(len(ids)):
        meta = metadatas[idx]
        score = float(distances[idx])
        
        chunk = Chunk(
            repository_id=meta.get("repository_id", repository_id),
            file_path=meta.get("file_path", ""),
            language=meta.get("language", ""),
            symbol_type=meta.get("symbol_type", ""),
            symbol_name=meta.get("symbol_name") or None,
            class_name=meta.get("class_name") or None,
            start_line=int(meta.get("start_line", 0)),
            end_line=int(meta.get("end_line", 0)),
            content=documents[idx]
        )
        retrieved_chunks.append(RetrievedChunk(
            chunk=chunk,
            score=score,
            retrieval_method="vector"
        ))
        
    # Convert distance to similarity score: similarity = 1.0 / (1.0 + distance)
    for rc in retrieved_chunks:
        rc.score = 1.0 / (1.0 + rc.score)
        
    retrieved_chunks.sort(key=lambda x: x.score, reverse=True)
    return retrieved_chunks


def collection_exists(repository_id: str) -> bool:
    """
    Checks if the repository has been indexed (is in the registry).
    Checks both the in-memory registry (for mock client) and the file registry (for real ChromaDB).
    """
    try:
        # Check in-memory registry first (works for both real and mock)
        if repository_id in _in_memory_registry:
            return True
        # Fall back to file-based registry for real ChromaDB
        return repository_id in _get_registered_repositories()
    except Exception:
        return False


def chunk_count(repository_id: str) -> int:
    """
    Returns the number of chunks stored for the given repository_id.
    """
    try:
        collection = get_collection()
        results = collection.get(where={"repository_id": repository_id}, include=[])
        if results and "ids" in results:
            return len(results["ids"])
        return 0
    except Exception as e:
        logger.error(f"Failed to get chunk count for repository '{repository_id}': {e}")
        return 0
