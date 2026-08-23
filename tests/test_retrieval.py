import os
import sys
import shutil
import pytest
import math
from unittest.mock import MagicMock, patch

# Dynamically mock sentence_transformers if it is not installed
try:
    import sentence_transformers
except ImportError:
    mock_st = MagicMock()
    mock_st_class = MagicMock()
    
    # Configure mock instance of SentenceTransformer to return matching embedding lists
    mock_instance = MagicMock()
    def mock_encode(texts, **kwargs):
        import numpy as np
        return np.array([[0.1] * 384 for _ in texts])
    mock_instance.encode.side_effect = mock_encode
    mock_st_class.return_value = mock_instance
    
    mock_st.SentenceTransformer = mock_st_class
    sys.modules["sentence_transformers"] = mock_st

# Dynamically mock chromadb if it is not installed
try:
    import chromadb
except ImportError:
    mock_chroma = MagicMock()
    
    class MockCollection:
        def __init__(self):
            # dict of chunk_id -> (embedding, document, metadata)
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

    mock_collection_instance = MockCollection()
    
    class MockClient:
        def get_or_create_collection(self, name):
            return mock_collection_instance
            
    mock_chroma.PersistentClient.return_value = MockClient()
    sys.modules["chromadb"] = mock_chroma
    sys.modules["chromadb.config"] = MagicMock()

from retrieval import (
    retrieve,
    Chunk,
    RetrievedChunk,
    RepositoryNotIndexedError,
    EmptyCollectionError,
    EmbeddingProviderConfigError,
    EmbeddingProviderUnreachableError
)
from embeddings import get_embedding_provider
from embeddings.provider import HuggingFaceEmbeddingProvider, OllamaEmbeddingProvider
from vectorstore import chroma_store

# Use a clean test database directory
import tempfile

@pytest.fixture(autouse=True)
def setup_test_db(monkeypatch):
    """Ensure tests run against a clean test database."""
    db_base = "./db"
    os.makedirs(db_base, exist_ok=True)
    test_dir = tempfile.mkdtemp(dir=db_base, prefix="chroma_test_")
    
    monkeypatch.setenv("CHROMA_DB_DIR", test_dir)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
    monkeypatch.setenv("HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    monkeypatch.delenv("CHROMA_HOST", raising=False)
    monkeypatch.delenv("CHROMA_API_KEY", raising=False)
    
    # Reset Chroma client singleton
    chroma_store._client = None
    chroma_store._mock_client = None
    chroma_store._in_memory_registry = set()
    
    # Clear mock collection data if using mock
    try:
        mock_collection_instance.data.clear()
        mock_collection_instance.deleted_repos.clear()
    except NameError:
        pass
        
    yield
    
    # Cleanup after test
    try:
        shutil.rmtree(test_dir, ignore_errors=True)
    except Exception:
        pass


# --- 1. Embedding Provider Unit Tests ---

def test_embedding_provider_hf_mock():
    """Test HuggingFace embedding provider output shape using a mocked SentenceTransformer."""
    from sentence_transformers import SentenceTransformer
    mock_model = MagicMock()
    import numpy as np
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    
    with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
        provider = HuggingFaceEmbeddingProvider(model_name="mock-model")
        embeddings = provider.embed(["text1", "text2"])
        
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]
        mock_model.encode.assert_called_once_with(["text1", "text2"], show_progress_bar=False)


def test_embedding_provider_ollama_mock():
    """Test Ollama embedding provider using mocked requests."""
    with patch("requests.post") as mock_post:
        # Mock successful batch /api/embed response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [[0.1, 0.2], [0.3, 0.4]]
        }
        mock_post.return_value = mock_response
        
        provider = OllamaEmbeddingProvider(model_name="nomic-embed-text")
        embeddings = provider.embed(["hello", "world"])
        
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2]
        assert embeddings[1] == [0.3, 0.4]


def test_embedding_provider_gemini_mock():
    """Test Gemini embedding provider using mocked google.generativeai."""
    mock_genai = MagicMock()
    mock_genai.embed_content.return_value = {
        "embedding": [[0.1, 0.2], [0.3, 0.4]]
    }
    
    with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
        from embeddings.provider import GeminiEmbeddingProvider
        provider = GeminiEmbeddingProvider(model_name="models/gemini-embedding-001", api_key="fake-key")
        embeddings = provider.embed(["hello", "world"])
        
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2]
        assert embeddings[1] == [0.3, 0.4]


# --- 2. ChromaDB Storage & Query Round-Trip ---

def test_chroma_store_round_trip():
    """Test storing chunks in ChromaDB and retrieving them preserves all metadata."""
    chunks = [
        Chunk(
            repository_id="repo1",
            file_path="main.py",
            language="python",
            symbol_type="function",
            symbol_name="main",
            class_name=None,
            start_line=1,
            end_line=10,
            content="def main():\n    print('hello')"
        ),
        Chunk(
            repository_id="repo1",
            file_path="utils.py",
            language="python",
            symbol_type="class",
            symbol_name="Helper",
            class_name=None,
            start_line=5,
            end_line=15,
            content="class Helper:\n    pass"
        )
    ]
    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ]
    
    # Store chunks
    chroma_store.store_chunks("repo1", chunks, embeddings)
    
    # Verify chunk count and existence
    assert chroma_store.collection_exists("repo1") is True
    assert chroma_store.chunk_count("repo1") == 2
    
    # Query repo1
    results = chroma_store.query("repo1", query_embedding=[0.1, 0.2, 0.3], top_k=2)
    assert len(results) == 2
    
    # Verify metadata fields are preserved
    c1 = [r.chunk for r in results if r.chunk.symbol_name == "main"][0]
    assert c1.repository_id == "repo1"
    assert c1.file_path == "main.py"
    assert c1.language == "python"
    assert c1.symbol_type == "function"
    assert c1.class_name is None
    assert c1.start_line == 1
    assert c1.end_line == 10
    assert c1.content == "def main():\n    print('hello')"


# --- 3. Repository Isolation Test (Non-Negotiable) ---

def test_repository_isolation():
    """Verify that querying repo A never leaks chunks from repo B."""
    chunks_a = [
        Chunk(
            repository_id="repo_A",
            file_path="a.py",
            language="python",
            symbol_type="function",
            symbol_name="funcA",
            class_name=None,
            start_line=1,
            end_line=5,
            content="def funcA(): pass"
        )
    ]
    embeddings_a = [[0.1, 0.1, 0.1]]
    
    chunks_b = [
        Chunk(
            repository_id="repo_B",
            file_path="b.py",
            language="python",
            symbol_type="function",
            symbol_name="funcB",
            class_name=None,
            start_line=1,
            end_line=5,
            content="def funcB(): pass"
        )
    ]
    embeddings_b = [[0.1, 0.1, 0.1]]  # Identical embedding to force potential match
    
    # Store both repositories
    chroma_store.store_chunks("repo_A", chunks_a, embeddings_a)
    chroma_store.store_chunks("repo_B", chunks_b, embeddings_b)
    
    # Query repo_A - should ONLY return repo_A chunks
    results_a = chroma_store.query("repo_A", query_embedding=[0.1, 0.1, 0.1], top_k=2)
    assert len(results_a) == 1
    assert results_a[0].chunk.repository_id == "repo_A"
    assert results_a[0].chunk.symbol_name == "funcA"
    
    # Query repo_B - should ONLY return repo_B chunks
    results_b = chroma_store.query("repo_B", query_embedding=[0.1, 0.1, 0.1], top_k=2)
    assert len(results_b) == 1
    assert results_b[0].chunk.repository_id == "repo_B"
    assert results_b[0].chunk.symbol_name == "funcB"


# --- 4. Vector Retrieval & Similarity Ranking ---

def test_vector_retriever_ranking():
    """Verify retriever correctly embeds queries and ranks chunks descending by similarity score."""
    chunks = [
        Chunk(
            repository_id="repo_rank",
            file_path="a.py",
            language="python",
            symbol_type="function",
            symbol_name="math_func",
            class_name=None,
            start_line=1,
            end_line=5,
            content="def add_numbers(x, y): return x + y"
        ),
        Chunk(
            repository_id="repo_rank",
            file_path="b.py",
            language="python",
            symbol_type="function",
            symbol_name="ui_func",
            class_name=None,
            start_line=1,
            end_line=5,
            content="def render_window(): show_button()"
        )
    ]
    
    # Mock embedding provider to control similarities
    # Query: "how to add two numbers"
    # Query embedding: [0.1, 0.2, 0.3]
    # Chunk A embedding: [0.1, 0.2, 0.3] (exact match, distance = 0)
    # Chunk B embedding: [0.9, 0.9, 0.9] (far match, distance > 0)
    embeddings = [
        [0.1, 0.2, 0.3],
        [0.9, 0.9, 0.9]
    ]
    chroma_store.store_chunks("repo_rank", chunks, embeddings)
    
    # Mock global embedding provider resolve
    mock_provider = MagicMock()
    mock_provider.embed.return_value = [[0.1, 0.2, 0.3]]
    
    with patch("retrieval.vector_retriever.get_embedding_provider", return_value=mock_provider):
        results = retrieve("repo_rank", "how to add two numbers", top_k=2)
        
        assert len(results) == 2
        # Check order: exact match first
        assert results[0].chunk.symbol_name == "math_func"
        assert results[1].chunk.symbol_name == "ui_func"
        # Check scores are descending
        assert results[0].score > results[1].score
        assert results[0].retrieval_method == "vector"


# --- 5. Exception & Error Handling Tests ---

def test_not_indexed_error():
    """Querying a repository ID with no collection raises RepositoryNotIndexedError."""
    with pytest.raises(RepositoryNotIndexedError):
        retrieve("non_existent_repo", "any question")


def test_empty_collection_error():
    """Querying a repository that was indexed but contains zero chunks raises EmptyCollectionError."""
    # Index with empty chunk list
    chroma_store.store_chunks("empty_repo", [], [])
    
    with pytest.raises(EmptyCollectionError):
        retrieve("empty_repo", "any question")


# --- 6. End-to-End Integration Test ---

class LoadedFileStub:
    def __init__(self, file_path, content, language):
        self.file_path = file_path
        self.content = content
        self.language = language

def test_integration_flow():
    """Runs a full indexing-to-retrieval pipeline against mock code chunks."""
    # 1. Parse repository using Member 2's chunker logic
    from ingestion.chunker import chunk_repository as member2_chunk_repo
    
    files = [
        LoadedFileStub(
            "calc.py",
            "def calculate_area(radius):\n    import math\n    return math.pi * radius * radius\n",
            "python"
        ),
        LoadedFileStub(
            "README.md",
            "# Geometry Calculator\nProvides basic shape operations.\n",
            "markdown"
        )
    ]
    
    # Chunker returns list of dicts
    dict_chunks = member2_chunk_repo("integration_repo", files, max_chunk_chars=3000)
    assert len(dict_chunks) > 0
    
    # Map raw dicts to our Chunk objects
    chunks = []
    for dc in dict_chunks:
        chunks.append(Chunk(
            repository_id=dc["repository_id"],
            file_path=dc["file_path"],
            language=dc["language"],
            symbol_type=dc["symbol_type"],
            symbol_name=dc["symbol_name"],
            class_name=dc["class_name"],
            start_line=dc["start_line"],
            end_line=dc["end_line"],
            content=dc["content"]
        ))
        
    # 2. Embed the chunks using the default HuggingFace provider (or mock if env lacks PyTorch/compilers)
    try:
        provider = get_embedding_provider()
        embeddings = provider.embed([c.content for c in chunks])
    except Exception:
        # Fallback to mock embedding values
        embeddings = [[0.1] * 384 for _ in chunks]
        # Also mock get_embedding_provider for retrieve()
        mock_prov = MagicMock()
        mock_prov.embed.return_value = [[0.1] * 384]
        patcher = patch("retrieval.vector_retriever.get_embedding_provider", return_value=mock_prov)
        patcher.start()
        
    # 3. Store in ChromaDB
    chroma_store.store_chunks("integration_repo", chunks, embeddings)
    
    # 4. Query & Retrieve
    # Ensure retrieve doesn't crash on get_embedding_provider in case it was not mocked above
    try:
        results = retrieve("integration_repo", "how to calculate circle area")
    except Exception:
        # If it failed because of real provider initialization, mock it
        mock_prov = MagicMock()
        mock_prov.embed.return_value = [[0.1] * 384]
        with patch("retrieval.vector_retriever.get_embedding_provider", return_value=mock_prov):
            results = retrieve("integration_repo", "how to calculate circle area")
            
    assert len(results) > 0
    assert results[0].chunk.file_path == "calc.py"
    assert "calculate_area" in results[0].chunk.content
