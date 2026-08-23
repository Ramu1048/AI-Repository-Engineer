from abc import ABC, abstractmethod
import os
import logging
from retrieval.exceptions import (
    EmbeddingProviderConfigError,
    EmbeddingProviderUnreachableError
)

logger = logging.getLogger("code_intelligence.embeddings")

class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generates embeddings for a list of texts."""
        pass


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.model = None

    def _init_model(self):
        if self.model is None:
            if not self.model_name:
                raise EmbeddingProviderConfigError("HF_EMBEDDING_MODEL environment variable or model_name parameter is not set.")
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
            except ImportError as e:
                logger.error("sentence-transformers package is missing.")
                raise EmbeddingProviderConfigError(
                    "sentence-transformers is not installed. Please install it to use HuggingFace embeddings."
                ) from e
            except Exception as e:
                logger.error(f"Failed to load sentence-transformer model '{self.model_name}': {e}")
                raise EmbeddingProviderConfigError(
                    f"Failed to initialize HuggingFace model '{self.model_name}': {e}"
                ) from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._init_model()
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"HuggingFace embedding generation failed: {e}")
            raise EmbeddingProviderUnreachableError(f"HuggingFace embedding generation failed: {e}") from e


class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = None, api_key: str = None):
        self.model_name = model_name or os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.configured = False

    def _init_api(self):
        if not self.configured:
            if not self.api_key:
                raise EmbeddingProviderConfigError(
                    "Gemini API key missing. Set GEMINI_API_KEY in environment variables."
                )
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.configured = True
            except Exception as e:
                logger.error(f"Failed to configure Google Generative AI client: {e}")
                raise EmbeddingProviderConfigError(f"Failed to configure Google Generative AI client: {e}") from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._init_api()
        try:
            import google.generativeai as genai
            result = genai.embed_content(
                model=self.model_name,
                content=texts,
                task_type="retrieval_document"
            )
            return result["embedding"]
        except Exception as e:
            logger.error(f"Gemini embedding generation failed: {e}")
            raise EmbeddingProviderUnreachableError(f"Gemini embedding generation failed: {e}") from e


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = None, base_url: str = None):
        self.model_name = model_name or os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.model_name:
            raise EmbeddingProviderConfigError("OLLAMA_EMBEDDING_MODEL environment variable or model_name parameter is not set.")
        
        import requests
        try:
            # Try Ollama's newer /api/embed (which supports batch inputs)
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model_name, "input": texts},
                timeout=60
            )
            if response.status_code == 200:
                result = response.json()
                if "embeddings" in result:
                    return result["embeddings"]
        except requests.RequestException as e:
            logger.warning(f"Ollama batch /api/embed failed or connection refused: {e}. Falling back to sequential /api/embeddings.")
        except Exception as e:
            logger.warning(f"Ollama batch /api/embed unexpected error: {e}. Falling back to sequential /api/embeddings.")

        # Fallback to sequential calls to /api/embeddings
        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model_name, "prompt": text},
                    timeout=30
                )
                response.raise_for_status()
                emb = response.json().get("embedding")
                if not emb:
                    raise EmbeddingProviderUnreachableError(
                        f"Ollama returned empty embedding for prompt: {text[:50]}..."
                    )
                embeddings.append(emb)
            except requests.RequestException as e:
                logger.error(f"Ollama service connection failed at {self.base_url}: {e}")
                raise EmbeddingProviderUnreachableError(f"Ollama service unreachable at {self.base_url}: {e}") from e
            except Exception as e:
                logger.error(f"Ollama embedding failure: {e}")
                raise EmbeddingProviderUnreachableError(f"Ollama embedding failure: {e}") from e
        return embeddings
