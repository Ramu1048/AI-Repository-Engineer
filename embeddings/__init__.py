import os
from .provider import EmbeddingProvider, HuggingFaceEmbeddingProvider, OllamaEmbeddingProvider, GeminiEmbeddingProvider

def get_embedding_provider() -> EmbeddingProvider:
    """
    Factory function to resolve and initialize the configured EmbeddingProvider
    based on the EMBEDDING_PROVIDER environment variable.
    """
    provider_name = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
    if provider_name == "huggingface":
        return HuggingFaceEmbeddingProvider()
    elif provider_name == "gemini":
        return GeminiEmbeddingProvider()
    elif provider_name == "ollama":
        return OllamaEmbeddingProvider()
    else:
        raise ValueError(f"Unknown embedding provider: {provider_name}")
