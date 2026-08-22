"""
LLM Provider Base Module.
Defines abstract interface, exception hierarchy, and provider factory.
"""

from abc import ABC, abstractmethod
import os
from typing import Optional


class LLMProviderError(Exception):
    """Base exception for all LLM provider errors."""
    pass


class LLMConfigurationError(LLMProviderError):
    """Raised when LLM configuration (e.g. missing API key, invalid host) is incorrect."""
    pass


class LLMTimeoutError(LLMProviderError):
    """Raised when an LLM provider request times out."""
    pass


class LLMRateLimitError(LLMProviderError):
    """Raised when an LLM provider rate limit is exceeded."""
    pass


class LLMResponseError(LLMProviderError):
    """Raised when an LLM returns an empty, malformed, or error response."""
    pass


class LLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, context: str = "") -> str:
        """
        Generate a text response given system prompt, user prompt, and context.

        :param system_prompt: Instructions defining AI role and constraints.
        :param user_prompt: The question or query from the user/analyzer.
        :param context: Retrieved context string or code snippet.
        :return: Generated string answer.
        """
        pass


def get_llm_provider(provider_name: Optional[str] = None) -> LLMProvider:
    """
    Factory function to retrieve the configured LLM provider instance.
    Reads `LLM_PROVIDER` environment variable ('gemini' or 'ollama') if not specified.
    """
    from llm.gemini import GeminiProvider
    from llm.ollama import OllamaProvider

    name = (provider_name or os.getenv("LLM_PROVIDER", "gemini")).lower().strip()

    if name == "gemini":
        return GeminiProvider()
    elif name == "ollama":
        return OllamaProvider()
    else:
        raise LLMConfigurationError(
            f"Unsupported LLM_PROVIDER '{name}'. Supported options are 'gemini' and 'ollama'."
        )
