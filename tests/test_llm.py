"""
Unit tests for LLM providers (Gemini, Ollama) and provider factory.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from llm.base import (
    LLMConfigurationError,
    LLMProvider,
    LLMProviderError,
    LLMResponseError,
    LLMTimeoutError,
    get_llm_provider,
)
from llm.gemini import GeminiProvider
from llm.ollama import OllamaProvider


class MockLLMProvider(LLMProvider):
    def generate(self, system_prompt: str, user_prompt: str, context: str = "") -> str:
        return f"Mock answer for '{user_prompt}' based on context length {len(context)}."


class TestLLMProviders(unittest.TestCase):

    def test_provider_factory_default(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key", "LLM_PROVIDER": "gemini"}):
            provider = get_llm_provider()
            self.assertIsInstance(provider, GeminiProvider)

    def test_provider_factory_ollama(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}):
            provider = get_llm_provider()
            self.assertIsInstance(provider, OllamaProvider)

    def test_provider_factory_invalid(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "unknown_provider"}):
            with self.assertRaises(LLMConfigurationError):
                get_llm_provider()

    def test_gemini_missing_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(LLMConfigurationError):
                GeminiProvider(api_key=None)

    def test_gemini_generate_success(self):
        mock_response = MagicMock()
        mock_response.getcode.return_value = 200
        mock_response.read.return_value = b'{"candidates": [{"content": {"parts": [{"text": "Grounded answer from Gemini"}]}}]}'

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = mock_response

            provider = GeminiProvider(api_key="test_key")
            result = provider.generate("System prompt", "User prompt", "Context code")
            self.assertEqual(result, "Grounded answer from Gemini")

    def test_ollama_generate_success(self):
        mock_response = MagicMock()
        mock_response.getcode.return_value = 200
        mock_response.read.return_value = b'{"message": {"content": "Ollama local model answer"}}'

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = mock_response

            provider = OllamaProvider(host="http://localhost:11434")
            result = provider.generate("System prompt", "User prompt", "Context code")
            self.assertEqual(result, "Ollama local model answer")


if __name__ == "__main__":
    unittest.main()
