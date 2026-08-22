"""
Ollama LLM Provider Implementation.
Connects to local or remote Ollama server REST API.
"""

import json
import os
import urllib.error
import urllib.request
from typing import Optional

from llm.base import (
    LLMConfigurationError,
    LLMProvider,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)


class OllamaProvider(LLMProvider):
    """LLM Provider implementation for Ollama local models."""

    def __init__(
        self,
        host: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 90.0,
    ):
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3")
        self.timeout = timeout

    def generate(self, system_prompt: str, user_prompt: str, context: str = "") -> str:
        """Calls Ollama `/api/chat` or `/api/generate` endpoint."""

        endpoint = f"{self.host}/api/chat"
        headers = {"Content-Type": "application/json"}

        user_content = user_prompt
        if context and context.strip():
            user_content = f"--- RETRIEVED CONTEXT ---\n{context}\n\n--- QUESTION / USER PROMPT ---\n{user_prompt}"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
            },
        }

        try:
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                status_code = response.getcode()
                response_data = json.loads(response.read().decode("utf-8"))

                if status_code != 200:
                    raise LLMResponseError(f"Ollama API HTTP Error {status_code}: {response_data}")

                message = response_data.get("message", {})
                content = message.get("content", "").strip()

                if not content:
                    # Try fallback to 'response' field in case of generate endpoint style
                    content = response_data.get("response", "").strip()

                if not content:
                    raise LLMResponseError("Ollama returned empty response content.")

                return content

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else str(e)
            if e.code == 429:
                raise LLMRateLimitError(f"Ollama API rate limit exceeded: {error_body}") from e
            raise LLMResponseError(f"Ollama API HTTP Error {e.code}: {error_body}") from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError) or "timed out" in str(e.reason).lower():
                raise LLMTimeoutError(f"Ollama API request timed out after {self.timeout}s: {e.reason}") from e
            raise LLMConfigurationError(
                f"Failed to connect to Ollama server at '{self.host}'. Ensure Ollama is running: {e.reason}"
            ) from e
        except json.JSONDecodeError as e:
            raise LLMResponseError(f"Failed to parse Ollama JSON response: {e}") from e
        except Exception as e:
            if isinstance(e, LLMProviderError):
                raise
            raise LLMProviderError(f"Unexpected error in OllamaProvider: {e}") from e
