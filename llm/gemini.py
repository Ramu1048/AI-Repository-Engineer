"""
Gemini LLM Provider Implementation.
Supports Google Gemini API with fallback to REST API to ensure robust execution.
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


class GeminiProvider(LLMProvider):
    """LLM Provider implementation for Google Gemini."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise LLMConfigurationError(
                "Gemini API key missing. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
            )
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.timeout = timeout

    def generate(self, system_prompt: str, user_prompt: str, context: str = "") -> str:
        """Calls Gemini API to generate response grounded in provided context."""

        combined_user_content = user_prompt
        if context and context.strip():
            combined_user_content = f"--- RETRIEVED CONTEXT ---\n{context}\n\n--- QUESTION / USER PROMPT ---\n{user_prompt}"

        # Build payload for Gemini REST API
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": combined_user_content}],
                }
            ],
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.95,
                "maxOutputTokens": 8192,
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
                    raise LLMResponseError(f"Gemini API HTTP Error {status_code}: {response_data}")

                candidates = response_data.get("candidates", [])
                if not candidates:
                    raise LLMResponseError("Gemini returned response with no candidates.")

                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    raise LLMResponseError("Gemini candidate content has no text parts.")

                answer = "".join(part.get("text", "") for part in parts).strip()
                if not answer:
                    raise LLMResponseError("Gemini returned empty text response.")

                return answer

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else str(e)
            if e.code == 429:
                raise LLMRateLimitError(f"Gemini API rate limit exceeded (429): {error_body}") from e
            elif e.code in (401, 403):
                raise LLMConfigurationError(f"Gemini API Authentication error ({e.code}): {error_body}") from e
            else:
                raise LLMResponseError(f"Gemini API HTTP Error {e.code}: {error_body}") from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError) or "timed out" in str(e.reason).lower():
                raise LLMTimeoutError(f"Gemini API request timed out after {self.timeout}s: {e.reason}") from e
            raise LLMProviderError(f"Network error connecting to Gemini API: {e.reason}") from e
        except json.JSONDecodeError as e:
            raise LLMResponseError(f"Failed to parse Gemini API JSON response: {e}") from e
        except Exception as e:
            if isinstance(e, LLMProviderError):
                raise
            raise LLMProviderError(f"Unexpected error in GeminiProvider: {e}") from e
