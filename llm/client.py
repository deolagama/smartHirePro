"""
SmartHirePro - LLM Client
===========================
Thin, provider-agnostic wrapper around large language model APIs.

Supported providers:
  - **openai**  – GPT-4o-mini / GPT-4o via openai SDK.
  - **ollama**  – Llama 3 / Mistral / any local model via Ollama REST API.

The :class:`LLMClient` exposes a single ``chat()`` method that accepts a list
of messages (dicts with "role" + "content" keys) and returns the model's
text response.

Usage:
    from llm.client import LLMClient
    client = LLMClient()
    response = client.chat([
        {"role": "system", "content": "You are helpful."},
        {"role": "user",   "content": "Summarise this resume …"},
    ])
"""

from __future__ import annotations

import json
import time
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)

Message = dict[str, str]   # {"role": "...", "content": "..."}


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseLLMClient:
    """Minimal interface every LLM backend must satisfy."""

    def chat(self, messages: list[Message], **kwargs) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

class OpenAIClient(BaseLLMClient):
    """
    OpenAI chat completion client.

    Args:
        model:       Model name (e.g. "gpt-4o-mini", "gpt-4o").
        temperature: Sampling temperature (0 = deterministic).
        max_tokens:  Maximum tokens in the completion.
        timeout:     Request timeout in seconds.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int = 120,
    ) -> None:
        from config import settings
        from openai import OpenAI

        if not settings.OPENAI_API_KEY:
            raise EnvironmentError(
                "OPENAI_API_KEY is missing. Set it in your .env file."
            )

        self.model = model or settings.OPENAI_MODEL
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY, timeout=timeout)

        logger.info("OpenAI client ready: model=%s", self.model)

    def chat(self, messages: list[Message], **kwargs) -> str:
        """Send a chat completion request and return the text response."""
        start = time.perf_counter()

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )

        text = response.choices[0].message.content or ""
        elapsed = time.perf_counter() - start

        logger.info(
            "OpenAI response received: tokens=%d, elapsed=%.2fs",
            response.usage.total_tokens if response.usage else -1,
            elapsed,
        )
        return text.strip()


# ---------------------------------------------------------------------------
# Ollama backend (local LLM)
# ---------------------------------------------------------------------------

class OllamaClient(BaseLLMClient):
    """
    Ollama local LLM client (Llama 3, Mistral, Gemma, etc.).

    Communicates with the Ollama REST API running at *base_url*.

    Args:
        model:    Ollama model tag (e.g. "llama3", "mistral").
        base_url: Ollama server URL.
        temperature: Sampling temperature.
        max_tokens: Max tokens in the response.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        from config import settings

        self.model = model or settings.OLLAMA_MODEL
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.max_tokens = max_tokens or settings.LLM_MAX_TOKENS

        logger.info(
            "Ollama client ready: model=%s, base_url=%s", self.model, self.base_url
        )

    def chat(self, messages: list[Message], **kwargs) -> str:
        """Send a chat request to the Ollama REST API."""
        import requests

        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        start = time.perf_counter()
        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
        except Exception as exc:
            logger.error("Ollama request failed: %s", exc)
            raise

        elapsed = time.perf_counter() - start
        data = resp.json()
        text = data.get("message", {}).get("content", "").strip()

        logger.info("Ollama response received: elapsed=%.2fs", elapsed)
        return text


# ---------------------------------------------------------------------------
# Factory / high-level client
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Provider-agnostic LLM wrapper that routes to OpenAI or Ollama based on
    ``settings.LLM_PROVIDER``.

    Provides helpers for JSON-structured responses with automatic retry and
    fallback parsing.

    Args:
        provider: "openai" | "ollama".  Defaults to settings.LLM_PROVIDER.
        **kwargs: Forwarded to the underlying client constructor.
    """

    def __init__(self, provider: str | None = None, **kwargs) -> None:
        from config import settings

        _provider = (provider or settings.LLM_PROVIDER).lower()

        if _provider == "openai":
            self._client: BaseLLMClient = OpenAIClient(**kwargs)
        elif _provider == "ollama":
            self._client = OllamaClient(**kwargs)
        else:
            raise ValueError(
                f"Unknown LLM provider '{_provider}'. Use 'openai' or 'ollama'."
            )

        self.provider = _provider
        logger.info("LLMClient initialised with provider='%s'.", _provider)

    def chat(self, messages: list[Message], **kwargs) -> str:
        """Send a chat request and return the raw text response."""
        return self._client.chat(messages, **kwargs)

    def chat_json(
        self, messages: list[Message], retries: int = 2, **kwargs
    ) -> dict | list:
        """
        Send a chat request and parse the response as JSON.

        Retries up to *retries* times on JSON parse failure.  Falls back to
        returning the raw text in a ``{"raw": "..."}`` dict if all attempts fail.

        Args:
            messages: List of message dicts.
            retries:  Number of retry attempts on parse failure.

        Returns:
            Parsed JSON dict or list.
        """
        for attempt in range(retries + 1):
            raw = self._client.chat(messages, **kwargs)

            try:
                # Strip markdown fences if the model added them
                cleaned = self._strip_fences(raw)
                return json.loads(cleaned)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "JSON parse failed (attempt %d/%d): %s", attempt + 1, retries + 1, exc
                )
                if attempt < retries:
                    # Ask the model to fix it
                    messages = messages + [
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                "Your response was not valid JSON. "
                                "Please return ONLY the JSON object, no markdown, no explanation."
                            ),
                        },
                    ]

        logger.error("All JSON parse attempts failed. Returning raw response.")
        return {"raw": raw, "error": "json_parse_failed"}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove ```json ... ``` or ``` ... ``` wrappers from LLM output."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # Drop first line (``` or ```json) and last line (```)
            inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            text = "\n".join(inner)
        return text.strip()
