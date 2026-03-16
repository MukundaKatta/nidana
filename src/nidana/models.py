"""Model adapters for Claude, OpenAI, and Ollama."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel, Field


class ModelResponse(BaseModel):
    """Raw response from a model adapter."""

    model_id: str
    raw_text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


class ModelAdapter(ABC):
    """Base class for LLM adapters."""

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> ModelResponse: ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} model={self.model_id}>"


# ---------------------------------------------------------------------------
# Claude (Anthropic) adapter
# ---------------------------------------------------------------------------

class ClaudeAdapter(ModelAdapter):
    """Adapter for Anthropic Claude models via the Anthropic SDK."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._max_tokens = max_tokens

    @property
    def model_id(self) -> str:
        return self._model

    def generate(self, system_prompt: str, user_prompt: str) -> ModelResponse:
        import time

        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        t0 = time.perf_counter()
        message = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        latency = (time.perf_counter() - t0) * 1000
        return ModelResponse(
            model_id=self._model,
            raw_text=message.content[0].text,
            prompt_tokens=message.usage.input_tokens,
            completion_tokens=message.usage.output_tokens,
            latency_ms=latency,
        )


# ---------------------------------------------------------------------------
# OpenAI adapter
# ---------------------------------------------------------------------------

class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI chat-completion models."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._max_tokens = max_tokens

    @property
    def model_id(self) -> str:
        return self._model

    def generate(self, system_prompt: str, user_prompt: str) -> ModelResponse:
        import time

        import openai

        client = openai.OpenAI(api_key=self._api_key)
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        latency = (time.perf_counter() - t0) * 1000
        choice = response.choices[0]
        usage = response.usage
        return ModelResponse(
            model_id=self._model,
            raw_text=choice.message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency,
        )


# ---------------------------------------------------------------------------
# Ollama (local) adapter
# ---------------------------------------------------------------------------

class OllamaAdapter(ModelAdapter):
    """Adapter for locally-hosted models via Ollama's REST API."""

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")

    @property
    def model_id(self) -> str:
        return f"ollama/{self._model}"

    def generate(self, system_prompt: str, user_prompt: str) -> ModelResponse:
        import time

        import httpx

        t0 = time.perf_counter()
        resp = httpx.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=300.0,
        )
        resp.raise_for_status()
        data = resp.json()
        latency = (time.perf_counter() - t0) * 1000
        return ModelResponse(
            model_id=f"ollama/{self._model}",
            raw_text=data.get("message", {}).get("content", ""),
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            latency_ms=latency,
        )


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    """Declarative model configuration for benchmark runs."""

    provider: str = Field(description="claude | openai | ollama")
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    def to_adapter(self) -> ModelAdapter:
        if self.provider == "claude":
            return ClaudeAdapter(model=self.model, api_key=self.api_key)
        elif self.provider == "openai":
            return OpenAIAdapter(model=self.model, api_key=self.api_key)
        elif self.provider == "ollama":
            return OllamaAdapter(
                model=self.model,
                base_url=self.base_url or "http://localhost:11434",
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
