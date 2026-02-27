"""
LLM Provider abstraction for the Enneagram testing framework.

Supports:
- Ollama (local, default)
- Anthropic (Claude models)
- OpenAI (GPT models)
- OpenRouter (any model via OpenRouter)

Usage:
    provider = create_provider("ollama", model="mistral")
    provider = create_provider("anthropic", model="claude-sonnet-4-20250514", api_key="sk-...")
    provider = create_provider("openai", model="gpt-4o", api_key="sk-...")
    provider = create_provider("openrouter", model="anthropic/claude-sonnet-4", api_key="sk-...")

    response_text = provider.generate("Your prompt here", temperature=0.7)
"""

import os
from abc import ABC, abstractmethod

import requests


class LLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def generate(self, prompt: str, temperature: float = None, no_context: bool = False) -> str:
        """Send a prompt and return the response text.

        Args:
            no_context: If True, explicitly clear context between requests.
                        Only meaningful for Ollama (cloud APIs are already stateless).
        """
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model!r})"


class OllamaProvider(LLMProvider):
    """Local Ollama instance."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        super().__init__(model)
        self.url = f"{base_url}/api/generate"

    def generate(self, prompt: str, temperature: float = None, no_context: bool = False) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        if temperature is not None:
            payload["temperature"] = temperature
        if no_context:
            payload["context"] = []

        resp = requests.post(self.url, json=payload, timeout=600)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(self, model: str = None, api_key: str = None):
        super().__init__(model or self.DEFAULT_MODEL)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass --api-key."
            )

    def generate(self, prompt: str, temperature: float = None, no_context: bool = False) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            body["temperature"] = temperature

        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=body,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return (data.get("content", [{}])[0].get("text", "")).strip()


class OpenAIProvider(LLMProvider):
    """OpenAI API (GPT models)."""

    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, model: str = None, api_key: str = None):
        super().__init__(model or self.DEFAULT_MODEL)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass --api-key."
            )

    def generate(self, prompt: str, temperature: float = None, no_context: bool = False) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64,
        }
        if temperature is not None:
            body["temperature"] = temperature

        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=body,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return (data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()


class OpenRouterProvider(LLMProvider):
    """OpenRouter API (any model)."""

    DEFAULT_MODEL = "anthropic/claude-sonnet-4"

    def __init__(self, model: str = None, api_key: str = None):
        super().__init__(model or self.DEFAULT_MODEL)
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var or pass --api-key."
            )

    def generate(self, prompt: str, temperature: float = None, no_context: bool = False) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64,
        }
        if temperature is not None:
            body["temperature"] = temperature

        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=body,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return (data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()


PROVIDERS = {
    "ollama": OllamaProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "openrouter": OpenRouterProvider,
}


def create_provider(provider_name: str, model: str = None, api_key: str = None, **kwargs) -> LLMProvider:
    """Factory function to create an LLM provider.

    Args:
        provider_name: One of 'ollama', 'anthropic', 'openai', 'openrouter'
        model: Model name (uses provider default if not specified)
        api_key: API key (uses env var if not specified, not needed for ollama)
        **kwargs: Additional provider-specific arguments
    """
    cls = PROVIDERS.get(provider_name)
    if cls is None:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(f"Unknown provider: {provider_name!r}. Available: {available}")

    if provider_name == "ollama":
        return cls(model=model or "mistral", **kwargs)
    else:
        return cls(model=model, api_key=api_key, **kwargs)
