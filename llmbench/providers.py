"""Provider abstraction for local and cloud LLM APIs."""

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class ModelResponse:
    """Unified response from any provider."""
    content: str
    reasoning_content: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    output_tokens: int = 0  # completion_tokens - reasoning_tokens
    elapsed: float = 0.0
    ttft: float = 0.0  # time to first token (approx for non-streaming)
    tok_per_sec: float = 0.0
    error: Optional[str] = None


@dataclass
class ModelConfig:
    """A model entry from config.yaml."""
    name: str
    provider: str
    model_id: str
    description: str
    type: str  # "local" or "cloud"
    port: Optional[int] = None


@dataclass
class ProviderConfig:
    """A provider entry from config.yaml."""
    name: str
    base_url: str
    api_key_env: Optional[str] = None
    api_key: Optional[str] = None


def resolve_api_key(provider: ProviderConfig) -> Optional[str]:
    """Resolve API key from env var or direct value."""
    if provider.api_key:
        return provider.api_key
    if provider.api_key_env:
        return os.environ.get(provider.api_key_env)
    return None


def call_model(
    provider: ProviderConfig,
    model: ModelConfig,
    messages: List[Dict[str, str]],
    max_tokens: int = 4096,
    temperature: float = 0.3,
    timeout: float = 120,
) -> ModelResponse:
    """Call a model via its provider's OpenAI-compatible API."""

    # Build base URL
    if model.port:
        base_url = provider.base_url.format(port=model.port)
    else:
        base_url = provider.base_url

    url = f"{base_url}/chat/completions"

    # Build headers
    headers = {"Content-Type": "application/json"}
    api_key = resolve_api_key(provider)
    if api_key and api_key != "not-needed":
        headers["Authorization"] = f"Bearer {api_key}"

    # Build payload
    payload = {
        "model": model.model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    # Make request
    start = time.time()
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
        elapsed = time.time() - start
    except httpx.TimeoutException:
        return ModelResponse(content="", error=f"Timeout after {timeout}s", elapsed=timeout)
    except httpx.HTTPStatusError as e:
        return ModelResponse(
            content="",
            error=f"HTTP {e.response.status_code}: {e.response.text[:500]}",
            elapsed=time.time() - start,
        )
    except Exception as e:
        return ModelResponse(content="", error=str(e), elapsed=time.time() - start)

    # Parse response
    try:
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content", "") or ""
        reasoning = msg.get("reasoning_content", "") or ""

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        details = usage.get("completion_tokens_details", {})
        reasoning_tokens = details.get("reasoning_tokens", 0)
        output_tokens = max(0, completion_tokens - reasoning_tokens) if reasoning_tokens else completion_tokens

        tps = output_tokens / elapsed if elapsed > 0 and output_tokens > 0 else 0

        return ModelResponse(
            content=content,
            reasoning_content=reasoning,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            output_tokens=output_tokens,
            elapsed=elapsed,
            ttft=elapsed * 0.3,  # approximate for non-streaming
            tok_per_sec=tps,
        )
    except Exception as e:
        return ModelResponse(content="", error=f"Parse error: {e}", elapsed=elapsed)


def check_health(provider: ProviderConfig, model: ModelConfig) -> bool:
    """Check if a model endpoint is healthy (local models only)."""
    if model.type != "local" or not model.port:
        return True  # cloud models assumed healthy
    base_url = provider.base_url.format(port=model.port)
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{base_url.replace('/v1', '')}/health")
            return resp.status_code == 200
    except Exception:
        return False
