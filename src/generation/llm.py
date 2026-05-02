"""Ollama generate API client (HTTP, stdlib-only)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Iterator, Optional

from config import LLM_MAX_TOKENS, LLM_MODEL, LLM_TEMPERATURE, OLLAMA_HOST


class LLMError(RuntimeError):
    pass


def _request(payload: dict, stream: bool, timeout: int = 120):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=f"{OLLAMA_HOST}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.URLError as exc:
        raise LLMError(
            f"Failed to reach Ollama at {OLLAMA_HOST}: {exc}. "
            "Is `ollama serve` running and the model pulled?"
        ) from exc


def generate(
    prompt: str,
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
) -> str:
    """Synchronously generate a complete answer."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    with _request(payload, stream=False) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    text = data.get("response")
    if not isinstance(text, str):
        raise LLMError(f"Unexpected response from Ollama: {data}")
    return text.strip()


def generate_stream(
    prompt: str,
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
) -> Iterator[str]:
    """Yield response tokens as they arrive (newline-delimited JSON stream)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    with _request(payload, stream=True) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            piece = event.get("response")
            if piece:
                yield piece
            if event.get("done"):
                break
