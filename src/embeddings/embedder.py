"""Local embeddings via Ollama's HTTP API.

Uses stdlib urllib so no extra dependency is required for this layer.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Sequence

from config import EMBED_MODEL, OLLAMA_HOST


class EmbeddingError(RuntimeError):
    pass


def _post(path: str, payload: dict, timeout: int = 60) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=f"{OLLAMA_HOST}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise EmbeddingError(
            f"Failed to reach Ollama at {OLLAMA_HOST}{path}: {exc}. "
            "Is `ollama serve` running and the model pulled?"
        ) from exc


def embed_text(text: str, model: str = EMBED_MODEL) -> list[float]:
    """Embed a single text and return its vector."""
    if not text.strip():
        raise EmbeddingError("Cannot embed empty text")
    data = _post("/api/embeddings", {"model": model, "prompt": text})
    vector = data.get("embedding")
    if not isinstance(vector, list) or not vector:
        raise EmbeddingError(f"Unexpected response from Ollama: {data}")
    return vector


def embed_texts(texts: Sequence[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Embed many texts. Ollama embeddings endpoint is single-input,
    so we issue one HTTP call per text. Acceptable for ~hundreds of chunks.
    """
    return [embed_text(t, model=model) for t in texts]
