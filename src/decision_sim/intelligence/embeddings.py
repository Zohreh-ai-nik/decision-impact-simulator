"""Embedding functions implementing the :class:`~decision_sim.intelligence.rag.EmbeddingFn`
protocol.

Two embedders are available, both optional:

* :class:`FastEmbedEmbedder` — a small local ONNX model via ``fastembed``. No API
  key, no cost, no network after the one-time model download. The default.
* :class:`LiteLLMEmbedder` — an API embedder via the core ``litellm`` dependency,
  for when a hosted model is preferred.

Neither is imported at module load; the heavy/optional dependency is imported only
when the embedder is constructed, so the core install and the test suite stay
dependency-light.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

__all__ = [
    "DEFAULT_FASTEMBED_MODEL",
    "FastEmbedEmbedder",
    "LiteLLMEmbedder",
]

#: Small, fast, widely-used default local model.
DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedEmbedder:
    """Local embeddings via ``fastembed`` (lazy import, injectable model)."""

    def __init__(
        self, *, model_name: str = DEFAULT_FASTEMBED_MODEL, model: Any | None = None
    ) -> None:
        if model is not None:
            self._model = model
            return
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                'fastembed is not installed. Install the embeddings extra: '
                'pip install -e ".[embeddings]".'
            ) from exc
        self._model = TextEmbedding(model_name=model_name)

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        # fastembed yields numpy arrays; convert to plain lists for the protocol.
        return [list(map(float, vector)) for vector in self._model.embed(list(texts))]


class LiteLLMEmbedder:
    """API embeddings via ``litellm`` (model from arg or ``EMBED_MODEL`` env)."""

    def __init__(self, *, model: str | None = None) -> None:
        resolved = model or os.environ.get("EMBED_MODEL")
        if not resolved:
            raise ValueError(
                "No embedding model: pass model= or set EMBED_MODEL in the environment."
            )
        self._model = resolved

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        import litellm

        response = litellm.embedding(model=self._model, input=list(texts))
        return [list(map(float, item["embedding"])) for item in response["data"]]
