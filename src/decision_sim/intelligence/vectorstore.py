"""Qdrant-backed :class:`~decision_sim.intelligence.rag.Retriever`.

Qdrant runs **in-memory** by default (``QdrantClient(location=":memory:")``), so
tests and the offline demo need no running server; point ``location``/``url`` at a
Docker Qdrant only for the full demo. An embedding function (see
:mod:`decision_sim.intelligence.embeddings`) is injected, keeping this class
independent of any particular model.

Each :class:`Evidence` is stored as a point whose payload carries the full chunk
(text + sources), so a search round-trips provenance intact. Retrieved text is
data, never instructions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from decision_sim.intelligence.rag import EmbeddingFn
from decision_sim.schemas import Evidence

__all__ = ["QdrantRetriever"]


class QdrantRetriever:
    """Vector retrieval over Qdrant, scored by cosine similarity."""

    def __init__(
        self,
        embed: EmbeddingFn,
        *,
        client: Any | None = None,
        collection: str = "evidence",
        location: str = ":memory:",
    ) -> None:
        self._embed = embed
        self._collection = collection
        self._next_id = 0
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                'qdrant-client is not installed. Install the vectorstore extra: '
                'pip install -e ".[vectorstore]".'
            ) from exc
        self._models = models
        self._client = client if client is not None else QdrantClient(location=location)

    def _ensure_collection(self, size: int) -> None:
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            self._collection,
            vectors_config=self._models.VectorParams(
                size=size, distance=self._models.Distance.COSINE
            ),
        )

    def add(self, evidence: Sequence[Evidence]) -> None:
        evidence = list(evidence)
        if not evidence:
            return
        vectors = self._embed([ev.text for ev in evidence])
        if len(vectors) != len(evidence):
            raise ValueError(
                f"Embedder returned {len(vectors)} vectors for {len(evidence)} texts."
            )
        self._ensure_collection(len(vectors[0]))
        points = []
        for ev, vector in zip(evidence, vectors, strict=True):
            points.append(
                self._models.PointStruct(
                    id=self._next_id,
                    vector=vector,
                    payload=ev.model_dump(mode="json"),
                )
            )
            self._next_id += 1
        self._client.upsert(self._collection, points=points)

    def search(self, query: str, *, k: int = 5) -> list[Evidence]:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}.")
        if not self._client.collection_exists(self._collection):
            return []
        query_vec = self._embed([query])[0]
        hits = self._client.query_points(
            self._collection, query=query_vec, limit=k
        ).points
        return [Evidence.model_validate(hit.payload) for hit in hits]
