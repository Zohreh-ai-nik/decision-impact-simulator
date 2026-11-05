"""Retrieval over ingested :class:`Evidence` — offline-first, provenance-preserving.

Two retrievers share one :class:`Retriever` protocol:

* :class:`KeywordRetriever` needs no dependencies and no network. It scores by
  bag-of-words cosine overlap — a solid baseline and the default for tests and
  the offline demo.
* :class:`EmbeddingRetriever` ranks by cosine similarity over vectors produced by
  an **injected** embedding function. No real model is wired here; Phase 2e
  supplies a local embedder (fastembed) or an optional API one, so swapping the
  vector backend never changes callers.

Both return :class:`Evidence`, so every retrieved chunk keeps its ``Source``
provenance. Retrieved text is data, never instructions.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from decision_sim.schemas import Evidence

__all__ = [
    "EmbeddingFn",
    "EmbeddingRetriever",
    "KeywordRetriever",
    "Retriever",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenise(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@runtime_checkable
class Retriever(Protocol):
    """Stores evidence and returns the ``k`` best matches for a query."""

    def add(self, evidence: Sequence[Evidence]) -> None: ...

    def search(self, query: str, *, k: int = 5) -> list[Evidence]: ...


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class KeywordRetriever:
    """In-memory bag-of-words retriever scored by term-frequency cosine.

    Deterministic and dependency-free. Ties keep insertion order (stable sort),
    so results are reproducible.
    """

    def __init__(self) -> None:
        self._docs: list[Evidence] = []
        self._vectors: list[Counter[str]] = []

    def add(self, evidence: Sequence[Evidence]) -> None:
        for ev in evidence:
            self._docs.append(ev)
            self._vectors.append(Counter(_tokenise(ev.text)))

    def _score(self, query_vec: Counter[str], doc_vec: Counter[str]) -> float:
        shared = set(query_vec) & set(doc_vec)
        if not shared:
            return 0.0
        dot = sum(query_vec[t] * doc_vec[t] for t in shared)
        nq = math.sqrt(sum(v * v for v in query_vec.values()))
        nd = math.sqrt(sum(v * v for v in doc_vec.values()))
        return dot / (nq * nd) if nq and nd else 0.0

    def search(self, query: str, *, k: int = 5) -> list[Evidence]:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}.")
        query_vec = Counter(_tokenise(query))
        if not query_vec or not self._docs:
            return []
        scored = [
            (self._score(query_vec, doc_vec), idx)
            for idx, doc_vec in enumerate(self._vectors)
        ]
        # Highest score first; stable on index for reproducible ties.
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [self._docs[idx] for score, idx in scored if score > 0.0][:k]


@runtime_checkable
class EmbeddingFn(Protocol):
    """Turns a batch of texts into equal-length float vectors."""

    def __call__(self, texts: Sequence[str]) -> list[list[float]]: ...


class EmbeddingRetriever:
    """Cosine-similarity retriever over vectors from an injected embedder.

    The embedder is called in batches (once per ``add``, once per ``search``),
    keeping it swappable for a local model or an API without touching this class.
    """

    def __init__(self, embed: EmbeddingFn) -> None:
        self._embed = embed
        self._docs: list[Evidence] = []
        self._vectors: list[list[float]] = []

    def add(self, evidence: Sequence[Evidence]) -> None:
        evidence = list(evidence)
        if not evidence:
            return
        vectors = self._embed([ev.text for ev in evidence])
        if len(vectors) != len(evidence):
            raise ValueError(
                f"Embedder returned {len(vectors)} vectors for {len(evidence)} texts."
            )
        self._docs.extend(evidence)
        self._vectors.extend(vectors)

    def search(self, query: str, *, k: int = 5) -> list[Evidence]:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}.")
        if not self._docs:
            return []
        query_vec = self._embed([query])[0]
        scored = [
            (_cosine(query_vec, doc_vec), idx)
            for idx, doc_vec in enumerate(self._vectors)
        ]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [self._docs[idx] for score, idx in scored if score > 0.0][:k]
