"""Search providers: deterministic offline sources of real facts.

A :class:`SearchProvider` turns a query into a list of :class:`SearchResult`.
Two providers live here and never touch the network:

* :class:`FakeSearchProvider` returns fixed, canned results — used by tests and
  for an offline demo.
* :class:`CachedSearchProvider` wraps any provider and persists results to a
  JSON file, so a repeated query is answered from disk without calling the inner
  provider again.

The real network provider (Tavily) arrives in Phase 2b and will implement the
same :class:`SearchProvider` protocol, so the rest of the system never changes.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "CachedSearchProvider",
    "FakeSearchProvider",
    "SearchProvider",
    "SearchResult",
    "DEFAULT_CACHE_DIR",
]

#: Default cache location. ``data/`` is git-ignored, so cached results never ship.
DEFAULT_CACHE_DIR = Path("data/cache/search")


class SearchResult(BaseModel):
    """A single search hit with enough provenance to cite it later."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    url: str = Field(min_length=1, description="Link to the result.")
    snippet: str = Field(default="", description="Short extract or summary.")
    published: datetime.date | None = Field(
        default=None, description="Publication date, if the provider reports one."
    )
    source: str = Field(default="", description="Name of the provider that returned it.")

    @field_validator("url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        # Mirrors schemas.Source: web (http[s]) or local document (file://).
        if not value.startswith(("http://", "https://", "file://")):
            raise ValueError(
                f"url must start with http://, https:// or file://, got {value!r}."
            )
        return value


@runtime_checkable
class SearchProvider(Protocol):
    """Anything that can answer a query with ranked results."""

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]: ...


class FakeSearchProvider:
    """Deterministic provider backed by an in-memory table of canned results.

    Results are keyed by exact query string. Unknown queries return an empty
    list. Each call returns at most ``max_results`` results, copied so callers
    cannot mutate the stored fixtures.
    """

    name = "fake"

    def __init__(self, results_by_query: Mapping[str, Sequence[SearchResult]]) -> None:
        self._results: dict[str, list[SearchResult]] = {
            query: list(results) for query, results in results_by_query.items()
        }

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        if max_results < 1:
            raise ValueError(f"max_results must be >= 1, got {max_results}.")
        hits = self._results.get(query, [])
        return [hit.model_copy(deep=True) for hit in hits[:max_results]]


class CachedSearchProvider:
    """Wrap a provider and persist its results to a JSON file on disk.

    The cache key is ``(query, max_results)``. On a cache hit the inner provider
    is **not** called; on a miss the inner provider is queried and the result is
    written back. This keeps demos and tests offline after a single warm-up.
    """

    def __init__(
        self,
        inner: SearchProvider,
        *,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
    ) -> None:
        self._inner = inner
        self._cache_dir = Path(cache_dir)
        self._cache_file = self._cache_dir / "cache.json"
        self._cache: dict[str, list[SearchResult]] = self._load()

    @staticmethod
    def _key(query: str, max_results: int) -> str:
        return f"{max_results}::{query}"

    def _load(self) -> dict[str, list[SearchResult]]:
        if not self._cache_file.exists():
            return {}
        raw = json.loads(self._cache_file.read_text(encoding="utf-8"))
        return {
            key: [SearchResult.model_validate(item) for item in items]
            for key, items in raw.items()
        }

    def _save(self) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        serialisable = {
            key: [item.model_dump(mode="json") for item in items]
            for key, items in self._cache.items()
        }
        self._cache_file.write_text(
            json.dumps(serialisable, indent=2, sort_keys=True), encoding="utf-8"
        )

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        key = self._key(query, max_results)
        if key in self._cache:
            return [hit.model_copy(deep=True) for hit in self._cache[key]]
        results = list(self._inner.search(query, max_results=max_results))
        self._cache[key] = results
        self._save()
        return [hit.model_copy(deep=True) for hit in results]
