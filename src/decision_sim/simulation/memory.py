"""Per-agent memory for the simulation — offline-first, with an optional Mem0 backend.

Each agent remembers its *own* prior turns (what it said and how it felt), so its
stance can evolve coherently across rounds and stages rather than only reacting to
the sliding window of others' recent posts.

Two backends share one :class:`MemoryStore` protocol:

* :class:`InMemoryStore` needs no dependencies and no network. It keeps a per-agent
  list and recalls the most recent entries. Deterministic, so it is the default for
  tests and the offline demo. Memory lives only for the duration of a run.
* :class:`Mem0Store` persists to Mem0 via the optional ``memory`` extra
  (``pip install -e ".[memory]"``). Lazy import; semantic recall; API key read from
  the environment (never logged). Only needed for cross-run persistence / the full demo.

Stored text is an agent's own note, never external instructions.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "InMemoryStore",
    "Mem0Store",
    "MemoryStore",
]


@runtime_checkable
class MemoryStore(Protocol):
    """Stores and recalls short memory notes, isolated per agent."""

    def add(self, agent: str, text: str) -> None: ...

    def recall(self, agent: str, query: str = "", *, limit: int = 5) -> list[str]: ...


class InMemoryStore:
    """In-process, per-agent memory recalled by recency (no dependencies).

    ``query`` is ignored (pure recency); it is accepted to match the protocol and
    the semantic :class:`Mem0Store`. An unknown agent recalls nothing.
    """

    def __init__(self) -> None:
        self._notes: dict[str, list[str]] = defaultdict(list)

    def add(self, agent: str, text: str) -> None:
        text = text.strip()
        if text:
            self._notes[agent].append(text)

    def recall(self, agent: str, query: str = "", *, limit: int = 5) -> list[str]:
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}.")
        return self._notes.get(agent, [])[-limit:]


class Mem0Store:
    """Mem0-backed memory store (optional ``memory`` extra).

    The client comes from the ``client`` argument or is constructed lazily from
    :class:`mem0.Memory`. API credentials are read by Mem0 from the environment
    (e.g. ``MEM0_API_KEY`` / ``OPENAI_API_KEY``) and are never logged here.
    """

    def __init__(self, *, client: Any | None = None) -> None:
        if client is not None:
            self._client = client
            return
        # Surface a clear error if the optional dependency is missing.
        try:
            from mem0 import Memory
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                'mem0 is not installed. Install the memory extra: pip install -e ".[memory]".'
            ) from exc
        if not os.environ.get("MEM0_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                "No Mem0 credentials: pass client= or set MEM0_API_KEY / OPENAI_API_KEY."
            )
        self._client = Memory()

    def add(self, agent: str, text: str) -> None:
        text = text.strip()
        if text:
            self._client.add(text, user_id=agent)

    def recall(self, agent: str, query: str = "", *, limit: int = 5) -> list[str]:
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}.")
        result = self._client.search(query, user_id=agent, limit=limit)
        # Mem0 returns {"results": [{"memory": "..."}, ...]} or a bare list.
        items = result.get("results", result) if isinstance(result, dict) else result
        memories: list[str] = []
        for item in items:
            memory = item.get("memory") if isinstance(item, dict) else item
            if isinstance(memory, str) and memory:
                memories.append(memory)
        return memories[:limit]
