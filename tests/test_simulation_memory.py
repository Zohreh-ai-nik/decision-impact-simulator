"""Offline tests for per-agent simulation memory."""

from __future__ import annotations

import pytest

from decision_sim.simulation.memory import InMemoryStore, MemoryStore


def test_in_memory_store_satisfies_protocol():
    assert isinstance(InMemoryStore(), MemoryStore)


def test_recall_returns_recent_entries_in_order():
    store = InMemoryStore()
    for i in range(5):
        store.add("A", f"note {i}")
    assert store.recall("A", limit=3) == ["note 2", "note 3", "note 4"]


def test_recall_limit_caps_results():
    store = InMemoryStore()
    store.add("A", "one")
    store.add("A", "two")
    assert store.recall("A", limit=1) == ["two"]


def test_memory_is_isolated_per_agent():
    store = InMemoryStore()
    store.add("A", "a-note")
    store.add("B", "b-note")
    assert store.recall("A") == ["a-note"]
    assert store.recall("B") == ["b-note"]


def test_unknown_agent_recalls_nothing():
    assert InMemoryStore().recall("ghost") == []


def test_blank_notes_are_ignored():
    store = InMemoryStore()
    store.add("A", "   ")
    store.add("A", "real")
    assert store.recall("A") == ["real"]


def test_recall_rejects_non_positive_limit():
    store = InMemoryStore()
    with pytest.raises(ValueError):
        store.recall("A", limit=0)


def test_mem0_store_requires_client_or_credentials(monkeypatch):
    pytest.importorskip("mem0")
    monkeypatch.delenv("MEM0_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from decision_sim.simulation.memory import Mem0Store

    with pytest.raises(ValueError):
        Mem0Store()


def test_mem0_store_uses_injected_client():
    from decision_sim.simulation.memory import Mem0Store

    class _FakeMem0:
        def __init__(self) -> None:
            self.added: list[tuple[str, str]] = []

        def add(self, text, *, user_id):
            self.added.append((user_id, text))

        def search(self, query, *, user_id, limit):
            return {"results": [{"memory": f"{user_id}:{query}"}]}

    client = _FakeMem0()
    store = Mem0Store(client=client)
    store.add("A", "hello")
    assert client.added == [("A", "hello")]
    assert store.recall("A", "goal", limit=1) == ["A:goal"]
