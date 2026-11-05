"""Tests for the Qdrant-backed retriever, using in-memory Qdrant (no server)."""

from __future__ import annotations

import pytest

pytest.importorskip("qdrant_client")

from decision_sim.intelligence.rag import Retriever  # noqa: E402
from decision_sim.intelligence.vectorstore import QdrantRetriever  # noqa: E402
from decision_sim.schemas import Evidence, Source  # noqa: E402


def _ev(text: str, url: str = "file:///docs/a.md") -> Evidence:
    return Evidence(text=text, sources=[Source(url=url, title="a.md")])


def _fake_embed(texts):
    """Deterministic 2-D embedder: counts of two marker words."""
    out = []
    for t in texts:
        low = t.lower()
        out.append([float(low.count("price")), float(low.count("weather"))])
    return out


def _store() -> QdrantRetriever:
    # location=":memory:" -> no server, no network.
    return QdrantRetriever(_fake_embed, location=":memory:")


def test_add_then_search_ranks_relevant_first():
    r = _store()
    r.add([_ev("price price price"), _ev("weather weather")])
    hits = r.search("price increase", k=2)
    assert hits[0].text == "price price price"


def test_search_k_limits_results():
    r = _store()
    r.add([_ev("price a"), _ev("price b"), _ev("price c")])
    assert len(r.search("price", k=2)) == 2


def test_search_empty_store_returns_empty():
    assert _store().search("price") == []


def test_add_empty_is_noop():
    r = _store()
    r.add([])
    assert r.search("price") == []


def test_search_rejects_nonpositive_k():
    r = _store()
    r.add([_ev("price")])
    with pytest.raises(ValueError):
        r.search("price", k=0)


def test_add_validates_vector_count():
    def bad_embed(texts):
        return [[1.0, 0.0]]

    r = QdrantRetriever(bad_embed, location=":memory:")
    with pytest.raises(ValueError, match="vectors"):
        r.add([_ev("one"), _ev("two")])


def test_provenance_round_trips_through_payload():
    r = _store()
    r.add([_ev("price", url="file:///docs/report.md")])
    hit = r.search("price")[0]
    assert hit.sources[0].url == "file:///docs/report.md"
    assert hit.sources[0].title == "a.md"


def test_satisfies_retriever_protocol():
    assert isinstance(_store(), Retriever)
