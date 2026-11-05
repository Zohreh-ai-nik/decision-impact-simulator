"""Offline tests for the retrieval layer."""

from __future__ import annotations

import pytest

from decision_sim.intelligence.rag import (
    EmbeddingRetriever,
    KeywordRetriever,
    Retriever,
)
from decision_sim.schemas import Evidence, Source


def _ev(text: str, url: str = "file:///docs/a.md") -> Evidence:
    return Evidence(text=text, sources=[Source(url=url, title="a.md")])


# --- KeywordRetriever -------------------------------------------------------


def test_keyword_ranks_relevant_first():
    r = KeywordRetriever()
    r.add(
        [
            _ev("customers are angry about the car insurance price rise"),
            _ev("the weather was sunny all weekend"),
        ]
    )
    hits = r.search("insurance price rise", k=2)
    assert hits[0].text.startswith("customers are angry")


def test_keyword_k_limits_results():
    r = KeywordRetriever()
    r.add([_ev("price rise one"), _ev("price rise two"), _ev("price rise three")])
    assert len(r.search("price rise", k=2)) == 2


def test_keyword_drops_zero_score_docs():
    r = KeywordRetriever()
    r.add([_ev("totally unrelated content"), _ev("insurance premium increase")])
    hits = r.search("insurance premium", k=5)
    assert len(hits) == 1
    assert hits[0].text == "insurance premium increase"


def test_keyword_empty_store_returns_empty():
    assert KeywordRetriever().search("anything") == []


def test_keyword_empty_query_returns_empty():
    r = KeywordRetriever()
    r.add([_ev("something")])
    assert r.search("   ") == []


def test_keyword_rejects_nonpositive_k():
    r = KeywordRetriever()
    r.add([_ev("x")])
    with pytest.raises(ValueError):
        r.search("x", k=0)


def test_keyword_preserves_provenance():
    r = KeywordRetriever()
    r.add([_ev("insurance price", url="file:///docs/report.md")])
    hit = r.search("insurance price")[0]
    assert hit.sources[0].url == "file:///docs/report.md"


def test_keyword_satisfies_protocol():
    assert isinstance(KeywordRetriever(), Retriever)


# --- EmbeddingRetriever (fake embedder) -------------------------------------


def _fake_embed(texts):
    """Tiny deterministic embedder: counts of two marker words -> 2-D vectors."""
    vectors = []
    for t in texts:
        low = t.lower()
        vectors.append([float(low.count("price")), float(low.count("weather"))])
    return vectors


def test_embedding_ranks_by_cosine():
    r = EmbeddingRetriever(_fake_embed)
    r.add([_ev("price price price"), _ev("weather weather")])
    hits = r.search("price increase", k=2)
    assert hits[0].text == "price price price"


def test_embedding_k_limits_results():
    r = EmbeddingRetriever(_fake_embed)
    r.add([_ev("price a"), _ev("price b"), _ev("price c")])
    assert len(r.search("price", k=2)) == 2


def test_embedding_empty_store_returns_empty():
    assert EmbeddingRetriever(_fake_embed).search("price") == []


def test_embedding_add_empty_is_noop():
    r = EmbeddingRetriever(_fake_embed)
    r.add([])
    assert r.search("price") == []


def test_embedding_rejects_nonpositive_k():
    r = EmbeddingRetriever(_fake_embed)
    r.add([_ev("price")])
    with pytest.raises(ValueError):
        r.search("price", k=0)


def test_embedding_validates_vector_count():
    def bad_embed(texts):
        return [[1.0, 0.0]]  # wrong count when given 2 texts

    r = EmbeddingRetriever(bad_embed)
    with pytest.raises(ValueError, match="vectors"):
        r.add([_ev("one"), _ev("two")])


def test_embedding_preserves_provenance():
    r = EmbeddingRetriever(_fake_embed)
    r.add([_ev("price", url="file:///docs/x.md")])
    hit = r.search("price")[0]
    assert hit.sources[0].url == "file:///docs/x.md"


def test_embedding_satisfies_protocol():
    assert isinstance(EmbeddingRetriever(_fake_embed), Retriever)
