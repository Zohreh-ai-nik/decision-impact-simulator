"""Offline tests for the search providers and the file:// source rule."""

from __future__ import annotations

import datetime

import pytest
from pydantic import ValidationError

from decision_sim.intelligence.web_search import (
    CachedSearchProvider,
    FakeSearchProvider,
    SearchProvider,
    SearchResult,
)
from decision_sim.schemas import Source


def _result(title: str, url: str = "https://example.com/a") -> SearchResult:
    return SearchResult(title=title, url=url, snippet="s", source="fake")


# --- SearchResult validation ------------------------------------------------


def test_search_result_accepts_file_url():
    r = SearchResult(title="doc", url="file:///tmp/report.pdf")
    assert r.url.startswith("file://")


def test_search_result_rejects_bare_url():
    with pytest.raises(ValidationError):
        SearchResult(title="x", url="example.com")


def test_search_result_parses_published_date():
    r = SearchResult(title="x", url="https://e.com", published=datetime.date(2025, 1, 2))
    assert r.published == datetime.date(2025, 1, 2)


# --- FakeSearchProvider -----------------------------------------------------


def test_fake_returns_canned_results():
    provider = FakeSearchProvider({"q": [_result("one"), _result("two")]})
    hits = provider.search("q")
    assert [h.title for h in hits] == ["one", "two"]


def test_fake_unknown_query_is_empty():
    provider = FakeSearchProvider({"q": [_result("one")]})
    assert provider.search("other") == []


def test_fake_respects_max_results():
    provider = FakeSearchProvider({"q": [_result("one"), _result("two"), _result("three")]})
    assert [h.title for h in provider.search("q", max_results=2)] == ["one", "two"]


def test_fake_rejects_nonpositive_max_results():
    provider = FakeSearchProvider({"q": [_result("one")]})
    with pytest.raises(ValueError):
        provider.search("q", max_results=0)


def test_fake_returns_copies_not_fixtures():
    stored = _result("one")
    provider = FakeSearchProvider({"q": [stored]})
    returned = provider.search("q")[0]
    returned.title = "mutated"
    assert stored.title == "one"


def test_fake_satisfies_protocol():
    assert isinstance(FakeSearchProvider({}), SearchProvider)


# --- CachedSearchProvider ---------------------------------------------------


class _CountingProvider:
    """Wraps a Fake provider and counts how often search() is called."""

    def __init__(self, inner: FakeSearchProvider) -> None:
        self._inner = inner
        self.calls = 0

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        self.calls += 1
        return self._inner.search(query, max_results=max_results)


def test_cached_first_call_delegates_and_writes(tmp_path):
    spy = _CountingProvider(FakeSearchProvider({"q": [_result("one")]}))
    cached = CachedSearchProvider(spy, cache_dir=tmp_path)
    hits = cached.search("q")
    assert [h.title for h in hits] == ["one"]
    assert spy.calls == 1
    assert (tmp_path / "cache.json").exists()


def test_cached_second_call_hits_cache(tmp_path):
    spy = _CountingProvider(FakeSearchProvider({"q": [_result("one")]}))
    cached = CachedSearchProvider(spy, cache_dir=tmp_path)
    cached.search("q")
    cached.search("q")
    assert spy.calls == 1  # second call served from cache


def test_cached_persists_across_instances(tmp_path):
    spy = _CountingProvider(FakeSearchProvider({"q": [_result("one")]}))
    CachedSearchProvider(spy, cache_dir=tmp_path).search("q")
    spy2 = _CountingProvider(FakeSearchProvider({"q": [_result("one")]}))
    hits = CachedSearchProvider(spy2, cache_dir=tmp_path).search("q")
    assert [h.title for h in hits] == ["one"]
    assert spy2.calls == 0  # loaded from disk, inner never called


def test_cached_distinguishes_max_results(tmp_path):
    spy = _CountingProvider(FakeSearchProvider({"q": [_result("one"), _result("two")]}))
    cached = CachedSearchProvider(spy, cache_dir=tmp_path)
    cached.search("q", max_results=1)
    cached.search("q", max_results=2)
    assert spy.calls == 2  # different max_results => different cache key


# --- Source accepts file:// -------------------------------------------------


def test_source_accepts_file_url():
    s = Source(url="file:///docs/policy.pdf", title="Policy")
    assert s.url.startswith("file://")


def test_source_still_rejects_junk():
    with pytest.raises(ValidationError):
        Source(url="not-a-url", title="x")
