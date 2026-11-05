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
    TavilySearchProvider,
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


# --- TavilySearchProvider (offline, injected client) ------------------------


class _FakeTavilyClient:
    """Stub Tavily client returning a canned raw response (no network)."""

    def __init__(self, response: dict) -> None:
        self._response = response
        self.last_kwargs: dict | None = None

    def search(self, **kwargs) -> dict:
        self.last_kwargs = kwargs
        return self._response


_RAW_RESPONSE = {
    "results": [
        {
            "title": "FCA raises concerns",
            "url": "https://fca.org.uk/news",
            "content": "regulator commentary",
            "published_date": "2025-07-22T00:00:00Z",
        },
        {
            "title": "Price rise reaction",
            "url": "https://news.example.com/story",
            "content": "customers shop around",
        },
    ]
}


def test_tavily_maps_raw_hits_to_results():
    client = _FakeTavilyClient(_RAW_RESPONSE)
    provider = TavilySearchProvider(client=client)
    hits = provider.search("insurer price rise", max_results=3)
    assert [h.title for h in hits] == ["FCA raises concerns", "Price rise reaction"]
    assert hits[0].url == "https://fca.org.uk/news"
    assert hits[0].snippet == "regulator commentary"
    assert hits[0].published == datetime.date(2025, 7, 22)
    assert all(h.source == "tavily" for h in hits)
    assert client.last_kwargs == {"query": "insurer price rise", "max_results": 3}


def test_tavily_missing_date_is_none():
    provider = TavilySearchProvider(client=_FakeTavilyClient(_RAW_RESPONSE))
    hits = provider.search("q")
    assert hits[1].published is None


def test_tavily_skips_malformed_hits():
    response = {
        "results": [
            {"content": "no url or title"},
            {"url": "https://ok.com", "title": "ok"},
        ]
    }
    provider = TavilySearchProvider(client=_FakeTavilyClient(response))
    hits = provider.search("q")
    assert [h.url for h in hits] == ["https://ok.com"]


def test_tavily_handles_empty_response():
    provider = TavilySearchProvider(client=_FakeTavilyClient({}))
    assert provider.search("q") == []


def test_tavily_rejects_nonpositive_max_results():
    provider = TavilySearchProvider(client=_FakeTavilyClient(_RAW_RESPONSE))
    with pytest.raises(ValueError):
        provider.search("q", max_results=0)


def test_tavily_requires_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Tavily API key"):
        TavilySearchProvider()


def test_tavily_satisfies_protocol():
    assert isinstance(TavilySearchProvider(client=_FakeTavilyClient({})), SearchProvider)


def test_tavily_composes_with_cache(tmp_path):
    client = _FakeTavilyClient(_RAW_RESPONSE)
    provider = TavilySearchProvider(client=client)
    cached = CachedSearchProvider(provider, cache_dir=tmp_path)
    first = cached.search("q")
    client._response = {"results": []}  # inner would now return nothing
    second = cached.search("q")
    assert [h.title for h in first] == [h.title for h in second]  # served from cache
