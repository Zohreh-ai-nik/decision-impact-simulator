"""Tests for the provenance schemas (Source, Evidence) and sources plumbing."""

from __future__ import annotations

import datetime

import pytest
from pydantic import ValidationError

from decision_sim.schemas import (
    AgentProfile,
    Evidence,
    Post,
    Source,
    Stance,
)


def _source(**overrides) -> Source:
    params = {
        "url": "https://example.com/report",
        "title": "A public report",
    }
    params.update(overrides)
    return Source(**params)


# --- Source --------------------------------------------------------------


def test_source_accepts_https_url() -> None:
    source = _source()
    assert source.url == "https://example.com/report"
    assert source.date is None
    assert source.snippet == ""


def test_source_accepts_http_url() -> None:
    assert _source(url="http://example.com").url == "http://example.com"


def test_source_rejects_non_url() -> None:
    with pytest.raises(ValidationError, match="http"):
        _source(url="not a url")


def test_source_requires_title() -> None:
    with pytest.raises(ValidationError):
        _source(title="")


def test_source_with_date_and_snippet() -> None:
    source = _source(date=datetime.date(2026, 3, 1), snippet="Premiums rose 12%.")
    assert source.date == datetime.date(2026, 3, 1)
    assert source.snippet == "Premiums rose 12%."


def test_source_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Source(url="https://x.com", title="t", author="nope")


def test_source_json_round_trip_preserves_date() -> None:
    source = _source(date=datetime.date(2026, 3, 1), snippet="s")
    restored = Source.model_validate_json(source.model_dump_json())
    assert restored == source
    assert restored.date == datetime.date(2026, 3, 1)


# --- Evidence ------------------------------------------------------------


def test_evidence_requires_at_least_one_source() -> None:
    with pytest.raises(ValidationError):
        Evidence(text="something", sources=[])


def test_evidence_holds_text_and_sources() -> None:
    evidence = Evidence(text="A claim.", sources=[_source()])
    assert evidence.text == "A claim."
    assert evidence.sources[0].title == "A public report"


def test_evidence_requires_text() -> None:
    with pytest.raises(ValidationError):
        Evidence(text="", sources=[_source()])


# --- back-compat: existing models still work without sources -------------


def test_agent_profile_defaults_to_no_sources() -> None:
    profile = AgentProfile(name="CEO", type="person", role="Chief Exec")
    assert profile.sources == []


def test_post_defaults_to_no_sources() -> None:
    post = Post(agent="CEO", channel="c", stage="S1", round=1, text="hi")
    assert post.sources == []


def test_stance_defaults_to_no_sources() -> None:
    stance = Stance(score=0.0, likely_action="wait", confidence=0.5, reason="r")
    assert stance.sources == []


# --- models accept populated sources -------------------------------------


def test_models_accept_sources() -> None:
    sources = [_source()]
    profile = AgentProfile(name="CEO", type="person", role="r", sources=sources)
    post = Post(agent="CEO", channel="c", stage="S1", round=1, text="hi", sources=sources)
    stance = Stance(
        score=0.0, likely_action="wait", confidence=0.5, reason="r", sources=sources
    )
    assert profile.sources == post.sources == stance.sources == sources
