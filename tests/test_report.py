"""Tests for activity metrics and the composed run report."""

from __future__ import annotations

from decision_sim.analysis.activity import (
    activity_counts,
    activity_dataframe,
    channel_counts,
)
from decision_sim.analysis.report import build_report
from decision_sim.schemas import (
    AgentProfile,
    Channel,
    Post,
    Scenario,
    Stage,
    Stance,
)

BREAKING = "Breaking News"


def _agent(name: str) -> AgentProfile:
    return AgentProfile(name=name, type="person", role=f"{name} role")


def _post(agent: str, channel: str = "public") -> Post:
    return Post(agent=agent, channel=channel, stage="S1", round=1, text="t")


def _stance(score: float) -> Stance:
    return Stance(score=score, likely_action="act", confidence=0.8, reason="r")


def _scenario() -> Scenario:
    return Scenario(
        decision="Raise prices.",
        client="c",
        stakeholders=[_agent("A"), _agent("B"), _agent("C")],
        stages=[Stage(name="S1", goal="g", rounds=1)],
        channels=[Channel(name="public", can_post=["A", "B", "C"], can_read=["A", "B", "C"])],
    )


# --- activity ------------------------------------------------------------


def test_activity_counts_excludes_breaking_news() -> None:
    posts = [_post("A"), _post("A"), _post("B"), _post(BREAKING)]
    assert activity_counts(posts) == {"A": 2, "B": 1}


def test_channel_counts_counts_all() -> None:
    posts = [_post("A", "public"), _post("B", "private"), _post(BREAKING, "public")]
    assert channel_counts(posts) == {"public": 2, "private": 1}


def test_activity_dataframe_sorted_desc() -> None:
    posts = [_post("B"), _post("A"), _post("A")]
    df = activity_dataframe(posts)
    assert list(df["agent"]) == ["A", "B"]
    assert list(df["posts"]) == [2, 1]


def test_activity_dataframe_empty() -> None:
    df = activity_dataframe([])
    assert df.empty
    assert list(df.columns) == ["agent", "posts"]


# --- report --------------------------------------------------------------


def test_build_report_aggregates() -> None:
    scenario = _scenario()
    posts = [_post("A"), _post("A"), _post("B"), _post(BREAKING)]
    stances = {"A": _stance(0.8), "B": _stance(0.7), "C": _stance(-0.9)}

    report = build_report(posts, stances, scenario)

    assert report.total_posts == 3  # Breaking News excluded
    assert report.activity == {"A": 2, "B": 1}
    assert ["A", "B"] in report.coalitions
    assert ["C"] in report.coalitions
    assert report.most_influential is not None
    assert report.mean_score == (0.8 + 0.7 - 0.9) / 3


def test_build_report_serialisable() -> None:
    scenario = _scenario()
    posts = [_post("A")]
    stances = {"A": _stance(0.5)}
    report = build_report(posts, stances, scenario)
    # Round-trips through JSON without error.
    assert "total_posts" in report.model_dump()
