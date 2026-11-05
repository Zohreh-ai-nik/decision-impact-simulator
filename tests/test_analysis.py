"""Tests for the analysis layer (coalitions + influence)."""

from __future__ import annotations

import pytest

from decision_sim.analysis.coalitions import detect_coalitions, final_scores
from decision_sim.analysis.influence import (
    BREAKING_NEWS_AGENT,
    build_interaction_graph,
    influence_ranking,
)
from decision_sim.schemas import (
    AgentProfile,
    Channel,
    Post,
    Scenario,
    Stage,
    Stance,
)


def _stance(score: float) -> Stance:
    return Stance(score=score, likely_action="act", confidence=0.8, reason="r")


def _agent(name: str) -> AgentProfile:
    return AgentProfile(name=name, type="person", role=f"{name} role")


# --- coalitions ----------------------------------------------------------


def test_final_scores() -> None:
    stances = {"A": _stance(0.5), "B": _stance(-0.2)}
    assert final_scores(stances) == {"A": 0.5, "B": -0.2}


def test_similar_same_side_agents_form_a_coalition() -> None:
    stances = {"A": _stance(0.8), "B": _stance(0.7), "C": _stance(-0.9)}
    coalitions = detect_coalitions(stances, threshold=0.4)

    # A and B are close and both supportive; C stands alone.
    assert ["A", "B"] in coalitions
    assert ["C"] in coalitions


def test_opposed_agents_do_not_group() -> None:
    stances = {"A": _stance(0.6), "B": _stance(-0.6)}
    coalitions = detect_coalitions(stances, threshold=0.4)
    assert sorted(coalitions) == [["A"], ["B"]]


def test_threshold_controls_grouping() -> None:
    stances = {"A": _stance(0.1), "B": _stance(0.9)}
    # Gap is 0.8: too far at 0.4, close enough at 1.0.
    assert detect_coalitions(stances, threshold=0.4) == [["A"], ["B"]]
    assert detect_coalitions(stances, threshold=1.0) == [["A", "B"]]


def test_detect_coalitions_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError):
        detect_coalitions({"A": _stance(0.0)}, threshold=-0.1)


# --- influence -----------------------------------------------------------


def _scenario() -> Scenario:
    return Scenario(
        decision="Raise prices.",
        client="c",
        stakeholders=[_agent("A"), _agent("B"), _agent("C")],
        stages=[Stage(name="S1", goal="g", rounds=1)],
        channels=[
            Channel(name="public", can_post=["A", "B", "C"], can_read=["A", "B", "C"]),
            Channel(name="private", can_post=["C"], can_read=["C"]),
        ],
    )


def test_breaking_news_excluded_from_graph() -> None:
    scenario = _scenario()
    posts = [
        Post(agent=BREAKING_NEWS_AGENT, channel="public", stage="S1", round=1, text="x"),
    ]
    graph = build_interaction_graph(posts, scenario)
    assert BREAKING_NEWS_AGENT not in graph.nodes or graph.out_degree(BREAKING_NEWS_AGENT) == 0
    assert graph.number_of_edges() == 0


def test_public_poster_outranks_private_poster() -> None:
    scenario = _scenario()
    posts = [
        # A posts widely-read messages; C only posts to its own private channel.
        Post(agent="A", channel="public", stage="S1", round=1, text="a1"),
        Post(agent="A", channel="public", stage="S1", round=1, text="a2"),
        Post(agent="C", channel="private", stage="S1", round=1, text="c1"),
    ]
    ranking = dict(influence_ranking(posts, scenario))
    assert ranking["A"] > ranking["C"]


def test_influence_ranking_no_posts_is_uniform() -> None:
    scenario = _scenario()
    ranking = influence_ranking([], scenario)
    scores = [s for _, s in ranking]
    assert len(ranking) == 3
    assert all(abs(s - scores[0]) < 1e-9 for s in scores)


def test_influence_ranking_sorted_descending() -> None:
    scenario = _scenario()
    posts = [
        Post(agent="A", channel="public", stage="S1", round=1, text="a"),
        Post(agent="B", channel="public", stage="S1", round=1, text="b"),
    ]
    ranking = influence_ranking(posts, scenario)
    scores = [s for _, s in ranking]
    assert scores == sorted(scores, reverse=True)
