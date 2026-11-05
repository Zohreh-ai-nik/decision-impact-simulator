"""Tests for the LangGraph simulation using FakeLLM."""

from __future__ import annotations

import json

import pytest

from decision_sim import llm
from decision_sim.config import SimConfig
from decision_sim.schemas import (
    AgentProfile,
    Channel,
    Post,
    Scenario,
    Signal,
    Stage,
)
from decision_sim.simulation import graph
from tests.conftest import FakeLLM

_DEFAULT_REPLY = json.dumps(
    {
        "channel": "Any",
        "text": "Here is my reaction.",
        "stance": {
            "score": 0.0,
            "likely_action": "wait and see",
            "confidence": 0.5,
            "reason": "Not enough information yet.",
        },
    }
)


def _agent(name: str) -> AgentProfile:
    return AgentProfile(name=name, type="person", role=f"{name} role")


def _config(**overrides) -> SimConfig:
    base = {"model": "fake/model", "seed": 1, "recent_posts_window": 50}
    base.update(overrides)
    return SimConfig(**base)


@pytest.fixture
def patched_llm(monkeypatch: pytest.MonkeyPatch, fake_llm: FakeLLM) -> FakeLLM:
    fake_llm.default = _DEFAULT_REPLY
    monkeypatch.setattr(llm, "completion", fake_llm)
    return fake_llm


def test_signal_applied_at_the_right_round(patched_llm: FakeLLM) -> None:
    scenario = Scenario(
        decision="Raise premiums by 12%.",
        client="ShieldDrive Insurance (fictional)",
        stakeholders=[_agent("A")],
        stages=[Stage(name="S1", goal="Announce", rounds=2)],
        channels=[Channel(name="Main", can_post=["A"], can_read=["A"])],
        signals=[Signal(text="A rival freezes prices.", stage="S1", round=2)],
    )

    final = graph.run_simulation(scenario, _config(max_rounds=10), save=False)

    breaking = [p for p in final["posts"] if p.agent == graph.BREAKING_NEWS_AGENT]
    assert breaking, "expected a breaking-news post"
    assert all(p.round == 2 for p in breaking)
    assert not any(p.round == 1 for p in breaking)
    assert final["signals_applied"] == [0]


def test_posting_respects_channel_permissions(patched_llm: FakeLLM) -> None:
    # The fake always wants to post to "pub"; A may only post to "priv".
    scenario = Scenario(
        decision="Raise premiums by 12%.",
        client="ShieldDrive Insurance (fictional)",
        stakeholders=[_agent("A"), _agent("B")],
        stages=[Stage(name="S1", goal="React", rounds=1)],
        channels=[
            Channel(name="priv", can_post=["A"], can_read=["A"]),
            Channel(name="pub", can_post=["B"], can_read=["A", "B"]),
        ],
    )

    final = graph.run_simulation(scenario, _config(), save=False)

    a_posts = [p for p in final["posts"] if p.agent == "A"]
    b_posts = [p for p in final["posts"] if p.agent == "B"]
    assert a_posts and all(p.channel == "priv" for p in a_posts)
    assert b_posts and all(p.channel == "pub" for p in b_posts)


def test_recent_readable_posts_filters_unreadable_channels() -> None:
    scenario = Scenario(
        decision="d",
        client="c",
        stakeholders=[_agent("A"), _agent("B")],
        stages=[Stage(name="S1", goal="g", rounds=1)],
        channels=[
            Channel(name="priv", can_post=["A"], can_read=["A"]),
            Channel(name="pub", can_post=["A", "B"], can_read=["A", "B"]),
        ],
    )
    posts = [
        Post(agent="A", channel="priv", stage="S1", round=1, text="secret"),
        Post(agent="A", channel="pub", stage="S1", round=1, text="public"),
    ]

    visible = graph.recent_readable_posts(posts, "B", scenario, window=50)

    assert [p.text for p in visible] == ["public"]


def test_stops_at_max_rounds(patched_llm: FakeLLM) -> None:
    scenario = Scenario(
        decision="Raise premiums by 12%.",
        client="ShieldDrive Insurance (fictional)",
        stakeholders=[_agent("A")],
        stages=[
            Stage(name="S1", goal="g1", rounds=3),
            Stage(name="S2", goal="g2", rounds=3),
        ],
        channels=[Channel(name="Main", can_post=["A"], can_read=["A"])],
    )

    final = graph.run_simulation(scenario, _config(max_rounds=2), save=False)

    assert final["rounds_executed"] == 2
    # Two rounds in S1 means we are still in S1, about to start round 3.
    assert final["stage_index"] == 0
    assert final["round"] == 3
    agent_rounds = {(p.stage, p.round) for p in final["posts"] if p.agent == "A"}
    assert len(agent_rounds) == 2


def test_save_run_writes_json(patched_llm: FakeLLM, tmp_path) -> None:
    scenario = Scenario(
        decision="Raise premiums by 12%.",
        client="ShieldDrive Insurance (fictional)",
        stakeholders=[_agent("A")],
        stages=[Stage(name="S1", goal="g", rounds=1)],
        channels=[Channel(name="Main", can_post=["A"], can_read=["A"])],
    )
    final = graph.run_simulation(scenario, _config(), save=False)

    path = graph.save_run(final, out_dir=tmp_path)

    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["client"] == "ShieldDrive Insurance (fictional)"
    assert "posts" in data and "stances" in data
