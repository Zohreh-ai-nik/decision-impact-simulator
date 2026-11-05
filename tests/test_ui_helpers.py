"""Tests for UI-supporting helpers that live in src (not the Streamlit layer)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from decision_sim import llm
from decision_sim.config import SimConfig
from decision_sim.scenario import add_signal, list_scenarios, load_named
from decision_sim.schemas import AgentProfile, Channel, Scenario, Stage
from decision_sim.simulation import graph
from decision_sim.simulation.graph import iter_simulation, stances_dataframe
from tests.conftest import FakeLLM

_DEFAULT_REPLY = json.dumps(
    {
        "channel": "Main",
        "text": "My reaction.",
        "stance": {
            "score": 0.25,
            "likely_action": "wait",
            "confidence": 0.6,
            "reason": "Monitoring the situation.",
        },
    }
)


def _agent(name: str) -> AgentProfile:
    return AgentProfile(name=name, type="person", role=f"{name} role")


def _scenario() -> Scenario:
    return Scenario(
        decision="Raise premiums by 12%.",
        client="ShieldDrive Insurance (fictional)",
        stakeholders=[_agent("A"), _agent("B")],
        stages=[
            Stage(name="S1", goal="g1", rounds=2),
            Stage(name="S2", goal="g2", rounds=1),
        ],
        channels=[Channel(name="Main", can_post=["A", "B"], can_read=["A", "B"])],
    )


@pytest.fixture
def patched_llm(monkeypatch: pytest.MonkeyPatch, fake_llm: FakeLLM) -> FakeLLM:
    fake_llm.default = _DEFAULT_REPLY
    monkeypatch.setattr(llm, "completion", fake_llm)
    return fake_llm


def test_list_scenarios_finds_bundled() -> None:
    files = list_scenarios()
    assert any(p.stem == "insurer_price_rise" for p in files)


def test_add_signal_appends_and_validates() -> None:
    scenario = load_named("insurer_price_rise")
    original_count = len(scenario.signals)

    updated = add_signal(scenario, "A new rival launches.", "Reaction", 1)

    assert len(updated.signals) == original_count + 1
    assert updated.signals[-1].text == "A new rival launches."
    # Original is untouched.
    assert len(scenario.signals) == original_count


def test_add_signal_rejects_unknown_stage() -> None:
    scenario = load_named("insurer_price_rise")
    with pytest.raises(ValidationError):
        add_signal(scenario, "x", "NoSuchStage", 1)


def test_iter_simulation_yields_one_state_per_round(patched_llm: FakeLLM) -> None:
    scenario = _scenario()
    config = SimConfig(model="fake/model", max_rounds=10, seed=1)

    states = list(iter_simulation(scenario, config))

    # 2 rounds in S1 + 1 round in S2 == 3 rounds.
    assert [s["rounds_executed"] for s in states] == [1, 2, 3]
    assert states[-1]["stage_index"] == len(scenario.stages)


def test_iter_simulation_matches_run_simulation(patched_llm: FakeLLM) -> None:
    scenario = _scenario()
    config = SimConfig(model="fake/model", max_rounds=10, seed=1)

    streamed = list(iter_simulation(scenario, config))[-1]
    final = graph.run_simulation(scenario, config, save=False)

    assert streamed["rounds_executed"] == final["rounds_executed"]
    assert len(streamed["posts"]) == len(final["posts"])


def test_stances_dataframe_shape(patched_llm: FakeLLM) -> None:
    scenario = _scenario()
    config = SimConfig(model="fake/model", max_rounds=10, seed=1)
    final = graph.run_simulation(scenario, config, save=False)

    df = stances_dataframe(final["stances"])

    assert list(df.columns) == ["agent", "stage", "round", "global_round", "score"]
    assert set(df["agent"]) == {"A", "B"}
    # Each agent acts once per round over 3 rounds.
    assert df[df["agent"] == "A"]["global_round"].tolist() == [1, 2, 3]


def test_stances_dataframe_empty() -> None:
    df = stances_dataframe([])
    assert df.empty
    assert list(df.columns) == ["agent", "stage", "round", "global_round", "score"]
