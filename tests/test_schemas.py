"""Validation tests for the Pydantic schemas in :mod:`decision_sim.schemas`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from decision_sim.schemas import (
    AgentProfile,
    Channel,
    Scenario,
    Signal,
    Stage,
    Stance,
)


def _agent(name: str) -> AgentProfile:
    return AgentProfile(name=name, type="person", role="role")


def _valid_scenario_kwargs() -> dict:
    return {
        "decision": "Raise premiums by 12%.",
        "client": "ShieldDrive Insurance (fictional)",
        "stakeholders": [_agent("CEO"), _agent("Regulator")],
        "stages": [Stage(name="Announcement", goal="Announce", rounds=1)],
        "channels": [
            Channel(name="Public", can_post=["CEO"], can_read=["CEO", "Regulator"])
        ],
        "signals": [Signal(text="A rival freezes prices.", stage="Announcement", round=1)],
        "assumptions": ["Fictional company."],
    }


def test_agent_profile_valid() -> None:
    agent = AgentProfile(
        name="CEO",
        type="person",
        role="Chief Executive",
        goals=["profit"],
        concerns=["churn"],
        style="decisive",
    )
    assert agent.type == "person"


def test_agent_profile_rejects_bad_type() -> None:
    with pytest.raises(ValidationError):
        AgentProfile(name="X", type="robot", role="role")  # type: ignore[arg-type]


@pytest.mark.parametrize("score", [-1.5, 1.5])
def test_stance_rejects_out_of_range_score(score: float) -> None:
    with pytest.raises(ValidationError):
        Stance(score=score, likely_action="wait", confidence=0.5, reason="because")


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_stance_rejects_out_of_range_confidence(confidence: float) -> None:
    with pytest.raises(ValidationError):
        Stance(score=0.0, likely_action="wait", confidence=confidence, reason="because")


def test_stance_accepts_boundaries() -> None:
    stance = Stance(score=-1.0, likely_action="leave", confidence=1.0, reason="angry")
    assert stance.score == -1.0
    assert stance.confidence == 1.0


def test_stage_requires_positive_rounds() -> None:
    with pytest.raises(ValidationError):
        Stage(name="S", goal="g", rounds=0)


def test_scenario_valid() -> None:
    scenario = Scenario(**_valid_scenario_kwargs())
    assert len(scenario.stakeholders) == 2


def test_scenario_rejects_unknown_channel_agent() -> None:
    kwargs = _valid_scenario_kwargs()
    kwargs["channels"] = [Channel(name="Public", can_post=["Ghost"], can_read=["CEO"])]
    with pytest.raises(ValidationError, match="unknown agent"):
        Scenario(**kwargs)


def test_scenario_rejects_unknown_signal_stage() -> None:
    kwargs = _valid_scenario_kwargs()
    kwargs["signals"] = [Signal(text="x", stage="Nonexistent", round=1)]
    with pytest.raises(ValidationError, match="unknown stage"):
        Scenario(**kwargs)


def test_scenario_rejects_duplicate_stakeholder_names() -> None:
    kwargs = _valid_scenario_kwargs()
    kwargs["stakeholders"] = [_agent("CEO"), _agent("CEO")]
    with pytest.raises(ValidationError, match="Duplicate stakeholder"):
        Scenario(**kwargs)


def test_scenario_forbids_extra_fields() -> None:
    kwargs = _valid_scenario_kwargs()
    kwargs["unexpected"] = "nope"
    with pytest.raises(ValidationError):
        Scenario(**kwargs)
