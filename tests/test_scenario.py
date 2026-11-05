"""Tests for loading the bundled YAML scenario."""

from __future__ import annotations

import pytest

from decision_sim.scenario import load_named, load_scenario
from decision_sim.schemas import Scenario


def test_load_insurer_price_rise() -> None:
    scenario = load_named("insurer_price_rise")

    assert isinstance(scenario, Scenario)
    assert scenario.client == "ShieldDrive Insurance (fictional)"
    assert "12%" in scenario.decision
    assert len(scenario.stakeholders) == 6
    assert len(scenario.stages) == 4
    assert len(scenario.channels) == 3
    assert len(scenario.signals) == 1


def test_load_named_accepts_suffix() -> None:
    scenario = load_named("insurer_price_rise.yaml")
    assert scenario.client == "ShieldDrive Insurance (fictional)"


def test_load_scenario_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_scenario("does_not_exist.yaml")
