"""Tests for the quantitative layer (mapping + Monte Carlo)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from decision_sim.quant.mapping import aggregate_stances, churn_probability
from decision_sim.quant.monte_carlo import MonteCarloConfig, simulate_impact
from decision_sim.scenario import load_named
from decision_sim.schemas import (
    AgentProfile,
    BusinessContext,
    Channel,
    CustomerGroup,
    Scenario,
    Stage,
    Stance,
)


def _stance(score: float, confidence: float = 1.0) -> Stance:
    return Stance(
        score=score,
        likely_action="act",
        confidence=confidence,
        reason="because",
    )


def _business(groups: list[CustomerGroup], base: int = 100_000) -> BusinessContext:
    return BusinessContext(
        customer_base=base,
        annual_premium=600.0,
        price_increase_pct=12.0,
        customer_groups=groups,
    )


# --- mapping -------------------------------------------------------------


def test_churn_zero_when_fully_supportive() -> None:
    assert churn_probability(_stance(1.0, 1.0)) == 0.0


def test_churn_hits_cap_when_hostile_and_confident() -> None:
    assert churn_probability(_stance(-1.0, 1.0), max_churn=0.6) == pytest.approx(0.6)


def test_churn_always_in_unit_interval() -> None:
    for score in (-1.0, -0.3, 0.0, 0.5, 1.0):
        for conf in (0.0, 0.4, 1.0):
            p = churn_probability(_stance(score, conf))
            assert 0.0 <= p <= 1.0


def test_churn_monotonic_in_score_and_confidence() -> None:
    assert churn_probability(_stance(-1.0)) > churn_probability(_stance(0.0))
    assert churn_probability(_stance(0.0)) > churn_probability(_stance(1.0))
    assert churn_probability(_stance(-1.0, 1.0)) > churn_probability(_stance(-1.0, 0.5))


def test_churn_rejects_bad_max() -> None:
    with pytest.raises(ValueError):
        churn_probability(_stance(0.0), max_churn=1.5)


def test_aggregate_stances_empty() -> None:
    result = aggregate_stances({})
    assert result == {"mean_score": 0.0, "mean_churn_probability": 0.0}


# --- monte carlo ---------------------------------------------------------


def test_simulate_impact_is_reproducible() -> None:
    business = _business([CustomerGroup(name="All", agent="Customers", share=1.0)])
    stances = {"Customers": _stance(-0.5, 0.8)}
    cfg = MonteCarloConfig(n_runs=500, seed=42)

    first = simulate_impact(stances, business, cfg)
    second = simulate_impact(stances, business, cfg)

    assert first.samples == second.samples


def test_percentiles_are_ordered() -> None:
    business = _business([CustomerGroup(name="All", agent="Customers", share=1.0)])
    stances = {"Customers": _stance(-0.6, 0.9)}
    result = simulate_impact(stances, business, MonteCarloConfig(n_runs=1000, seed=1))

    assert result.revenue_impact_p10 <= result.revenue_impact_p50 <= result.revenue_impact_p90


def test_supportive_customers_give_positive_impact() -> None:
    business = _business([CustomerGroup(name="All", agent="Customers", share=1.0)])
    stances = {"Customers": _stance(1.0, 1.0)}  # zero churn
    result = simulate_impact(stances, business, MonteCarloConfig(n_runs=200, seed=1))

    assert result.churn_rate_mean == 0.0
    # No churn + higher premium => revenue strictly rises.
    assert result.revenue_impact_mean > 0.0


def test_hostile_customers_churn_more_than_supportive() -> None:
    business = _business([CustomerGroup(name="All", agent="Customers", share=1.0)])
    cfg = MonteCarloConfig(n_runs=1000, seed=3)

    hostile = simulate_impact({"Customers": _stance(-1.0, 1.0)}, business, cfg)
    supportive = simulate_impact({"Customers": _stance(0.5, 1.0)}, business, cfg)

    assert hostile.churn_rate_mean > supportive.churn_rate_mean


def test_non_customer_agents_are_ignored() -> None:
    business = _business([CustomerGroup(name="All", agent="Customers", share=1.0)])
    cfg = MonteCarloConfig(n_runs=500, seed=9)

    without_extras = simulate_impact({"Customers": _stance(-0.4, 0.7)}, business, cfg)
    with_extras = simulate_impact(
        {
            "Customers": _stance(-0.4, 0.7),
            "CEO": _stance(1.0, 1.0),
            "Regulator": _stance(-1.0, 1.0),
        },
        business,
        cfg,
    )

    assert without_extras.samples == with_extras.samples


def test_missing_customer_agent_raises() -> None:
    business = _business([CustomerGroup(name="All", agent="Customers", share=1.0)])
    with pytest.raises(KeyError):
        simulate_impact({"Someone else": _stance(0.0)}, business)


def test_multiple_groups_sum_churn() -> None:
    business = _business(
        [
            CustomerGroup(name="Loyal", agent="Loyal", share=0.6),
            CustomerGroup(name="New", agent="New", share=0.4),
        ]
    )
    stances = {"Loyal": _stance(0.8, 1.0), "New": _stance(-1.0, 1.0)}
    result = simulate_impact(stances, business, MonteCarloConfig(n_runs=500, seed=5))

    # Hostile 40% group drives most churn; rate should be clearly positive.
    assert result.churn_rate_mean > 0.0
    assert 0.0 <= result.churn_rate_mean <= 1.0


# --- schema validation ---------------------------------------------------


def test_business_rejects_shares_over_one() -> None:
    with pytest.raises(ValidationError):
        _business(
            [
                CustomerGroup(name="A", agent="X", share=0.7),
                CustomerGroup(name="B", agent="Y", share=0.5),
            ]
        )


def test_scenario_rejects_unknown_customer_agent() -> None:
    with pytest.raises(ValidationError, match="unknown agent"):
        Scenario(
            decision="Raise prices.",
            client="c",
            stakeholders=[AgentProfile(name="A", type="person", role="r")],
            stages=[Stage(name="S1", goal="g", rounds=1)],
            channels=[Channel(name="M", can_post=["A"], can_read=["A"])],
            business=_business(
                [CustomerGroup(name="G", agent="Ghost", share=1.0)]
            ),
        )


def test_bundled_scenario_has_business() -> None:
    scenario = load_named("insurer_price_rise")
    assert scenario.business is not None
    assert scenario.business.customer_base == 500_000
    assert scenario.business.price_increase_pct == 12.0
    assert scenario.business.customer_groups[0].agent == "Loyal Customers"
