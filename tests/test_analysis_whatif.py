"""Offline tests for the what-if (counterfactual) analysis."""

from __future__ import annotations

import pytest

from decision_sim.analysis.whatif import (
    compare_impacts,
    override_stances,
    run_whatif,
    with_price_increase,
)
from decision_sim.quant.monte_carlo import MonteCarloConfig, simulate_impact
from decision_sim.schemas import BusinessContext, CustomerGroup, Stance

CUSTOMER = "Customers"
GROUPS = [CustomerGroup(name="All", agent=CUSTOMER, share=1.0)]


def _stance(score: float, confidence: float = 1.0) -> Stance:
    return Stance(score=score, likely_action="act", confidence=confidence, reason="because")


def _business(**overrides: float) -> BusinessContext:
    params: dict = {
        "customer_base": 100_000,
        "annual_premium": 600.0,
        "price_increase_pct": 12.0,
        "customer_groups": GROUPS,
        "base_churn_sd": 0.0,
        "max_extra_churn_sd": 0.0,
    }
    params.update(overrides)
    return BusinessContext(**params)


def _config() -> MonteCarloConfig:
    return MonteCarloConfig(n_runs=2000, seed=42)


# --- override_stances -------------------------------------------------------


def test_override_stances_replaces_only_named_agents() -> None:
    base = {CUSTOMER: _stance(0.5), "Regulator": _stance(-0.5)}
    updated = override_stances(base, {CUSTOMER: _stance(-1.0)})
    assert updated[CUSTOMER].score == -1.0
    assert updated["Regulator"].score == -0.5


def test_override_stances_does_not_mutate_original() -> None:
    base = {CUSTOMER: _stance(0.5)}
    override_stances(base, {CUSTOMER: _stance(-1.0)})
    assert base[CUSTOMER].score == 0.5


def test_override_stances_rejects_unknown_agent() -> None:
    with pytest.raises(KeyError):
        override_stances({CUSTOMER: _stance(0.5)}, {"Ghost": _stance(0.0)})


# --- with_price_increase ----------------------------------------------------


def test_with_price_increase_copies_without_mutating() -> None:
    business = _business()
    variant = with_price_increase(business, 20.0)
    assert variant.price_increase_pct == 20.0
    assert business.price_increase_pct == 12.0


# --- compare_impacts / run_whatif -------------------------------------------


def test_compare_impacts_computes_deltas() -> None:
    stances = {CUSTOMER: _stance(0.0)}
    business = _business()
    base = simulate_impact(stances, business, _config())
    variant = simulate_impact(
        override_stances(stances, {CUSTOMER: _stance(-1.0)}), business, _config()
    )

    cmp = compare_impacts(base, variant)

    assert cmp.baseline_impact_mean == base.revenue_impact_mean
    assert cmp.variant_impact_mean == variant.revenue_impact_mean
    assert cmp.delta_mean == pytest.approx(
        variant.revenue_impact_mean - base.revenue_impact_mean
    )
    assert cmp.delta_churn_rate_mean == pytest.approx(
        variant.churn_rate_mean - base.churn_rate_mean
    )


def test_more_hostile_customer_worsens_impact() -> None:
    # Customer agents drive churn; a more hostile CUSTOMER means more churn and a
    # worse (more negative) revenue impact.
    stances = {CUSTOMER: _stance(0.5)}
    business = _business()

    cmp = run_whatif(
        stances,
        business,
        stance_overrides={CUSTOMER: _stance(-1.0)},
        config=_config(),
    )

    assert cmp.delta_churn_rate_mean > 0.0
    assert cmp.delta_mean < 0.0


def test_price_increase_changes_revenue_not_churn() -> None:
    # In this quant model the price-change churn is a fixed assumption, NOT scaled
    # by the size of the rise, so changing the price moves revenue but leaves churn
    # unchanged. A larger rise therefore yields more revenue per retained customer.
    stances = {CUSTOMER: _stance(0.0)}
    business = _business()

    cmp = run_whatif(
        stances,
        business,
        price_increase_pct=4.0,
        config=_config(),
    )

    # Churn is independent of the price magnitude here.
    assert cmp.delta_churn_rate_mean == 0.0
    # A smaller rise (4% vs 12%) means less premium per customer -> lower impact.
    assert cmp.variant_impact_mean < cmp.baseline_impact_mean
    assert cmp.delta_mean < 0.0


def test_non_customer_override_has_no_effect_on_churn() -> None:
    # A Regulator is not a customer group's agent, so overriding it must not change
    # the quant impact at all (confirms churn is customer-driven here).
    stances = {CUSTOMER: _stance(0.0), "Regulator": _stance(0.5)}
    business = _business()

    cmp = run_whatif(
        stances,
        business,
        stance_overrides={"Regulator": _stance(-1.0)},
        config=_config(),
    )

    assert cmp.delta_mean == 0.0
    assert cmp.delta_churn_rate_mean == 0.0
