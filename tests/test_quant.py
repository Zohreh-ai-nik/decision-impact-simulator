"""Tests for the quantitative layer (mapping + Monte Carlo)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from decision_sim.quant.mapping import (
    aggregate_stances,
    extra_churn_probability,
    total_churn_probability,
)
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

ALL = [CustomerGroup(name="All", agent="Customers", share=1.0)]


def _stance(score: float, confidence: float = 1.0) -> Stance:
    return Stance(
        score=score,
        likely_action="act",
        confidence=confidence,
        reason="because",
    )


def _business(
    groups: list[CustomerGroup] | None = None, base: int = 100_000, **overrides: float
) -> BusinessContext:
    params: dict = {
        "customer_base": base,
        "annual_premium": 600.0,
        "price_increase_pct": 12.0,
        "customer_groups": groups or ALL,
    }
    params.update(overrides)
    return BusinessContext(**params)


def _no_uncertainty() -> dict[str, float]:
    return {"base_churn_sd": 0.0, "max_extra_churn_sd": 0.0}


def _spread(samples: list[float]) -> float:
    ordered = sorted(samples)
    n = len(ordered)
    return ordered[int(n * 0.9)] - ordered[int(n * 0.1)]


# --- mapping -------------------------------------------------------------


def test_extra_churn_zero_when_fully_supportive() -> None:
    assert extra_churn_probability(_stance(1.0, 1.0)) == 0.0


def test_total_churn_is_base_when_supportive() -> None:
    assert total_churn_probability(_stance(1.0, 1.0), base_churn=0.08) == pytest.approx(0.08)


def test_total_churn_hits_cap_when_hostile_and_confident() -> None:
    p = total_churn_probability(_stance(-1.0, 1.0), base_churn=0.08, max_extra_churn=0.15)
    assert p == pytest.approx(0.23)


def test_total_churn_within_bounds_for_extreme_stances() -> None:
    for score in (-1.0, -0.3, 0.0, 0.5, 1.0):
        for conf in (0.0, 0.4, 1.0):
            p = total_churn_probability(
                _stance(score, conf), base_churn=0.08, max_extra_churn=0.15
            )
            assert 0.08 <= p <= 0.23 + 1e-12


def test_total_churn_monotonic_in_score_and_confidence() -> None:
    assert total_churn_probability(_stance(-1.0)) > total_churn_probability(_stance(0.0))
    assert total_churn_probability(_stance(0.0)) > total_churn_probability(_stance(1.0))
    assert total_churn_probability(_stance(-1.0, 1.0)) > total_churn_probability(
        _stance(-1.0, 0.5)
    )


def test_churn_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError):
        extra_churn_probability(_stance(0.0), max_extra_churn=1.5)
    with pytest.raises(ValueError):
        total_churn_probability(_stance(0.0), base_churn=-0.1)


def test_aggregate_stances_empty_is_base_churn() -> None:
    result = aggregate_stances({}, base_churn=0.08)
    assert result == {"mean_score": 0.0, "mean_churn_probability": 0.08}


# --- monte carlo: reproducibility and structure --------------------------


def test_same_seed_gives_same_result() -> None:
    business = _business()
    stances = {"Customers": _stance(-0.5, 0.8)}
    cfg = MonteCarloConfig(n_runs=500, seed=42)

    first = simulate_impact(stances, business, cfg)
    second = simulate_impact(stances, business, cfg)

    assert first.samples == second.samples
    assert first.revenue_impact_pct_p50 == second.revenue_impact_pct_p50


def test_different_seed_changes_result() -> None:
    business = _business()
    stances = {"Customers": _stance(-0.5, 0.8)}

    a = simulate_impact(stances, business, MonteCarloConfig(n_runs=300, seed=1))
    b = simulate_impact(stances, business, MonteCarloConfig(n_runs=300, seed=2))

    assert a.samples != b.samples


def test_percentiles_are_ordered() -> None:
    result = simulate_impact(
        {"Customers": _stance(-0.6, 0.9)},
        _business(),
        MonteCarloConfig(n_runs=1000, seed=1),
    )

    assert result.revenue_impact_p10 <= result.revenue_impact_p50 <= result.revenue_impact_p90
    assert (
        result.revenue_impact_pct_p10
        <= result.revenue_impact_pct_p50
        <= result.revenue_impact_pct_p90
    )


# --- monte carlo: calibration and caps ----------------------------------


def test_hostile_churn_stays_within_caps() -> None:
    business = _business(**_no_uncertainty())  # base 40% + up to 15pp
    result = simulate_impact(
        {"Customers": _stance(-1.0, 1.0)}, business, MonteCarloConfig(n_runs=300, seed=3)
    )

    # Roughly 55%, i.e. base 40% plus the full 15pp extra churn cap.
    assert 0.50 <= result.churn_rate_mean <= 0.60
    assert result.baseline_churn_rate_mean == pytest.approx(0.40, abs=0.005)


def test_extreme_stances_never_exceed_base_plus_cap() -> None:
    business = _business(**_no_uncertainty())
    cap = business.base_churn + business.max_extra_churn

    for score, conf in [(-1.0, 1.0), (-1.0, 0.0), (1.0, 1.0), (0.0, 0.5)]:
        result = simulate_impact(
            {"Customers": _stance(score, conf)}, business, MonteCarloConfig(n_runs=200, seed=4)
        )
        assert result.baseline_churn_rate_mean - 0.005 <= result.churn_rate_mean
        assert result.churn_rate_mean <= cap + 0.005


def test_churn_rate_always_valid_even_with_wild_uncertainty() -> None:
    business = _business(
        base_churn=0.5, base_churn_sd=2.0, max_extra_churn=0.9, max_extra_churn_sd=2.0
    )
    result = simulate_impact(
        {"Customers": _stance(-1.0, 1.0)}, business, MonteCarloConfig(n_runs=300, seed=5)
    )

    assert 0.0 <= result.churn_rate_mean <= 1.0
    assert 0.0 <= result.baseline_churn_rate_mean <= 1.0


def test_supportive_customers_only_churn_at_base_rate() -> None:
    result = simulate_impact(
        {"Customers": _stance(1.0, 1.0)},
        _business(**_no_uncertainty()),
        MonteCarloConfig(n_runs=200, seed=1),
    )

    assert result.churn_rate_mean == pytest.approx(result.baseline_churn_rate_mean)
    # Same churn as baseline + a higher premium => revenue strictly rises.
    assert result.revenue_impact_mean > 0.0
    assert result.revenue_impact_pct_mean == pytest.approx(0.12, abs=1e-9)


def test_hostile_customers_churn_more_than_supportive() -> None:
    cfg = MonteCarloConfig(n_runs=1000, seed=3)
    business = _business()

    hostile = simulate_impact({"Customers": _stance(-1.0, 1.0)}, business, cfg)
    supportive = simulate_impact({"Customers": _stance(0.5, 1.0)}, business, cfg)

    assert hostile.churn_rate_mean > supportive.churn_rate_mean
    assert hostile.revenue_impact_mean < supportive.revenue_impact_mean


# --- monte carlo: no-price-rise baseline ---------------------------------


def test_no_price_rise_and_supportive_stance_gives_zero_impact() -> None:
    """With no rise and no extra churn, scenario == baseline exactly (coupled draws)."""
    result = simulate_impact(
        {"Customers": _stance(1.0, 1.0)},
        _business(price_increase_pct=0.0),
        MonteCarloConfig(n_runs=200, seed=8),
    )

    assert result.revenue_impact_mean == pytest.approx(0.0)
    assert result.revenue_impact_pct_p10 == pytest.approx(0.0)
    assert result.revenue_impact_pct_p90 == pytest.approx(0.0)


def test_no_price_rise_but_hostile_stance_loses_revenue() -> None:
    result = simulate_impact(
        {"Customers": _stance(-1.0, 1.0)},
        _business(price_increase_pct=0.0),
        MonteCarloConfig(n_runs=200, seed=8),
    )

    assert result.revenue_impact_mean < 0.0


def test_impact_pct_is_relative_to_baseline() -> None:
    result = simulate_impact(
        {"Customers": _stance(-0.4, 0.7)},
        _business(),
        MonteCarloConfig(n_runs=500, seed=6),
    )

    assert result.revenue_impact_pct_mean == pytest.approx(
        result.revenue_impact_mean / result.baseline_revenue, rel=0.05
    )


# --- monte carlo: parameter uncertainty ----------------------------------


def test_uncertainty_widens_the_spread() -> None:
    stances = {"Customers": _stance(-0.7, 0.9)}
    cfg = MonteCarloConfig(n_runs=2000, seed=11)

    fixed = simulate_impact(stances, _business(**_no_uncertainty()), cfg)
    uncertain = simulate_impact(
        stances,
        _business(base_churn_sd=0.03, max_extra_churn_sd=0.05),
        cfg,
    )

    assert _spread(uncertain.samples) > 3 * _spread(fixed.samples)
    assert uncertain.revenue_impact_std > fixed.revenue_impact_std


# --- monte carlo: segments and agents ------------------------------------


def test_non_customer_agents_are_ignored() -> None:
    business = _business()
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
    with pytest.raises(KeyError):
        simulate_impact({"Someone else": _stance(0.0)}, _business())


def test_multiple_groups_mix_churn() -> None:
    business = _business(
        [
            CustomerGroup(name="Loyal", agent="Loyal", share=0.6),
            CustomerGroup(name="New", agent="New", share=0.4),
        ],
        **_no_uncertainty(),
    )
    stances = {"Loyal": _stance(1.0, 1.0), "New": _stance(-1.0, 1.0)}
    result = simulate_impact(stances, business, MonteCarloConfig(n_runs=300, seed=5))

    # 40% base for everyone + 15pp extra for the hostile 40% => about 46%.
    assert result.churn_rate_mean == pytest.approx(0.40 + 0.4 * 0.15, abs=0.01)


def test_ungrouped_customers_churn_at_base_rate_only() -> None:
    half = [CustomerGroup(name="Half", agent="Customers", share=0.5)]
    business = _business(half, **_no_uncertainty())
    result = simulate_impact(
        {"Customers": _stance(-1.0, 1.0)}, business, MonteCarloConfig(n_runs=300, seed=5)
    )

    # Hostile half adds 15pp * 0.5 on top of the 40% base.
    assert result.churn_rate_mean == pytest.approx(0.40 + 0.5 * 0.15, abs=0.01)


# --- schema validation ---------------------------------------------------


def test_business_rejects_shares_over_one() -> None:
    with pytest.raises(ValidationError):
        _business(
            [
                CustomerGroup(name="A", agent="X", share=0.7),
                CustomerGroup(name="B", agent="Y", share=0.5),
            ]
        )


def test_business_rejects_out_of_range_churn() -> None:
    with pytest.raises(ValidationError):
        _business(base_churn=1.5)
    with pytest.raises(ValidationError):
        _business(max_extra_churn_sd=-0.1)


def test_scenario_rejects_unknown_customer_agent() -> None:
    with pytest.raises(ValidationError, match="unknown agent"):
        Scenario(
            decision="Raise prices.",
            client="c",
            stakeholders=[AgentProfile(name="A", type="person", role="r")],
            stages=[Stage(name="S1", goal="g", rounds=1)],
            channels=[Channel(name="M", can_post=["A"], can_read=["A"])],
            business=_business([CustomerGroup(name="G", agent="Ghost", share=1.0)]),
        )


def test_bundled_scenario_has_calibrated_business() -> None:
    scenario = load_named("insurer_price_rise")
    assert scenario.business is not None
    assert scenario.business.customer_base == 500_000
    assert scenario.business.price_increase_pct == 12.0
    assert scenario.business.customer_groups[0].agent == "Loyal Customers"
    assert scenario.business.base_churn == 0.40
    assert scenario.business.max_extra_churn == 0.15
    assert scenario.business.base_churn_sd > 0.0
    assert scenario.business.max_extra_churn_sd > 0.0


def test_bundled_scenario_base_churn_within_fca_bounds() -> None:
    """Guard against calibration drift: base churn stays inside the FCA subgroup range."""
    business = load_named("insurer_price_rise").business
    assert business is not None
    assert business.base_churn_min <= business.base_churn <= business.base_churn_max
    assert business.base_churn_min == pytest.approx(0.33)
    assert business.base_churn_max == pytest.approx(0.54)


def test_bundled_scenario_cites_sources() -> None:
    business = load_named("insurer_price_rise").business
    assert business is not None
    assert len(business.sources) >= 1
    assert any("fca.org.uk" in source.url for source in business.sources)


def test_business_rejects_min_above_max() -> None:
    with pytest.raises(ValidationError):
        _business(base_churn_min=0.6, base_churn_max=0.4)


def test_sampled_base_churn_respects_clip_bounds() -> None:
    """Even with a huge sd, baseline churn stays within [min, max]."""
    business = _business(
        base_churn=0.40,
        base_churn_sd=0.5,
        base_churn_min=0.33,
        base_churn_max=0.54,
    )
    result = simulate_impact(
        {"Customers": _stance(1.0, 1.0)},  # supportive => no extra churn
        business,
        MonteCarloConfig(n_runs=2000, seed=12),
    )
    # Supportive stance adds no extra churn, so total == baseline, bounded by clips.
    assert 0.33 - 0.01 <= result.churn_rate_mean <= 0.54 + 0.01
    assert 0.33 - 0.01 <= result.baseline_churn_rate_mean <= 0.54 + 0.01
