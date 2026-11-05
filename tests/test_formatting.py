"""Tests for the impact presentation helpers in quant.formatting."""

from __future__ import annotations

import pytest

from decision_sim.quant.formatting import (
    churn_assumption_text,
    churn_breakdown_rows,
    format_gbp,
    format_gbp_millions,
    format_pct_change,
    impact_histogram,
    impact_summary_rows,
)
from decision_sim.schemas import BusinessContext, CustomerGroup, ImpactResult


def _result(samples: list[float]) -> ImpactResult:
    return ImpactResult(
        n_runs=len(samples) or 1,
        baseline_revenue=1_000_000.0,
        new_annual_premium=728.0,
        baseline_churn_rate_mean=0.40,
        churn_rate_mean=0.52,
        extra_churn_rate_mean=0.12,
        price_extra_churn_rate_mean=0.09,
        agent_adjustment_churn_rate_mean=0.03,
        lost_customers_mean=6000.0,
        revenue_impact_mean=-50_000.0,
        revenue_impact_std=10_000.0,
        revenue_impact_p10=-70_000.0,
        revenue_impact_p50=-50_000.0,
        revenue_impact_p90=-30_000.0,
        revenue_impact_pct_mean=-0.05,
        revenue_impact_pct_p10=-0.07,
        revenue_impact_pct_p50=-0.05,
        revenue_impact_pct_p90=-0.03,
        samples=samples,
    )


def test_histogram_lengths_match() -> None:
    result = _result([float(i) for i in range(100)])
    centers, counts = impact_histogram(result, bins=10)
    assert len(centers) == len(counts) == 10
    assert sum(counts) == 100


def test_histogram_empty_samples() -> None:
    centers, counts = impact_histogram(_result([]))
    assert centers == []
    assert counts == []


def test_histogram_rejects_bad_bins() -> None:
    with pytest.raises(ValueError):
        impact_histogram(_result([1.0, 2.0]), bins=0)


def test_histogram_in_millions_scales_centres() -> None:
    result = _result([1_000_000.0, 2_000_000.0, 3_000_000.0])
    raw, _ = impact_histogram(result, bins=3)
    scaled, _ = impact_histogram(result, bins=3, in_millions=True)
    assert scaled == pytest.approx([c / 1_000_000.0 for c in raw])


def test_format_gbp() -> None:
    assert format_gbp(1234.4) == "£1,234"
    assert format_gbp(-1234.4) == "-£1,234"
    assert format_gbp(0.0) == "£0"


def test_format_gbp_millions() -> None:
    assert format_gbp_millions(2_350_000.0) == "£2.35m"
    assert format_gbp_millions(-2_350_000.0) == "-£2.35m"


def test_format_pct_change_is_signed() -> None:
    assert format_pct_change(0.032) == "+3.20%"
    assert format_pct_change(-0.032) == "-3.20%"


def test_summary_rows_contain_key_metrics() -> None:
    rows = dict(impact_summary_rows(_result([1.0, 2.0, 3.0])))
    assert "Mean revenue impact" in rows
    assert "Revenue impact (p50)" in rows
    assert rows["Mean total churn rate"] == "52.00%"
    assert rows["Baseline churn rate (no price rise)"] == "40.00%"
    assert rows["Extra churn from price change"] == "9.00%"
    assert rows["Agent adjustment"] == "+3.00%"


def test_churn_breakdown_four_lines_from_model() -> None:
    rows = churn_breakdown_rows(_result([1.0]))
    labels = [label for label, _ in rows]
    assert labels == [
        "UK base churn (FCA, adjusted by customer type)",
        "Extra churn from price change (model)",
        "Agent adjustment",
        "Total",
    ]
    values = dict(rows)
    assert values["UK base churn (FCA, adjusted by customer type)"] == "40.0%"
    assert values["Extra churn from price change (model)"] == "9.0%"
    assert values["Agent adjustment"] == "+3.0%"
    assert values["Total"] == "52.0%"


def test_churn_breakdown_labels_fallback_as_assumption() -> None:
    rows = dict(churn_breakdown_rows(_result([1.0]), price_from_model=False))
    assert "Extra churn from price change (assumption)" in rows


def test_summary_rows_show_pounds_and_percent() -> None:
    rows = dict(impact_summary_rows(_result([1.0, 2.0, 3.0])))
    assert rows["Revenue impact (p50)"] == "-£0.05m (-5.00%)"
    assert rows["Baseline annual revenue (no price rise)"] == "£1.00m"


def test_churn_assumption_text_states_parameters() -> None:
    business = BusinessContext(
        customer_base=1000,
        annual_premium=500.0,
        price_increase_pct=12.0,
        customer_groups=[CustomerGroup(name="G", agent="A", share=1.0)],
        base_churn=0.40,
        base_churn_sd=0.03,
        base_churn_min=0.33,
        base_churn_max=0.54,
        max_extra_churn=0.15,
        max_extra_churn_sd=0.04,
    )
    text = churn_assumption_text(business)
    assert "40%" in text
    assert "33%–54%" in text
    assert "15pp" in text
    assert "ASSUMPTION" in text
