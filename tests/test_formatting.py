"""Tests for the impact presentation helpers in quant.formatting."""

from __future__ import annotations

import pytest

from decision_sim.quant.formatting import impact_histogram, impact_summary_rows
from decision_sim.schemas import ImpactResult


def _result(samples: list[float]) -> ImpactResult:
    return ImpactResult(
        n_runs=len(samples) or 1,
        baseline_revenue=1_000_000.0,
        new_annual_premium=728.0,
        churn_rate_mean=0.12,
        lost_customers_mean=6000.0,
        revenue_impact_mean=50_000.0,
        revenue_impact_std=10_000.0,
        revenue_impact_p10=30_000.0,
        revenue_impact_p50=50_000.0,
        revenue_impact_p90=70_000.0,
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


def test_summary_rows_contain_key_metrics() -> None:
    rows = impact_summary_rows(_result([1.0, 2.0, 3.0]))
    labels = {label for label, _ in rows}
    assert "Mean revenue impact" in labels
    assert "Mean churn rate" in labels
    assert "Revenue impact (p50)" in labels
    # Churn rate is formatted as a percentage.
    churn = dict(rows)["Mean churn rate"]
    assert churn == "12.00%"
