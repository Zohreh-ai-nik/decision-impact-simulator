"""Presentation helpers for :class:`~decision_sim.schemas.ImpactResult`.

Pure and dependency-light (NumPy + schemas only, no ``simulation``/``ui``), so
the quantitative layer stays easy to test and to extract into a service. These
functions shape an ``ImpactResult`` for display; the UI decides how to render it.
"""

from __future__ import annotations

import numpy as np

from decision_sim.schemas import ImpactResult


def impact_histogram(
    result: ImpactResult, bins: int = 30
) -> tuple[list[float], list[int]]:
    """Return ``(bin_centers, counts)`` for a histogram of the impact samples.

    Args:
        result: The Monte Carlo result whose ``samples`` to bin.
        bins: Number of histogram bins (must be >= 1).

    Returns:
        A tuple of bin centres and integer counts of equal length. Both are
        empty when there are no samples.
    """
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}.")
    if not result.samples:
        return [], []

    counts, edges = np.histogram(result.samples, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers.tolist(), counts.astype(int).tolist()


def _money(value: float) -> str:
    return f"{value:,.0f}"


def impact_summary_rows(result: ImpactResult) -> list[tuple[str, str]]:
    """Return labelled, formatted metric rows summarising an impact result."""
    return [
        ("Baseline annual revenue", _money(result.baseline_revenue)),
        ("New annual premium", _money(result.new_annual_premium)),
        ("Mean revenue impact", _money(result.revenue_impact_mean)),
        ("Revenue impact (p10)", _money(result.revenue_impact_p10)),
        ("Revenue impact (p50)", _money(result.revenue_impact_p50)),
        ("Revenue impact (p90)", _money(result.revenue_impact_p90)),
        ("Mean churn rate", f"{result.churn_rate_mean:.2%}"),
        ("Mean lost customers", _money(result.lost_customers_mean)),
        ("Monte Carlo runs", f"{result.n_runs:,}"),
    ]
