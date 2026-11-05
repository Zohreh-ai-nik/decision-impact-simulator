"""Presentation helpers for :class:`~decision_sim.schemas.ImpactResult`.

Pure and dependency-light (NumPy + schemas only, no ``simulation``/``ui``), so
the quantitative layer stays easy to test and to extract into a service. These
functions shape an ``ImpactResult`` for display; the UI decides how to render it.
"""

from __future__ import annotations

import numpy as np

from decision_sim.schemas import BusinessContext, ImpactResult


def impact_histogram(
    result: ImpactResult, bins: int = 30, *, in_millions: bool = False
) -> tuple[list[float], list[int]]:
    """Return ``(bin_centers, counts)`` for a histogram of the impact samples.

    Args:
        result: The Monte Carlo result whose ``samples`` to bin.
        bins: Number of histogram bins (must be >= 1).
        in_millions: Express the bin centres in millions (e.g. for a £m axis).

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
    if in_millions:
        centers = centers / 1_000_000.0
    return centers.tolist(), counts.astype(int).tolist()


def format_gbp(value: float) -> str:
    """Format a monetary value in pounds with thousands separators, e.g. ``-£1,234``."""
    sign = "-" if round(value) < 0 else ""
    return f"{sign}£{abs(value):,.0f}"


def format_gbp_millions(value: float) -> str:
    """Format a monetary value in millions of pounds, e.g. ``-£2.35m``."""
    sign = "-" if round(value / 1_000_000.0, 2) < 0 else ""
    return f"{sign}£{abs(value) / 1_000_000.0:,.2f}m"


def format_pct_change(fraction: float) -> str:
    """Format a fractional change as a signed percentage, e.g. ``-3.20%``."""
    return f"{fraction * 100:+.2f}%"


def _count(value: float) -> str:
    return f"{value:,.0f}"


def churn_assumption_text(business: BusinessContext) -> str:
    """Describe the churn assumptions used, for display next to the results."""
    return (
        f"Churn assumption: base churn {business.base_churn:.1%} "
        f"(±{business.base_churn_sd * 100:.1f}pp) plus up to "
        f"{business.max_extra_churn * 100:.1f}pp extra from negative customer stance "
        f"(±{business.max_extra_churn_sd * 100:.1f}pp)."
    )


def impact_summary_rows(result: ImpactResult) -> list[tuple[str, str]]:
    """Return labelled, formatted metric rows summarising an impact result.

    Revenue impact is versus the no-price-rise baseline, shown in £ and as a
    percentage change.
    """
    return [
        ("Baseline annual revenue (no price rise)", format_gbp_millions(result.baseline_revenue)),
        ("New annual premium", format_gbp(result.new_annual_premium)),
        (
            "Mean revenue impact",
            f"{format_gbp_millions(result.revenue_impact_mean)} "
            f"({format_pct_change(result.revenue_impact_pct_mean)})",
        ),
        (
            "Revenue impact (p10)",
            f"{format_gbp_millions(result.revenue_impact_p10)} "
            f"({format_pct_change(result.revenue_impact_pct_p10)})",
        ),
        (
            "Revenue impact (p50)",
            f"{format_gbp_millions(result.revenue_impact_p50)} "
            f"({format_pct_change(result.revenue_impact_pct_p50)})",
        ),
        (
            "Revenue impact (p90)",
            f"{format_gbp_millions(result.revenue_impact_p90)} "
            f"({format_pct_change(result.revenue_impact_pct_p90)})",
        ),
        ("Baseline churn rate (no price rise)", f"{result.baseline_churn_rate_mean:.2%}"),
        ("Mean churn rate", f"{result.churn_rate_mean:.2%}"),
        ("Mean lost customers", _count(result.lost_customers_mean)),
        ("Monte Carlo runs", f"{result.n_runs:,}"),
    ]
