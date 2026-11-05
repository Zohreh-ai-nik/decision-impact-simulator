"""Counterfactual ("what-if") re-scoring of the quantitative impact.

Pure and deterministic: given a run's final stances and the business context, it
re-runs the Monte Carlo :func:`simulate_impact` under a changed assumption and
compares the two outcomes -- without re-running the (expensive, LLM-driven)
discussion simulation.

Only agents named in ``business.customer_groups`` drive churn, so a what-if that
changes behaviour must override one of those *customer* agents (or the price
rise). Non-customer agents (regulator, media, competitors) have no direct effect
on the churn maths here; they only matter indirectly, by moving customer stances
during the simulation itself.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from decision_sim.quant.monte_carlo import MonteCarloConfig, simulate_impact
from decision_sim.schemas import BusinessContext, ImpactResult, Stance

__all__ = [
    "WhatIfComparison",
    "compare_impacts",
    "override_stances",
    "run_whatif",
    "with_price_increase",
]


class WhatIfComparison(BaseModel):
    """Baseline vs. what-if revenue impact, with the difference between them."""

    model_config = ConfigDict(extra="forbid")

    baseline_impact_mean: float = Field(description="Mean revenue impact of the base case.")
    variant_impact_mean: float = Field(description="Mean revenue impact of the what-if.")
    delta_mean: float = Field(description="variant minus baseline (positive = better).")
    baseline_churn_rate_mean: float
    variant_churn_rate_mean: float
    delta_churn_rate_mean: float = Field(
        description="variant minus baseline churn (positive = more churn)."
    )


def override_stances(
    agent_stances: dict[str, Stance], overrides: dict[str, Stance]
) -> dict[str, Stance]:
    """Return a copy of ``agent_stances`` with some agents' stances replaced.

    The original mapping is left untouched. Every agent in ``overrides`` must
    already exist, so a typo cannot silently add a phantom stakeholder.
    """
    updated = dict(agent_stances)
    for agent, stance in overrides.items():
        if agent not in updated:
            raise KeyError(f"Cannot override unknown agent {agent!r}.")
        updated[agent] = stance
    return updated


def with_price_increase(
    business: BusinessContext, price_increase_pct: float
) -> BusinessContext:
    """Return a copy of ``business`` with a different price increase.

    Note: in the current quant model the price-change churn is a fixed assumption,
    not scaled by the size of the rise, so this changes revenue per customer but
    not the churn rate. The behavioural lever is :func:`override_stances` on a
    *customer* agent.
    """
    return business.model_copy(update={"price_increase_pct": price_increase_pct})


def compare_impacts(base: ImpactResult, variant: ImpactResult) -> WhatIfComparison:
    """Compare two :class:`ImpactResult` outcomes into a :class:`WhatIfComparison`."""
    return WhatIfComparison(
        baseline_impact_mean=base.revenue_impact_mean,
        variant_impact_mean=variant.revenue_impact_mean,
        delta_mean=variant.revenue_impact_mean - base.revenue_impact_mean,
        baseline_churn_rate_mean=base.churn_rate_mean,
        variant_churn_rate_mean=variant.churn_rate_mean,
        delta_churn_rate_mean=variant.churn_rate_mean - base.churn_rate_mean,
    )


def run_whatif(
    agent_stances: dict[str, Stance],
    business: BusinessContext,
    *,
    stance_overrides: dict[str, Stance] | None = None,
    price_increase_pct: float | None = None,
    config: MonteCarloConfig | None = None,
    model_base_churn: float | None = None,
    model_price_extra_churn: float | None = None,
) -> WhatIfComparison:
    """Score a baseline and a what-if variant, then compare them.

    The variant applies ``stance_overrides`` (customer agents) and/or a new
    ``price_increase_pct``. Both runs share the same Monte Carlo config (and
    therefore seed), so the comparison is apples-to-apples and deterministic.
    """
    config = config or MonteCarloConfig()
    base_result = simulate_impact(
        agent_stances,
        business,
        config,
        model_base_churn=model_base_churn,
        model_price_extra_churn=model_price_extra_churn,
    )

    variant_stances = (
        override_stances(agent_stances, stance_overrides)
        if stance_overrides
        else agent_stances
    )
    variant_business = (
        with_price_increase(business, price_increase_pct)
        if price_increase_pct is not None
        else business
    )
    variant_result = simulate_impact(
        variant_stances,
        variant_business,
        config,
        model_base_churn=model_base_churn,
        model_price_extra_churn=model_price_extra_churn,
    )
    return compare_impacts(base_result, variant_result)
