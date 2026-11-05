"""Monte Carlo estimation of the revenue impact of a decision.

Pure and self-contained: depends only on the schemas, NumPy and the churn
mapping. Nothing here imports from ``simulation`` or ``ui``, so the whole
quantitative layer can be lifted into a standalone service.

Model
-----
Each run first draws two uncertain parameters from the business context:

* ``base_churn``  ~ Normal(``base_churn``, ``base_churn_sd``), clipped to [0, 1]
* ``sensitivity`` ~ Normal(``max_extra_churn``, ``max_extra_churn_sd``), clipped to [0, 1]

so the reported percentile range reflects parameter uncertainty and not only
sampling noise. For every customer group:

* **No-price-rise baseline:** ``Binomial(n, base_churn)`` customers leave and the
  rest pay the *current* premium.
* **With the rise:** on top of those baseline leavers, a further
  ``sensitivity * dissatisfaction`` share of the remaining customers leaves, so
  total churn is ``base_churn + sensitivity * dissatisfaction`` and the rest pay
  the *new* premium.

Baseline and scenario share the same baseline leavers (coupled draws), so the
impact is measured against a like-for-like world.

Only agents that represent customer groups (via
:class:`~decision_sim.schemas.BusinessContext`) influence extra churn. Other
stakeholders -- CEO, Regulator, Press, and so on -- are ignored by construction.
Customers not covered by any group churn at the base rate only.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from decision_sim.quant.mapping import dissatisfaction
from decision_sim.schemas import BusinessContext, ImpactResult, Stance


class MonteCarloConfig(BaseModel):
    """Knobs for the Monte Carlo simulation (business numbers live elsewhere)."""

    model_config = ConfigDict(extra="forbid")

    n_runs: int = Field(default=2000, ge=1)
    seed: int = Field(default=7)


def _sample_bounded(
    rng: np.random.Generator, mean: float, sd: float, size: int
) -> np.ndarray:
    """Draw ``Normal(mean, sd)`` samples clipped to the valid range ``[0, 1]``."""
    return np.clip(rng.normal(mean, sd, size=size), 0.0, 1.0)


def simulate_impact(
    agent_stances: dict[str, Stance],
    business: BusinessContext,
    config: MonteCarloConfig | None = None,
) -> ImpactResult:
    """Estimate the annual revenue impact of the price change via Monte Carlo.

    Args:
        agent_stances: Final stance per agent. Must include every customer
            group's representing agent; non-customer agents are ignored.
        business: The business context (customer base, premium, groups and
            churn calibration).
        config: Monte Carlo settings; defaults are used when ``None``.

    Returns:
        An :class:`ImpactResult` whose impact is measured against the
        no-price-rise baseline, in absolute and percentage terms.

    Raises:
        KeyError: If a customer group's agent has no stance in ``agent_stances``.
    """
    config = config or MonteCarloConfig()
    rng = np.random.default_rng(config.seed)
    n = config.n_runs

    old_premium = business.annual_premium
    new_premium = old_premium * (1.0 + business.price_increase_pct / 100.0)

    # (size, dissatisfaction) per segment; resolve stances up front so a missing
    # agent fails before any random numbers are drawn.
    segments: list[tuple[int, float]] = []
    for group in business.customer_groups:
        if group.agent not in agent_stances:
            raise KeyError(
                f"Customer group {group.name!r} needs a stance for agent "
                f"{group.agent!r}, but none was provided."
            )
        size = int(round(business.customer_base * group.share))
        if size > 0:
            segments.append((size, dissatisfaction(agent_stances[group.agent])))

    ungrouped = business.customer_base - sum(size for size, _ in segments)
    if ungrouped > 0:
        segments.append((ungrouped, 0.0))  # no stance signal -> base churn only

    base_churn = _sample_bounded(rng, business.base_churn, business.base_churn_sd, n)
    sensitivity = _sample_bounded(
        rng, business.max_extra_churn, business.max_extra_churn_sd, n
    )
    # Probability that a customer who survived base churn leaves because of the rise.
    survivors = np.maximum(1.0 - base_churn, 1e-12)

    baseline_revenue = np.zeros(n, dtype=float)
    scenario_revenue = np.zeros(n, dtype=float)
    baseline_lost = np.zeros(n, dtype=float)
    scenario_lost = np.zeros(n, dtype=float)

    for size, dissat in segments:
        base_leavers = rng.binomial(size, base_churn)
        extra_prob = np.clip(sensitivity * dissat / survivors, 0.0, 1.0)
        extra_leavers = rng.binomial(size - base_leavers, extra_prob)
        scenario_leavers = base_leavers + extra_leavers

        baseline_revenue += (size - base_leavers) * old_premium
        scenario_revenue += (size - scenario_leavers) * new_premium
        baseline_lost += base_leavers
        scenario_lost += scenario_leavers

    impact = scenario_revenue - baseline_revenue
    # A run where every baseline customer churns has zero baseline revenue; treat
    # its percentage change as 0 rather than dividing by zero.
    impact_pct = np.divide(
        impact, baseline_revenue, out=np.zeros_like(impact), where=baseline_revenue > 0
    )

    return ImpactResult(
        n_runs=n,
        baseline_revenue=float(baseline_revenue.mean()),
        new_annual_premium=new_premium,
        baseline_churn_rate_mean=float((baseline_lost / business.customer_base).mean()),
        churn_rate_mean=float((scenario_lost / business.customer_base).mean()),
        lost_customers_mean=float(scenario_lost.mean()),
        revenue_impact_mean=float(impact.mean()),
        revenue_impact_std=float(impact.std()),
        revenue_impact_p10=float(np.percentile(impact, 10)),
        revenue_impact_p50=float(np.percentile(impact, 50)),
        revenue_impact_p90=float(np.percentile(impact, 90)),
        revenue_impact_pct_mean=float(impact_pct.mean()),
        revenue_impact_pct_p10=float(np.percentile(impact_pct, 10)),
        revenue_impact_pct_p50=float(np.percentile(impact_pct, 50)),
        revenue_impact_pct_p90=float(np.percentile(impact_pct, 90)),
        samples=impact.tolist(),
    )
