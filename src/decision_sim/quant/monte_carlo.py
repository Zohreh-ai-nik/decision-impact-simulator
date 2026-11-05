"""Monte Carlo estimation of the revenue impact of a decision.

Pure and self-contained: depends only on the schemas, NumPy and the churn
mapping. Nothing here imports from ``simulation`` or ``ui``, so the whole
quantitative layer can be lifted into a standalone service.

Only agents that represent customer groups (via
:class:`~decision_sim.schemas.BusinessContext`) contribute to churn. Any other
stakeholders -- CEO, Regulator, Press, and so on -- are ignored by construction
because they are never referenced by a customer group.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from decision_sim.quant.mapping import DEFAULT_MAX_CHURN, churn_probability
from decision_sim.schemas import BusinessContext, ImpactResult, Stance


class MonteCarloConfig(BaseModel):
    """Knobs for the Monte Carlo simulation (business numbers live elsewhere)."""

    model_config = ConfigDict(extra="forbid")

    n_runs: int = Field(default=2000, ge=1)
    seed: int = Field(default=7)
    max_churn: float = Field(default=DEFAULT_MAX_CHURN, ge=0.0, le=1.0)


def simulate_impact(
    agent_stances: dict[str, Stance],
    business: BusinessContext,
    config: MonteCarloConfig | None = None,
) -> ImpactResult:
    """Estimate the annual revenue impact of the price change via Monte Carlo.

    For each customer group the representing agent's final stance is mapped to a
    churn probability; the number of churning customers is sampled with a
    binomial draw across ``n_runs``. Revenue impact is the post-change revenue
    from retained customers minus the baseline revenue.

    Args:
        agent_stances: Final stance per agent. Must include every customer
            group's representing agent; non-customer agents are ignored.
        business: The business context (customer base, premium, groups).
        config: Monte Carlo settings; defaults are used when ``None``.

    Returns:
        An :class:`ImpactResult` with mean/percentile revenue impact and churn.

    Raises:
        KeyError: If a customer group's agent has no stance in ``agent_stances``.
    """
    config = config or MonteCarloConfig()
    rng = np.random.default_rng(config.seed)

    baseline_revenue = business.customer_base * business.annual_premium
    new_premium = business.annual_premium * (1.0 + business.price_increase_pct / 100.0)

    retained_revenue = np.zeros(config.n_runs, dtype=float)
    total_churned = np.zeros(config.n_runs, dtype=float)

    for group in business.customer_groups:
        if group.agent not in agent_stances:
            raise KeyError(
                f"Customer group {group.name!r} needs a stance for agent "
                f"{group.agent!r}, but none was provided."
            )
        group_size = int(round(business.customer_base * group.share))
        if group_size <= 0:
            continue

        prob = churn_probability(agent_stances[group.agent], max_churn=config.max_churn)
        churned = rng.binomial(group_size, prob, size=config.n_runs)
        retained = group_size - churned

        total_churned += churned
        retained_revenue += retained * new_premium

    revenue_impact = retained_revenue - baseline_revenue
    churn_rate = total_churned / business.customer_base

    return ImpactResult(
        n_runs=config.n_runs,
        baseline_revenue=baseline_revenue,
        new_annual_premium=new_premium,
        churn_rate_mean=float(churn_rate.mean()),
        lost_customers_mean=float(total_churned.mean()),
        revenue_impact_mean=float(revenue_impact.mean()),
        revenue_impact_std=float(revenue_impact.std()),
        revenue_impact_p10=float(np.percentile(revenue_impact, 10)),
        revenue_impact_p50=float(np.percentile(revenue_impact, 50)),
        revenue_impact_p90=float(np.percentile(revenue_impact, 90)),
        samples=revenue_impact.tolist(),
    )
