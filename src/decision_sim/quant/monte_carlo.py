"""Monte Carlo estimation of the revenue impact of a decision.

Pure and self-contained: depends only on the schemas, NumPy and the churn
mapping. Nothing here imports from ``simulation`` or ``ui``, so the whole
quantitative layer can be lifted into a standalone service.

Model
-----
Each run first draws two uncertain parameters from the business context:

* ``base_churn``  ~ Normal(``base_churn``, ``base_churn_sd``), clipped to
  ``[base_churn_min, base_churn_max]`` (hard subgroup bounds, not just [0, 1])
* ``sensitivity`` ~ Normal(``max_extra_churn``, ``max_extra_churn_sd``), clipped to [0, 1]

so the reported percentile range reflects parameter uncertainty and not only
sampling noise. For every customer group, extra churn (on top of base churn) is
split into two additive parts, both expressed as a fraction of the group size:

* **Price-change extra** -- the market reaction to the rise itself. If a trained
  model supplies a per-portfolio estimate (``model_price_extra_churn``), that is
  used (clipped to ``[0, max_extra_churn]``); otherwise it falls back to the
  labelled assumption ``sensitivity * 0.5`` (a neutral-stance reaction).
* **Agent adjustment** -- how the simulated discourse pushes churn above or below
  that neutral reaction: ``sensitivity * (dissatisfaction - 0.5)``. It is positive
  for hostile stances and **negative** for supportive ones.

Their sum is the total extra churn; with the assumption fallback this collapses to
the original ``sensitivity * dissatisfaction``, so nothing changes when no model is
used.

* **No-price-rise baseline:** ``Binomial(n, base_churn)`` customers leave and the
  rest pay the *current* premium.
* **With the rise:** on top of those baseline leavers, a further
  ``price + agent`` share of the remaining customers leaves, and the rest pay the
  *new* premium.

Baseline and scenario use **common random numbers**: each run shares the same
sampled ``base_churn`` and the same baseline leavers, so the revenue impact
reflects only the effect of the decision, not Monte Carlo noise.

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
    rng: np.random.Generator,
    mean: float,
    sd: float,
    size: int,
    *,
    lo: float = 0.0,
    hi: float = 1.0,
) -> np.ndarray:
    """Draw ``Normal(mean, sd)`` samples clipped to ``[lo, hi]``."""
    return np.clip(rng.normal(mean, sd, size=size), lo, hi)


def simulate_impact(
    agent_stances: dict[str, Stance],
    business: BusinessContext,
    config: MonteCarloConfig | None = None,
    *,
    model_base_churn: float | None = None,
    model_price_extra_churn: float | None = None,
) -> ImpactResult:
    """Estimate the annual revenue impact of the price change via Monte Carlo.

    Args:
        agent_stances: Final stance per agent. Must include every customer
            group's representing agent; non-customer agents are ignored.
        business: The business context (customer base, premium, groups and
            churn calibration).
        config: Monte Carlo settings; defaults are used when ``None``.
        model_base_churn: Optional model-derived mean base churn (already rescaled
            to the UK base). Defaults to ``business.base_churn`` when ``None``.
        model_price_extra_churn: Optional model-derived extra churn from the price
            change (fraction of customers). When ``None`` the labelled assumption
            ``sensitivity * 0.5`` is used instead.

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

    base_churn_mean = business.base_churn if model_base_churn is None else model_base_churn
    base_churn = _sample_bounded(
        rng,
        base_churn_mean,
        business.base_churn_sd,
        n,
        lo=business.base_churn_min,
        hi=business.base_churn_max,
    )
    sensitivity = _sample_bounded(
        rng, business.max_extra_churn, business.max_extra_churn_sd, n
    )
    # Fraction of customers surviving base churn; converts "extra as a fraction of
    # the whole group" into "probability for a base-churn survivor".
    survivors = np.maximum(1.0 - base_churn, 1e-12)

    baseline_revenue = np.zeros(n, dtype=float)
    scenario_revenue = np.zeros(n, dtype=float)
    baseline_lost = np.zeros(n, dtype=float)
    scenario_lost = np.zeros(n, dtype=float)
    price_lost = np.zeros(n, dtype=float)
    agent_lost = np.zeros(n, dtype=float)

    for size, dissat in segments:
        # Two additive extra-churn components, as a fraction of the whole segment.
        if model_price_extra_churn is None:
            price_component = sensitivity * 0.5  # neutral-stance assumption
        else:
            price_component = np.full(n, float(model_price_extra_churn))
        price_component = np.clip(price_component, 0.0, business.max_extra_churn)
        agent_component = sensitivity * (dissat - 0.5)  # signed discourse effect

        # Convert to per-survivor probabilities and draw leavers (common randoms).
        price_prob = price_component / survivors
        agent_prob = agent_component / survivors
        combined_prob = np.clip(price_prob + agent_prob, 0.0, 1.0)

        base_leavers = rng.binomial(size, base_churn)
        survivor_count = size - base_leavers
        extra_leavers = rng.binomial(survivor_count, combined_prob)

        # Split the drawn extra leavers between price and agent by their signed
        # expected contributions; the agent part can be negative (net supportive).
        exp_price = survivor_count * price_prob
        exp_agent = survivor_count * agent_prob
        denom = exp_price + exp_agent
        price_ratio = np.divide(
            exp_price, denom, out=np.zeros_like(exp_price), where=denom > 0.0
        )
        seg_price = extra_leavers * price_ratio
        seg_agent = extra_leavers - seg_price

        scenario_leavers = base_leavers + extra_leavers
        baseline_revenue += (size - base_leavers) * old_premium
        scenario_revenue += (size - scenario_leavers) * new_premium
        baseline_lost += base_leavers
        scenario_lost += scenario_leavers
        price_lost += seg_price
        agent_lost += seg_agent

    impact = scenario_revenue - baseline_revenue
    # A run where every baseline customer churns has zero baseline revenue; treat
    # its percentage change as 0 rather than dividing by zero.
    impact_pct = np.divide(
        impact, baseline_revenue, out=np.zeros_like(impact), where=baseline_revenue > 0
    )

    base = business.customer_base
    baseline_churn_rate = baseline_lost / base
    total_churn_rate = scenario_lost / base
    price_extra_rate = price_lost / base
    agent_adjust_rate = agent_lost / base

    return ImpactResult(
        n_runs=n,
        baseline_revenue=float(baseline_revenue.mean()),
        new_annual_premium=new_premium,
        baseline_churn_rate_mean=float(baseline_churn_rate.mean()),
        churn_rate_mean=float(total_churn_rate.mean()),
        extra_churn_rate_mean=float((total_churn_rate - baseline_churn_rate).mean()),
        price_extra_churn_rate_mean=float(price_extra_rate.mean()),
        agent_adjustment_churn_rate_mean=float(agent_adjust_rate.mean()),
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
