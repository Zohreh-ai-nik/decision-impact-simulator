# Dev log

A short diary of what was built, why, and the results.

## 2026-11-05 — Quantitative impact layer (`quant/`)

**Goal:** Turn agent stances into a quantified revenue impact with uncertainty
ranges, the core promise of the project.

**What changed:**
- `src/decision_sim/schemas.py` — added `CustomerGroup`, `BusinessContext`
  (with validators: unique group names, shares sum ≤ 1.0), `ImpactResult`, and
  an optional `business` field on `Scenario`. Extended `Scenario` validation so
  every customer group's `agent` must be a real stakeholder.
- `scenarios/insurer_price_rise.yaml` — added a `business:` section
  (customer_base 500k, annual_premium £650, +12%, one customer group mapped to
  "Loyal Customers").
- `src/decision_sim/quant/mapping.py` — pure `churn_probability(stance)` (bounded,
  monotonic in score/confidence) and `aggregate_stances(...)`.
- `src/decision_sim/quant/monte_carlo.py` — `MonteCarloConfig` and
  `simulate_impact(agent_stances, business, config)` using
  `numpy.random.default_rng(seed)` with per-group binomial churn draws. Only
  customer-group agents contribute; all other stakeholders are ignored by
  construction. Missing customer agent raises `KeyError`.
- `src/decision_sim/simulation/graph.py` — added `final_stance_by_agent(...)`
  adapter so the simulation can hand a plain dict to `quant` without `quant`
  importing `simulation`.
- `tests/test_quant.py` — 15 tests: mapping bounds/monotonicity, Monte Carlo
  reproducibility, percentile ordering, supportive→positive impact,
  hostile>supportive churn, non-customer agents ignored, missing-agent error,
  multi-group, and schema/YAML validation.

**Design choices:**
- `quant/` imports only `schemas` + NumPy — no `simulation`/`ui` — so it can be
  split into a standalone service later.
- Churn cap (`max_churn`, default 0.6) keeps probabilities plausible and in range.
- Business numbers live in `BusinessContext`; `MonteCarloConfig` holds only
  simulation knobs (n_runs, seed, max_churn).

**Commands:** `ruff check src tests`, `pytest -q`.
**Result:** all tests pass, lint clean.
