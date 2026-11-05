# Dev log

A short diary of what was built, why, and the results.

## 2026-11-05 — Phase 7: calibrate base churn to FCA EP25/2, split base/extra/total

**Goal:** Replace the placeholder 8% base churn with a figure grounded in a
verified public source, and make the UI distinguish churn that happens anyway
from churn caused by the decision. No numbers were invented; every figure carries
its quote, URL, publisher and date.

**What changed:**
- `schemas.py` — `BusinessContext.base_churn` now 0.40 (UK motor attrition, FCA
  EP25/2 p.44) with a narrow `base_churn_sd` 0.03 and new hard clip bounds
  `base_churn_min` 0.33 / `base_churn_max` 0.54 (the with-/without-autorenewal
  subgroups, used as bounds not as uncertainty). `max_extra_churn` kept at 0.15
  but relabelled an ASSUMPTION. Added `sources: list[Source]` and a validator that
  rejects `base_churn_min > base_churn_max`. `ImpactResult` gains
  `extra_churn_rate_mean` (total minus baseline).
- `quant/monte_carlo.py` — sampled base churn is clipped to `[min, max]`;
  `_sample_bounded` takes `lo`/`hi`. Baseline and scenario share the same sampled
  base churn (common random numbers), so revenue impact reflects only the
  decision. Returns the new extra-churn mean.
- `quant/formatting.py` — `churn_assumption_text` now states the FCA base, the
  clip range and the ASSUMPTION label; new `churn_breakdown_rows` returns the
  three lines (base / extra / total); summary rows renamed to "Mean total churn
  rate" and add the extra-churn row.
- `ui/app.py` — results page shows the three churn lines as metrics.
- `scenarios/insurer_price_rise.yaml` — business block updated to the calibrated
  values with a `sources:` block (FCA EP25/2 + eudirectlapse) and comments
  marking the assumption.
- `docs/calibration.md`, `docs/data.md` — new: the sourcing rationale (verbatim
  FCA quotes with page numbers; why the subgroup figures are bounds not spread;
  why the European dataset is patterns-only) and the dataset licence/provenance.
- `tests/test_quant.py`, `tests/test_formatting.py` — updated churn-level
  expectations to the 0.40 base; new tests for the clip bounds, `min > max`
  rejection, the cited sources, the calibration drift-guard, and the three-line
  breakdown.

**Results:** `ruff check` clean; `pytest -q` → 105 passed. Fully offline, no new
dependencies.

## 2026-11-05 — Phase 1: provenance schemas (Source + Evidence)

**Goal:** Lay the foundation for the intelligence layer — a provenance vocabulary
so every real fact can carry where it came from. Nothing is populated yet; this
is just the plumbing that Phases 2–7 fill in.

**What changed:**
- `schemas.py` — new `Source` (url validated to start with http(s), title, optional
  `date`, snippet) and `Evidence` (text + >=1 `Source`). Added a non-breaking
  `sources: list[Source] = []` to `AgentProfile`, `Post` and `Stance` so messages
  and reasoning can cite evidence later. `date` imported as `datetime.date` to
  avoid shadowing the field named `date`.
- `tests/test_sources.py` — URL validation, optional date/snippet, JSON round-trip
  preserving the date, `Evidence` requires >=1 source, and back-compat (existing
  models still validate with no `sources`).

**Results:** `ruff check` clean; `pytest -q` → 100 passed. Fully offline, no new
dependencies.

## 2026-11-05 — Recalibrated churn, parameter uncertainty and no-rise baseline

**Problem:** The first quant model gave ~40.8% churn from a 12% price rise (churn
was a single `max_churn=0.6` cap scaled by stance) and a p10–p90 range of only
~0.6%, because the only randomness was binomial sampling noise on a fixed
probability. Both were unrealistic.

**What changed:**
- `schemas.py` — `BusinessContext` gains `base_churn` (8%), `base_churn_sd` (2pp),
  `max_extra_churn` (15pp) and `max_extra_churn_sd` (4pp), all validated to
  sensible ranges. `ImpactResult` gains `baseline_churn_rate_mean` and
  `revenue_impact_pct_{mean,p10,p50,p90}`; `baseline_revenue` is now the mean
  revenue in the *no price rise* world.
- `quant/mapping.py` — replaced the single cap with
  `total_churn = base_churn + max_extra_churn * dissatisfaction`
  (`extra_churn_probability`, `total_churn_probability`), so a hostile, confident
  customer group tops out at 8% + 15pp = 23%.
- `quant/monte_carlo.py` — each run samples `base_churn` and the extra-churn
  sensitivity from clipped Normal distributions (parameter uncertainty). A
  no-price-rise baseline (base churn, old premium) is simulated alongside the
  scenario with *coupled* draws: scenario leavers = baseline leavers + extra
  leavers, so the impact compares like with like. Customers not in any group
  churn at the base rate only (previously they were silently counted as lost).
  `MonteCarloConfig.max_churn` removed; calibration now lives in the YAML.
- `quant/formatting.py` — £ / £m / signed-% formatters, `impact_histogram(...,
  in_millions=True)` for a £m axis, `churn_assumption_text(business)`; summary
  rows show £ and % change versus baseline.
- `ui/app.py` — impact panel shows £m and % per percentile, a caption with the
  churn assumptions used, and a histogram labelled "Revenue impact (£m)".
- `scenarios/insurer_price_rise.yaml` — churn calibration added to `business:`.
- Tests — mapping bounds/monotonicity, caps hold for extreme stances, wider
  spread when uncertainty > 0, same seed reproducible, zero impact when no rise
  and a supportive stance, baseline coupling, ungrouped customers, formatting.

**Results (bundled scenario, example stances, 2,000 runs, seed 7):**
- Hostile customers (score -0.8, confidence 0.9): churn ~20.1% (baseline ~7.9%),
  median impact -£8.3m (-2.7%), p10–p90 -£23.4m to +£6.7m (-7.8% to +2.3%).
- Neutral customers (score 0.0, confidence 0.7): churn ~13.2%, median impact
  +£16.9m (+5.6%), p10–p90 +£10.3m to +£23.3m.
- `ruff check` clean; `pytest -q` → 86 passed.

**Note:** The calibration values (8%, 15pp and their spreads) are illustrative
assumptions for a fictional insurer, not measured data; they are editable in the
YAML.

## 2026-11-05 — Activity metrics, run report, and UI analysis panel

**Goal:** Surface per-agent activity and bundle coalitions + influence +
activity into one consolidated, serialisable report, then show it in the UI.

**What changed:**
- `src/decision_sim/analysis/activity.py` — `activity_counts` (per agent,
  excludes the `Breaking News` pseudo-agent), `channel_counts` (per channel,
  all posts), `activity_dataframe` (tidy `agent`/`posts` frame sorted by volume).
- `src/decision_sim/analysis/report.py` — `RunReport` (Pydantic, `extra="forbid"`)
  and `build_report(posts, agent_stances, scenario)` composing coalitions,
  influence, activity, and mean stance into one object. Stays pure (schemas +
  analysis/quant only; no `simulation`/`ui` imports).
- `src/decision_sim/analysis/__init__.py` — export the new activity + report API.
- `src/decision_sim/ui/app.py` — new `_render_analysis` panel (coalitions list,
  influence bar chart, activity bar chart), called after `_render_impact`.
- `tests/test_report.py` — activity counts exclude Breaking News, dataframe
  shape/empty, and `build_report` aggregation + serialisation.

**Results:** `ruff check` clean; `pytest -q` → 69 passed.

## 2026-11-05 — Analysis layer (`analysis/`: coalitions + influence)

**Goal:** Add "who aligns with whom" (coalitions) and "who is most influential"
(influence ranking) on top of the simulation output.

**What changed:**
- `src/decision_sim/analysis/coalitions.py` — `final_scores(...)` and
  `detect_coalitions(agent_stances, threshold)`: links same-side agents whose
  final stance scores are within `threshold`, returns connected components
  (NetworkX), sorted deterministically.
- `src/decision_sim/analysis/influence.py` — `build_interaction_graph(posts,
  scenario)` builds a reach-weighted information-flow graph (poster -> every
  reader of the channel); `influence_ranking(...)` runs a pure-Python
  power-iteration PageRank on the *reversed* graph so agents read by many score
  highest. `Breaking News` is excluded.
- `src/decision_sim/analysis/__init__.py` — package exports.
- `tests/test_analysis.py` — 11 tests: coalition grouping/threshold/opposition,
  influence ordering, public-vs-private reach, Breaking News exclusion, empty
  case.

**Design choices:**
- `analysis/` depends only on `schemas` + NetworkX (no `simulation`/`ui`),
  matching the decoupled `quant/` approach.
- Implemented PageRank by hand (power iteration) to avoid pulling in SciPy, which
  `nx.pagerank` now requires. Edge direction: `build_interaction_graph` models
  information flow (poster -> reader); influence reverses it so "read by
  influential agents" raises your score.

**Commands:** `ruff check src tests`, `pytest -q`.
**Result:** 63 tests pass, lint clean.

## 2026-11-05 — Impact in the UI (`quant/formatting` + Streamlit)

**Goal:** Show the quantified impact at the end of a run in the Streamlit app.

**What changed:**
- `src/decision_sim/quant/formatting.py` — pure, NumPy-only helpers:
  `impact_histogram(result, bins)` (bin centres + counts) and
  `impact_summary_rows(result)` (formatted metric rows). No simulation/ui imports.
- `src/decision_sim/ui/app.py` — after a run, if the scenario has a `business`
  section, computes impact via `final_stance_by_agent(...)` ->
  `simulate_impact(...)` and renders p10/p50/p90 metrics, a summary list, and a
  histogram (`st.bar_chart`). Logic stays in `src`; the UI only renders.
- `tests/test_formatting.py` — 4 tests: histogram lengths/counts, empty samples,
  bad bins, and summary-row content (incl. percentage formatting).

**Design choices:**
- Formatting helpers return plain lists/tuples so they are trivially testable and
  keep `quant` free of UI concerns.

**Commands:** `ruff check src tests`, `pytest -q`.
**Result:** 54 tests pass, lint clean, app imports cleanly.

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
