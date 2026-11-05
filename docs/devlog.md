# Dev log

A short diary of what was built, why, and the results.

## 2026-11-05 — Phase 2c: document ingestion

**Goal:** Turn local documents into citable `Evidence` chunks so uploaded files
feed the research layer the same way web results do — with provenance.

**What changed:**
- `src/decision_sim/intelligence/ingest.py` (new, stdlib + Pydantic only):
  - `load_text(path)` — UTF-8 read of `.txt`/`.md`; explicit `ValueError` for
    unsupported types (PDF deferred) and `FileNotFoundError` for missing files.
  - `chunk_text(text, chunk_size=800, overlap=100)` — overlapping character
    windows so a fact straddling a boundary survives; empty text → no chunks;
    validates params.
  - `ingest_file(path)` — one `Evidence` per chunk, each cited with a `file://`
    `Source` (absolute URI, filename as title, chunk preview as snippet).
  - `ingest_dir(dir, pattern="*.md")` — ingest matching files, sorted by name.
  - Document text is treated strictly as **data** (only stored in
    `Evidence.text`, never interpreted).
- `tests/test_intelligence_ingest.py` (new) — 13 offline tests using `tmp_path`:
  load/validation, chunking (short/empty/size+overlap/boundary/bad params),
  ingest_file (one-per-chunk, file:// sources, empty doc), ingest_dir (pattern
  filter + sorted, missing dir).

**Result:** full suite **187 passed**, `ruff` clean.

## 2026-11-05 — Phase 2b: live Tavily search provider

**Goal:** Add a real web provider behind the existing `SearchProvider` protocol,
without making the core install or the test suite depend on the network.

**What changed:**
- `pyproject.toml` — new optional `research = ["tavily-python"]` extra (mirrors
  the `ml` extra). The app runs offline with Fake/Cached; only live research needs
  this extra.
- `src/decision_sim/intelligence/web_search.py` — `TavilySearchProvider`:
  - Key from `api_key=` or `TAVILY_API_KEY` (never logged); raises a clear error
    if absent. `client=` is injectable so tests stub the API; the `tavily` import
    is lazy (only when no client is passed), so core + tests don't need the extra.
  - `search()` maps raw hits to `SearchResult` (`content`→snippet,
    `published_date`→date via a tolerant `_parse_date`, `source="tavily"`), and
    defensively skips malformed hits at this API boundary.
- `tests/test_intelligence_search.py` — 8 new offline tests with a
  `_FakeTavilyClient` stub: mapping, missing date, malformed-hit skipping, empty
  response, `max_results` validation + passthrough, missing-key error, protocol
  conformance, and composition with `CachedSearchProvider`.

**Result:** full suite **174 passed**, `ruff` clean. The network is never hit in
tests.

## 2026-11-05 — Phase 2a: offline-first search providers

**Goal:** Give the research layer a clean, swappable way to fetch real facts
without any network during tests or demos, and let local documents be cited the
same way as web pages.

**What changed:**
- `src/decision_sim/intelligence/web_search.py` (new):
  - `SearchResult` (Pydantic v2, `extra="forbid"`): `title`, `url`, `snippet`,
    optional `published` date, `source`. URL must be `http(s)://` or `file://`.
  - `SearchProvider` — a `runtime_checkable` Protocol: `search(query, *,
    max_results) -> list[SearchResult]`.
  - `FakeSearchProvider` — deterministic, canned results keyed by exact query;
    returns deep copies so fixtures can't be mutated.
  - `CachedSearchProvider` — wraps any provider, persists to
    `data/cache/search/cache.json` (git-ignored). Cache key is
    `(max_results, query)`; a hit never calls the inner provider.
- `src/decision_sim/schemas.py` — `Source.url` now also accepts `file://` so
  uploaded/local documents can be cited alongside web URLs.
- `tests/test_intelligence_search.py` (new) — 15 offline tests: result
  validation (file:// ok, bare URL rejected, date parsing), Fake behaviour
  (canned/unknown/max_results/copies/protocol), Cached behaviour (delegate+write,
  cache hit via a counting spy, persistence across instances, max_results keying),
  and `Source` file:// acceptance.

**Deferred to 2b:** real `TavilySearchProvider` (network, key via `.env`),
implementing the same `SearchProvider` protocol.

**Result:** full suite **166 passed**, `ruff` clean.

## 2026-11-05 — Phase 8e: wire the model into quant + UI, fix the confounding

**Goal:** Use the churn model for the price-change line of the impact breakdown,
but only if its price effect is plausible — and first try to fix the price-change
confounding (bigger rises go to stickier customers, so a plain model learns the
wrong sign) before falling back to the labelled assumption.

**Fixing the confounding (in the model):**
- `ml/data.py` — added `price_vs_technical = prem_final / prem_pure - 1` (margin
  over the technical premium) so the model can tell "risky customer" from "charged
  a lot above risk"; drops non-positive `prem_pure`.
- `ml/train.py` — `HistGradientBoostingClassifier(monotonic_cst=...)` forcing
  `price_change`, `price_vs_market`, `price_vs_technical` to only increase churn
  (via `set_output(transform="pandas")` + transformed column names). `monotonic`
  flag threaded through `train_model`.
- `notebooks/01_churn_eda.ipynb` §7 — trains unconstrained vs constrained and
  prints the comparison with a plain-words conclusion.

**Real results (seed 7, 5,000-customer portfolio, 12% rise):** discrimination
basically unchanged (ROC AUC 0.611 vs 0.613; Brier ~0.110 both), but the mean
price-delta flips from **−0.3pp** (unconstrained, 54% positive) to **+6.6pp**
(constrained, 98% positive). Plausible and within the 15pp cap → **the model is
used** on the price line, with the assumption as fallback.

**Wiring (the 4-line breakdown):**
- `ml/predict.py` — `churn_inputs_from_model(...)` and `churn_inputs_or_none(...)`:
  return `(base_churn, price_extra_churn)` as plain floats, or `(None, None)` when
  the model/data is missing or the effect is implausible. Keeps `quant/` decoupled
  from `ml/` (floats cross the boundary, not objects).
- `schemas.py` — `ImpactResult` gains `price_extra_churn_rate_mean` and signed
  `agent_adjustment_churn_rate_mean`.
- `quant/monte_carlo.py` — `simulate_impact(..., model_base_churn=None,
  model_price_extra_churn=None)`. Extra churn splits into a **price component**
  (model constant, clipped to `max_extra_churn`) and a signed **agent component**
  (`sensitivity * (dissatisfaction - 0.5)`). The three parts (base + price + agent)
  sum to the total exactly, per run.
- `quant/formatting.py` — `churn_breakdown_rows(result, price_from_model=...)`
  returns the 4 lines: base / price (model|assumption) / agent / total.
- `scripts/run_scenario.py` and `ui/app.py` — compute the two floats via
  `churn_inputs_or_none`, pass them in, and show a status note (no model /
  implausible / using model +X%).

**End-to-end check (hostile stance, 12% rise):** base 39.9% + price (model) 6.3% +
agent +2.7% = **48.9%** total — additive as designed.

**Tests:** +data/train/predict/quant/formatting tests. Full suite **151 passed**,
`ruff` clean. Docs: `docs/calibration.md` updated with the confounding fix and the
decision to use the model.

## 2026-11-05 — Phase 8d: per-customer churn prediction

**Goal:** Turn the trained model into calibrated, per-customer churn estimates
without letting the European dataset's absolute lapse level (~12.8%) override the
FCA UK base (0.40).

**What changed:**
- `src/decision_sim/ml/predict.py` — the prediction layer:
  - `build_portfolio(features, n, seed)` — fictional ShieldDrive portfolio by
    seeded resampling of dataset rows.
  - `rescale_to_base(probs, base=0.40)` — a **monotonic logit shift** (found by
    bisection) so the portfolio mean equals the FCA base while keeping the
    relative risk ordering; result stays in [0, 1].
  - `predict_base_churn(model, portfolio, base)` — model risk ranking rescaled to
    the FCA base (relative risk, not absolute level).
  - `apply_price_increase(portfolio, pct)` — raises `prem_final` and recomputes
    the price ratios (`prem_last`/`prem_market` unchanged by a renewal rise).
  - `predict_price_effect(model, portfolio, pct)` — per-customer `pred(new) -
    pred(old)`, returned **unclipped** so the caller can judge plausibility.
- `tests/test_ml_predict.py` — 11 offline tests (portfolio size/determinism,
  rescale hits target mean + preserves ordering + stays in range, base churn ==
  FCA mean, price-increase scaling and zero-identity, price-effect shape/zero).

**Real results (HGB, portfolio of 5,000 resampled rows, seed 7):**
- Base churn mean = **0.4000** (= FCA target), spread **0.16–0.86** across
  customer types — the relative-risk rescaling works as intended.
- Price effect @12%: mean **−0.0128**, only **36%** of customers positive (range
  −0.43 to +0.23). **The model's price-delta is negative on average** — the same
  confounding found in 8c (bigger rises went to stickier customers). So the raw
  model price effect is **not plausible**; Phase 8e will fall back to the labelled
  `max_extra_churn` (0.15) assumption and upper bound, as planned.

**Validation:** `ruff check` clean; `pytest -q` → **136 passed**.

## 2026-11-05 — Phase 8c: churn model training, evaluation and EDA

**Goal:** Train the churn model on `eudirectlapse`, evaluate it honestly, and
document the price-change confounding before wiring the model into the simulator.

**What changed:**
- `src/decision_sim/ml/train.py` — a preprocessing + estimator `Pipeline` for a
  `LogisticRegression` baseline and a `HistGradientBoostingClassifier` main model
  (dense one-hot for the tree, which rejects sparse input). Stratified split
  (seed 7), ROC AUC + Brier, `save_model`/`load_model` (joblib), `train_and_save`.
- `src/decision_sim/ml/tracking.py` — a thin, **optional** MLflow logger: a
  `start_run` context manager that uses MLflow if installed and enabled, else a
  no-op. Tests never need MLflow. Full tracking is deferred to Phase 12.
- `scripts/train_churn.py` — trains both models on `data/raw/`, prints metrics,
  saves the HGB model to `models/` (now git-ignored).
- `notebooks/01_churn_eda.ipynb` — lapse by age, tenure, number of policies, price
  change and price-vs-market, plus an honest write-up of the confounding.
- `tests/test_ml_train.py` — 9 offline tests (metrics in range, determinism,
  save/load round-trip, missing-model error, no-op tracking, MLflow-absent
  fallback), all on the tiny fixture.
- `.gitignore` — added `models/`, `mlruns/`, `mlartifacts/`.

**Real results (23,060 rows, 25% hold-out, seed 7):**
- LogReg: ROC AUC **0.624**, Brier **0.110**. HGB: ROC AUC **0.620**, Brier
  **0.109**. Modest — churn is genuinely hard here.
- **Confounding confirmed:** price *cut* → ~16% lapse; price *10%+ rise* → ~10%
  lapse; `corr(price_change, lapse) ≈ -0.02`. Bigger rises go to stickier
  customers. Clean, monotonic signals do exist for age (<30 ~17% vs 60+ ~9%) and
  tenure (6+ years ~8% vs 1-2 years ~15%).
- **Decision:** keep the labelled `max_extra_churn` assumption (0.15) as the
  fallback and upper bound for price-driven extra churn; use the model's
  price-delta only if positive/plausible (wired in 8d/8e).

**Results:** `ruff check` clean; `pytest -q` → 125 passed. Dataset and model stay
git-ignored.

## 2026-11-05 — Phase 8a/8b: churn dataset + feature engineering

**Goal:** Start the real churn model. Acquire the public `eudirectlapse` dataset
and build a pure, tested data-prep layer. The model will supply *relative* churn
risk and the price-change effect only — never the absolute UK level (this is a
European direct insurer, ~12.8% lapse), which stays pinned to the FCA 0.40 base.

**What changed:**
- `pyproject.toml` — added `scikit-learn` + `pyreadr` as core deps (scikit-learn
  also pulls in scipy, fixing the earlier networkx pagerank import); added an
  optional `ml` extra (`mlflow`, `matplotlib`) for the notebook and optional
  experiment logging, kept out of the default install and test path.
- `scripts/download_data.py` — downloads the `.rda` to `data/raw/` (git-ignored),
  converts to parquet via pyreadr, and asserts the shape. Ran it:
  **23,060 rows × 19 columns** confirmed; lapse rate 12.81% (matches the docs).
- `src/decision_sim/ml/data.py` — pure cleaning + feature engineering (pandas
  only): `load_dataset`, `clean` (validates columns, binary target, drops
  non-positive premiums), `build_features` (adds `price_change =
  prem_final/prem_last - 1` and `price_vs_market = prem_final/prem_market - 1`),
  `split_xy`, `prepare`. Feature sets exported as constants for the trainer.
- `tests/fixtures/eudirectlapse_tiny.csv` — a 40-row **synthetic** fixture with
  the real schema (no real dataset rows redistributed; dataset is GPL and not
  committed).
- `tests/test_ml_data.py` — 11 tests covering loading, cleaning, ratio maths,
  column selection, immutability and validation. Fully offline.

**Results:** `ruff check` clean; `pytest -q` → 116 passed. `data/` and the trained
model dir stay git-ignored.

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
