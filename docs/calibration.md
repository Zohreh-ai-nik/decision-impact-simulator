# Calibration

How the quantitative churn parameters are chosen, and the public sources behind
them. **Rule:** every figure records the number, the exact quote, the URL, the
publisher and the date. Numbers are never invented. Where a value is an
assumption rather than a sourced figure, it says so explicitly.

The company (ShieldDrive Insurance) is fictional; only the *assumptions* are
grounded in public data.

## Parameters at a glance

| Parameter | Value | Basis |
|---|---|---|
| `base_churn` | 0.40 | FCA EP25/2 — UK motor attrition overall, 2022 |
| `base_churn_sd` | 0.03 | Narrow: overall rate was stable at 39–40% |
| `base_churn_min` / `base_churn_max` | 0.33 / 0.54 | FCA EP25/2 — motor with- / without-autorenewal subgroup bounds |
| `max_extra_churn` | 0.15 | **ASSUMPTION — not yet sourced.** Refined in Phase 8 |
| `max_extra_churn_sd` | 0.04 | **ASSUMPTION — not yet sourced.** Refined in Phase 8 |

## Base churn — UK motor (primary source)

**Value:** `base_churn = 0.40`, with a deliberately **narrow** `base_churn_sd =
0.03` and **hard clip bounds** `[0.33, 0.54]`.

**Why narrow, with subgroup bounds instead of a wide spread.** The FCA reports the
*overall* motor attrition rate was stable at 39–40% across the evaluation window.
The 33–35% and 49–54% figures are **two subgroups** (with and without
autorenewal), **not** uncertainty about the overall rate. So the overall rate is
modelled with a small standard deviation (±3pp), while the subgroup figures are
used only as **hard lower/upper clips** on the sampled value — a sampled base
churn can never fall below the most-sticky subgroup (33%) or above the
least-sticky subgroup (54%).

**Source.** Financial Conduct Authority (FCA), *Evaluation Paper 25/2: An
evaluation of our GIPP remedies*. First published 22/07/2025.
PDF: https://www.fca.org.uk/publication/corporate/ep25-2.pdf
Landing: https://www.fca.org.uk/publications/corporate-documents/evaluation-paper-25-2-general-insurance-pricing-practices-remedies

Exact quotes:

> "between 2020 and 2022 the home attrition rate overall fell modestly from 26% to
> 25%, while the motor attrition rate overall rose modestly over the same period
> from 39% to 40%." — p.44

> "Calculating the same rates for motor policies gives a range for those policies
> without autorenewal to be 49%-54%, and those policies with autorenewal to be
> 33%-35%." — p.44

Supporting context on the shift to shopping around (motor):

> "Price Comparison Website (PCW) sales rose from 60% to 66%, while direct sales
> fell from 24% to 19%, showing a strong consumer preference for aggregator
> channels." — p.24

Note: the paper also reports a *home* PCW figure ("Sales via Price Comparison
Websites (PCWs) increased from 35% to 39%", p.21). We use the **motor** figure
(60%→66%) only; the home figure is not used.

## Base churn — European direct insurer (secondary source, patterns only)

**Dataset.** `eudirectlapse` (CASdatasets). Confirmed from the reference page: "a
data frame of 19 columns and 23,060 rows", a binary `lapse` target, and premium
variables `prem_final`, `prem_last`, `prem_market`.
URL: https://dutangc.github.io/CASdatasets/reference/eudirectlapse.html

**Derived figure:** overall lapse ≈ 12.8% (computed from the data, not printed on
the page).

**How it is used.** This is an *unknown European direct insurer*, not the UK
market, and its 12.8% lapse is far below the UK motor ~40% attrition. We therefore
do **not** use it for the UK base churn level. It is used in Phase 8 for the
*patterns* of churn — how lapse varies with driver age, policy tenure, number of
policies, price change and price vs market — to train the churn model. Licence and
provenance for the dataset are recorded in `docs/data.md`.

## Extra churn from the decision — ASSUMPTION (not sourced)

`max_extra_churn` (0.15) and `max_extra_churn_sd` (0.04) are the additional churn a
fully hostile customer stance adds on top of base churn. These are **not grounded
in a public figure yet** and are labelled as assumptions in the scenario YAML and
in the Streamlit UI. They will be estimated in Phase 8 from the churn model, using
price change and price-vs-market features with the confounding handled (the raw
data shows lapse does not rise with price change, likely because larger rises are
given to stickier customers).

## How churn is combined

Total churn = **base churn** (happens anyway, ~40%) + **extra churn caused by the
decision** (from customer stance, capped). Each Monte Carlo run uses common random
numbers: the no-price-rise baseline and the price-rise scenario share the same
sampled base churn and the same baseline leavers, so the reported revenue impact
reflects only the effect of the decision. Revenue impact = price-rise scenario
minus no-rise baseline, over the same runs.
