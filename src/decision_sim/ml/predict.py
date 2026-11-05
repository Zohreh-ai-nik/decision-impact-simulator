"""Turn the trained churn model into per-customer churn estimates.

The model (:mod:`decision_sim.ml.train`) is trained on a European direct insurer
whose absolute lapse rate (~12.8%) does not fit the UK. So we use it in two
calibrated ways, per the project design (see ``docs/calibration.md``):

1. **Relative risk.** Rank customers by predicted lapse, then *rescale* so the
   portfolio's mean churn equals the FCA UK base (default 0.40), preserving the
   differences by age, tenure, number of policies, etc. The rescaling is a
   monotonic logit shift, so it never changes the ordering and stays in [0, 1].
2. **Price-change effect.** For the same customer, predict at the new price minus
   at the old price. Because the raw data is confounded (bigger rises went to
   stickier customers), this effect can be weak or negative; the caller decides
   whether to trust it or fall back to the labelled ``max_extra_churn`` assumption.

A fictional portfolio is built by seeded resampling of dataset rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from decision_sim.ml.data import FEATURES

# UK motor base churn (FCA EP25/2). Mirrors BusinessContext.base_churn default.
FCA_BASE_CHURN = 0.40

# Features whose value changes when the premium is raised.
_PRICE_LEVEL = "prem_final"
_PRICE_CHANGE = "price_change"
_PRICE_VS_MARKET = "price_vs_market"
_PRICE_VS_TECHNICAL = "price_vs_technical"


def build_portfolio(features: pd.DataFrame, n: int, *, seed: int = 7) -> pd.DataFrame:
    """Build a fictional portfolio by resampling dataset rows with replacement.

    Args:
        features: A feature frame (output of ``data.build_features``/``prepare``).
        n: Number of customers in the portfolio.
        seed: Seed for deterministic resampling.

    Returns:
        A frame of ``n`` rows containing at least the model ``FEATURES``.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(features), size=n)
    return features.iloc[idx].reset_index(drop=True)


def _predict_proba(model: Pipeline, portfolio: pd.DataFrame) -> np.ndarray:
    """Return the model's probability of lapse for each row."""
    x = portfolio[FEATURES]
    return model.predict_proba(x)[:, 1]


def rescale_to_base(probs: np.ndarray, base: float = FCA_BASE_CHURN) -> np.ndarray:
    """Shift probabilities so their mean equals ``base``, preserving ordering.

    Applies a constant shift in logit space found by bisection. Monotonic, so the
    relative ranking of customers is unchanged, and the result stays in [0, 1].
    """
    if not 0.0 <= base <= 1.0:
        raise ValueError(f"base must be in [0, 1], got {base}.")
    probs = np.asarray(probs, dtype=float)
    if probs.size == 0:
        return probs

    eps = 1e-6
    clipped = np.clip(probs, eps, 1.0 - eps)
    logit = np.log(clipped / (1.0 - clipped))

    def mean_at(shift: float) -> float:
        return float((1.0 / (1.0 + np.exp(-(logit + shift)))).mean())

    lo, hi = -50.0, 50.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if mean_at(mid) < base:
            lo = mid
        else:
            hi = mid
    shift = (lo + hi) / 2.0
    return 1.0 / (1.0 + np.exp(-(logit + shift)))


def predict_base_churn(
    model: Pipeline, portfolio: pd.DataFrame, *, base: float = FCA_BASE_CHURN
) -> np.ndarray:
    """Per-customer base churn: model risk ranking rescaled to the FCA base."""
    return rescale_to_base(_predict_proba(model, portfolio), base)


def apply_price_increase(portfolio: pd.DataFrame, price_increase_pct: float) -> pd.DataFrame:
    """Return a copy of the portfolio with the premium raised by the given %.

    Raises ``prem_final`` and recomputes the price ratios. ``prem_last``,
    ``prem_market`` and ``prem_pure`` are unchanged by a renewal price rise, so
    the ratios scale as ``(1 + old_ratio) * (1 + f) - 1`` where ``f`` is the
    fractional increase.
    """
    out = portfolio.copy()
    factor = 1.0 + price_increase_pct / 100.0
    out[_PRICE_LEVEL] = out[_PRICE_LEVEL] * factor
    out[_PRICE_CHANGE] = (1.0 + out[_PRICE_CHANGE]) * factor - 1.0
    out[_PRICE_VS_MARKET] = (1.0 + out[_PRICE_VS_MARKET]) * factor - 1.0
    out[_PRICE_VS_TECHNICAL] = (1.0 + out[_PRICE_VS_TECHNICAL]) * factor - 1.0
    return out


def predict_price_effect(
    model: Pipeline, portfolio: pd.DataFrame, price_increase_pct: float
) -> np.ndarray:
    """Per-customer extra churn from the price rise: pred(new) - pred(old).

    Not clipped: the raw (possibly negative) model delta is returned so the caller
    can judge whether it is positive and plausible, or fall back to the assumption.
    """
    old = _predict_proba(model, portfolio)
    new = _predict_proba(model, apply_price_increase(portfolio, price_increase_pct))
    return new - old


def churn_inputs_from_model(
    model: Pipeline,
    features: pd.DataFrame,
    *,
    price_increase_pct: float,
    base_churn: float = FCA_BASE_CHURN,
    max_extra_churn: float = 0.15,
    n: int = 5000,
    seed: int = 7,
    min_plausible: float = 0.005,
) -> tuple[float, float | None]:
    """Collapse the model into the two scalars the Monte Carlo layer consumes.

    Builds a seeded portfolio, then returns ``(base, price_extra)`` where:

    * ``base`` is the model's mean base churn, rescaled to the UK base (so it
      equals ``base_churn`` by construction; it carries the model provenance).
    * ``price_extra`` is the mean per-customer extra churn from the price rise,
      each customer clipped to ``[0, max_extra_churn]``. It is returned only if
      the **raw** mean effect is positive and the clipped mean clears
      ``min_plausible``; otherwise ``None`` signals the caller to fall back to the
      labelled assumption.

    Keeping this bridge in terms of plain floats (not a ``BusinessContext``) keeps
    the quantitative layer decoupled from the ML layer.
    """
    portfolio = build_portfolio(features, n, seed=seed)
    base = float(predict_base_churn(model, portfolio, base=base_churn).mean())

    effect = predict_price_effect(model, portfolio, price_increase_pct)
    price_extra = float(np.clip(effect, 0.0, max_extra_churn).mean())
    plausible = float(effect.mean()) > 0.0 and price_extra >= min_plausible
    return base, (price_extra if plausible else None)


def churn_inputs_or_none(
    *,
    price_increase_pct: float,
    base_churn: float = FCA_BASE_CHURN,
    max_extra_churn: float = 0.15,
    model_path: str | Path | None = None,
    data_path: str | Path | None = None,
    n: int = 5000,
    seed: int = 7,
) -> tuple[float | None, float | None]:
    """Load the saved model + dataset and return the two Monte Carlo churn inputs.

    Returns ``(None, None)`` when no trained model or no prepared dataset is
    present, so callers can transparently fall back to the FCA constant and the
    labelled assumption. Otherwise defers to :func:`churn_inputs_from_model`.
    """
    # Imported lazily so the pure prediction helpers above carry no I/O imports.
    from decision_sim.ml.data import DEFAULT_DATA_PATH, prepare
    from decision_sim.ml.train import DEFAULT_MODEL_PATH, load_model

    data_path = Path(data_path) if data_path is not None else DEFAULT_DATA_PATH
    try:
        model = load_model(model_path or DEFAULT_MODEL_PATH)
    except FileNotFoundError:
        return None, None
    if not Path(data_path).exists():
        return None, None

    features = prepare(data_path)
    return churn_inputs_from_model(
        model,
        features,
        price_increase_pct=price_increase_pct,
        base_churn=base_churn,
        max_extra_churn=max_extra_churn,
        n=n,
        seed=seed,
    )


__all__ = [
    "FCA_BASE_CHURN",
    "apply_price_increase",
    "build_portfolio",
    "churn_inputs_from_model",
    "churn_inputs_or_none",
    "predict_base_churn",
    "predict_price_effect",
    "rescale_to_base",
]
