"""Tests for the churn-prediction layer.

A tiny model is trained on the synthetic fixture; never the real dataset,
never MLflow, never the network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from decision_sim.ml.data import prepare
from decision_sim.ml.predict import (
    FCA_BASE_CHURN,
    apply_price_increase,
    build_portfolio,
    predict_base_churn,
    predict_price_effect,
    rescale_to_base,
)
from decision_sim.ml.train import train_model

FIXTURE = Path(__file__).parent / "fixtures" / "eudirectlapse_tiny.csv"


@pytest.fixture(scope="module")
def features():
    return prepare(FIXTURE)


@pytest.fixture(scope="module")
def model(features):
    return train_model(features, "hgb", seed=7).pipeline


# --- build_portfolio -------------------------------------------------------


def test_build_portfolio_size_and_columns(features) -> None:
    portfolio = build_portfolio(features, 50, seed=7)
    assert len(portfolio) == 50
    assert set(features.columns).issubset(portfolio.columns)


def test_build_portfolio_is_deterministic(features) -> None:
    a = build_portfolio(features, 30, seed=7)
    b = build_portfolio(features, 30, seed=7)
    assert a.equals(b)


def test_build_portfolio_rejects_non_positive(features) -> None:
    with pytest.raises(ValueError, match="positive"):
        build_portfolio(features, 0)


# --- rescale_to_base -------------------------------------------------------


def test_rescale_hits_target_mean() -> None:
    probs = np.array([0.05, 0.10, 0.15, 0.20, 0.25])
    out = rescale_to_base(probs, 0.40)
    assert out.mean() == pytest.approx(0.40, abs=1e-4)


def test_rescale_preserves_ordering() -> None:
    probs = np.array([0.30, 0.05, 0.20, 0.10])
    out = rescale_to_base(probs, 0.40)
    assert list(np.argsort(out)) == list(np.argsort(probs))
    assert ((out >= 0.0) & (out <= 1.0)).all()


def test_rescale_empty_returns_empty() -> None:
    out = rescale_to_base(np.array([]), 0.40)
    assert out.size == 0


def test_rescale_rejects_bad_base() -> None:
    with pytest.raises(ValueError, match="base"):
        rescale_to_base(np.array([0.1, 0.2]), 1.5)


# --- predict_base_churn ----------------------------------------------------


def test_predict_base_churn_matches_fca_mean(model, features) -> None:
    portfolio = build_portfolio(features, 100, seed=7)
    churn = predict_base_churn(model, portfolio, base=FCA_BASE_CHURN)
    assert churn.mean() == pytest.approx(FCA_BASE_CHURN, abs=1e-4)
    assert ((churn >= 0.0) & (churn <= 1.0)).all()


# --- apply_price_increase / predict_price_effect ---------------------------


def test_apply_price_increase_scales_premium_and_ratios(features) -> None:
    portfolio = build_portfolio(features, 20, seed=7)
    raised = apply_price_increase(portfolio, 10.0)
    assert np.allclose(raised["prem_final"], portfolio["prem_final"] * 1.10)
    expected_pc = (1.0 + portfolio["price_change"]) * 1.10 - 1.0
    assert np.allclose(raised["price_change"], expected_pc)


def test_apply_price_increase_zero_is_identity(features) -> None:
    portfolio = build_portfolio(features, 20, seed=7)
    raised = apply_price_increase(portfolio, 0.0)
    assert np.allclose(raised["prem_final"], portfolio["prem_final"])
    assert np.allclose(raised["price_change"], portfolio["price_change"])


def test_predict_price_effect_shape_and_zero(model, features) -> None:
    portfolio = build_portfolio(features, 40, seed=7)
    effect = predict_price_effect(model, portfolio, 12.0)
    assert effect.shape == (40,)
    zero_effect = predict_price_effect(model, portfolio, 0.0)
    assert np.allclose(zero_effect, 0.0)
