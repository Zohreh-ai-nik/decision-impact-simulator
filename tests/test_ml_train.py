"""Tests for the churn-model training layer.

All training happens on the tiny synthetic fixture; never the real dataset,
never MLflow, never the network.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sklearn.pipeline import Pipeline

from decision_sim.ml.data import prepare
from decision_sim.ml.tracking import start_run
from decision_sim.ml.train import (
    EvalResult,
    load_model,
    save_model,
    train_and_save,
    train_model,
)

FIXTURE = Path(__file__).parent / "fixtures" / "eudirectlapse_tiny.csv"


def _features():
    return prepare(FIXTURE)


@pytest.mark.parametrize("model_name", ["logreg", "hgb"])
def test_train_model_returns_valid_metrics(model_name: str) -> None:
    result = train_model(_features(), model_name)  # type: ignore[arg-type]
    assert isinstance(result, EvalResult)
    assert 0.0 <= result.roc_auc <= 1.0
    assert 0.0 <= result.brier <= 1.0
    assert result.n_train > 0 and result.n_test > 0
    assert isinstance(result.pipeline, Pipeline)


def test_pipeline_predicts_probabilities_in_range() -> None:
    features = _features()
    result = train_model(features, "hgb")
    probs = result.pipeline.predict_proba(features[result.pipeline.feature_names_in_])[:, 1]
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_training_is_deterministic_with_fixed_seed() -> None:
    a = train_model(_features(), "hgb", seed=7)
    b = train_model(_features(), "hgb", seed=7)
    assert a.roc_auc == b.roc_auc
    assert a.brier == b.brier


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    result = train_model(_features(), "hgb")
    path = save_model(result.pipeline, tmp_path / "m.joblib")
    assert path.exists()

    loaded = load_model(path)
    features = _features()
    x = features[loaded.feature_names_in_]
    original = result.pipeline.predict_proba(x)[:, 1]
    reloaded = loaded.predict_proba(x)[:, 1]
    assert (original == reloaded).all()


def test_load_model_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No trained churn model"):
        load_model(tmp_path / "nope.joblib")


def test_train_and_save_end_to_end(tmp_path: Path) -> None:
    out = tmp_path / "churn.joblib"
    result = train_and_save(FIXTURE, model_path=out)
    assert out.exists()
    assert 0.0 <= result.roc_auc <= 1.0


def test_monotonic_hgb_price_effect_never_negative() -> None:
    """With the constraint on, raising the price can only raise predicted churn."""
    features = _features()
    model = train_model(features, "hgb", monotonic=True).pipeline
    x = features[model.feature_names_in_]
    old = model.predict_proba(x)[:, 1]
    raised = x.copy()
    for col in ("price_change", "price_vs_market", "price_vs_technical"):
        raised[col] = (1.0 + raised[col]) * 1.12 - 1.0
    new = model.predict_proba(raised)[:, 1]
    assert (new >= old - 1e-9).all()


def test_tracking_disabled_is_noop_context() -> None:
    with start_run("x", enabled=False) as logger:
        logger.log_params({"a": 1})
        logger.log_metrics({"m": 0.5})  # must not raise


def test_tracking_enabled_without_mlflow_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """If mlflow import fails, start_run must still yield a working no-op logger."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mlflow":
            raise ImportError("simulated missing mlflow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with start_run("x", enabled=True) as logger:
        logger.log_params({"a": 1})
        logger.log_metrics({"m": 0.5})  # must not raise
