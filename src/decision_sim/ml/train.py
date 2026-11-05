"""Train and evaluate the churn (lapse) model.

Two models are built on the engineered features from :mod:`decision_sim.ml.data`:
a `LogisticRegression` baseline and a `HistGradientBoostingClassifier` main
model, both wrapped in a preprocessing `Pipeline` (one-hot categoricals, scaled
numerics for the linear baseline). We report ROC AUC and Brier score on a
stratified hold-out split and persist the fitted pipeline with joblib.

The model captures *relative* churn risk and the price-change effect; the
absolute level is rescaled to the FCA UK base at prediction time (see
:mod:`decision_sim.ml.predict`). Training never happens in the test suite against
the real dataset — tests use the tiny synthetic fixture.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from decision_sim.ml.data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    prepare,
    split_xy,
)
from decision_sim.ml.tracking import start_run

ModelName = Literal["logreg", "hgb"]

DEFAULT_MODEL_PATH = Path("models/churn_model.joblib")
DEFAULT_SEED = 7

# Price features may only *increase* churn, never decrease it. This removes the
# dataset confounding (bigger rises were given to stickier customers) so the
# price-change effect comes out with the right sign. Keyed on the transformed
# column names produced by the ColumnTransformer (``num__<feature>``).
_MONOTONIC_INCREASING_PRICE = {
    "num__price_change": 1,
    "num__price_vs_market": 1,
    "num__price_vs_technical": 1,
}


@dataclass(frozen=True)
class EvalResult:
    """Metrics and the fitted pipeline for one trained model."""

    model_name: ModelName
    roc_auc: float
    brier: float
    n_train: int
    n_test: int
    pipeline: Pipeline


def _build_pipeline(model_name: ModelName, seed: int, *, monotonic: bool = True) -> Pipeline:
    """Construct a preprocessing + estimator pipeline for the named model.

    When ``monotonic`` is true, the HGB model constrains the price features to be
    monotonically increasing in churn (see ``_MONOTONIC_INCREASING_PRICE``); it
    has no effect on the linear baseline.
    """
    if model_name == "logreg":
        pre = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUMERIC_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ]
        )
        estimator = LogisticRegression(max_iter=1000, random_state=seed)
    elif model_name == "hgb":
        # Tree model does not need scaling; one-hot the categoricals as a dense
        # array (HistGradientBoostingClassifier does not accept sparse input).
        dense_one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        pre = ColumnTransformer(
            transformers=[
                ("num", "passthrough", NUMERIC_FEATURES),
                ("cat", dense_one_hot, CATEGORICAL_FEATURES),
            ]
        )
        # Pandas output so the monotonic constraints can be keyed by column name.
        pre.set_output(transform="pandas")
        monotonic_cst = dict(_MONOTONIC_INCREASING_PRICE) if monotonic else None
        estimator = HistGradientBoostingClassifier(
            random_state=seed, monotonic_cst=monotonic_cst
        )
    else:  # pragma: no cover - guarded by Literal typing
        raise ValueError(f"Unknown model_name: {model_name!r}.")

    return Pipeline([("pre", pre), ("model", estimator)])


def train_model(
    features: pd.DataFrame,
    model_name: ModelName = "hgb",
    *,
    seed: int = DEFAULT_SEED,
    test_size: float = 0.25,
    track: bool = False,
    monotonic: bool = True,
) -> EvalResult:
    """Fit one model on a stratified split and return metrics + the pipeline.

    Args:
        features: Output of :func:`decision_sim.ml.data.build_features`.
        model_name: ``"logreg"`` baseline or ``"hgb"`` main model.
        seed: Random seed for the split and the estimator.
        test_size: Fraction held out for evaluation.
        track: Log params/metrics to MLflow when available.
        monotonic: Constrain price features to only increase churn (HGB only).

    Returns:
        An :class:`EvalResult` with ROC AUC, Brier score and the fitted pipeline.
    """
    x, y = split_xy(features)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )

    pipeline = _build_pipeline(model_name, seed, monotonic=monotonic)
    pipeline.fit(x_train, y_train)

    probs = pipeline.predict_proba(x_test)[:, 1]
    roc_auc = float(roc_auc_score(y_test, probs))
    brier = float(brier_score_loss(y_test, probs))

    with start_run(f"churn-{model_name}", enabled=track) as logger:
        logger.log_params(
            {
                "model": model_name,
                "seed": seed,
                "test_size": test_size,
                "n_train": len(x_train),
                "monotonic": monotonic,
            }
        )
        logger.log_metrics({"roc_auc": roc_auc, "brier": brier})

    return EvalResult(
        model_name=model_name,
        roc_auc=roc_auc,
        brier=brier,
        n_train=len(x_train),
        n_test=len(x_test),
        pipeline=pipeline,
    )


def save_model(pipeline: Pipeline, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    """Persist a fitted pipeline to ``path`` (parent dirs created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_model(path: str | Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load a fitted pipeline, raising a clear error if it is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained churn model at {path}. Run scripts/train_churn.py first."
        )
    return joblib.load(path)


def train_and_save(
    data_path: str | Path,
    *,
    model_name: ModelName = "hgb",
    model_path: str | Path = DEFAULT_MODEL_PATH,
    seed: int = DEFAULT_SEED,
    track: bool = False,
) -> EvalResult:
    """End-to-end: prepare data, train the model, save it, return metrics."""
    features = prepare(data_path)
    result = train_model(features, model_name, seed=seed, track=track)
    save_model(result.pipeline, model_path)
    return result


__all__ = [
    "EvalResult",
    "TARGET",
    "load_model",
    "save_model",
    "train_and_save",
    "train_model",
]
