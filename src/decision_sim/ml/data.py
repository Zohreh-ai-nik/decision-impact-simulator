"""Pure data cleaning and feature engineering for the churn (lapse) model.

Operates on the public ``eudirectlapse`` dataset (see ``docs/data.md``). This
module is deliberately dependency-light (pandas only) and has no I/O beyond
reading a parquet/CSV file, so it is easy to unit-test on a tiny fixture without
downloading the real dataset.

The model trained on these features is **not** used for the absolute UK churn
level (this is a European direct insurer with ~12.8% lapse). It is used for
relative risk between customers and for the price-change effect; the absolute
level is rescaled to the FCA UK base elsewhere. See ``docs/calibration.md``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "lapse"

# Default location of the prepared real dataset (git-ignored; see docs/data.md).
DEFAULT_DATA_PATH = Path("data/raw/eudirectlapse.parquet")

# Raw numeric columns used directly as model features.
RAW_NUMERIC_FEATURES = [
    "polholder_age",  # driver age
    "policy_age",  # tenure: years with the insurer
    "policy_nbcontract",  # number of policies/contracts held
    "vehicl_age",
    "vehicl_agepurchase",
    "prem_final",  # absolute price level matters too
]

# Engineered numeric features derived in :func:`build_features`.
# Order matters: the HGB monotonic constraints in train.py are positional.
ENGINEERED_FEATURES = [
    "price_change",  # prem_final / prem_last - 1
    "price_vs_market",  # prem_final / prem_market - 1
    "price_vs_technical",  # prem_final / prem_pure - 1 (margin over risk cost)
]

NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES

# Categorical columns one-hot encoded by the model pipeline (see train.py).
CATEGORICAL_FEATURES = [
    "polholder_BMCevol",
    "polholder_diffdriver",
    "polholder_gender",
    "polholder_job",
    "policy_caruse",
    "prem_freqperyear",
    "vehicl_garage",
    "vehicl_powerkw",
    "vehicl_region",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns that must be present in the raw dataset before feature engineering.
_REQUIRED_RAW_COLUMNS = [
    TARGET,
    *RAW_NUMERIC_FEATURES,
    "prem_last",
    "prem_market",
    "prem_pure",
    *CATEGORICAL_FEATURES,
]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the dataset from parquet or CSV, chosen by file extension."""
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the raw dataset.

    Checks required columns exist, coerces the target to ``{0, 1}``, casts
    categoricals to plain strings, and drops rows where a premium used in a ratio
    is non-positive (which would make ``price_change``/``price_vs_market``
    undefined). Returns a new frame; the input is not mutated.
    """
    missing = [c for c in _REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}.")

    out = df.copy()

    out[TARGET] = out[TARGET].astype(int)
    bad_target = set(out[TARGET].unique()) - {0, 1}
    if bad_target:
        raise ValueError(f"Target {TARGET!r} must be binary 0/1, found {sorted(bad_target)}.")

    for column in CATEGORICAL_FEATURES:
        out[column] = out[column].astype(str)

    positive = (
        (out["prem_last"] > 0)
        & (out["prem_market"] > 0)
        & (out["prem_final"] > 0)
        & (out["prem_pure"] > 0)
    )
    return out.loc[positive].reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with the target plus the engineered model features.

    Adds ``price_change``, ``price_vs_market`` and ``price_vs_technical`` and
    keeps only ``TARGET`` and ``FEATURES``. Expects an already-:func:`clean`-ed
    frame.
    """
    out = df.copy()
    out["price_change"] = out["prem_final"] / out["prem_last"] - 1.0
    out["price_vs_market"] = out["prem_final"] / out["prem_market"] - 1.0
    out["price_vs_technical"] = out["prem_final"] / out["prem_pure"] - 1.0
    return out[[TARGET, *FEATURES]].copy()


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a feature frame into ``(X, y)`` for model training."""
    return df[FEATURES].copy(), df[TARGET].copy()


def prepare(path: str | Path) -> pd.DataFrame:
    """Convenience: load, clean and build features from a dataset file."""
    return build_features(clean(load_dataset(path)))
