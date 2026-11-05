"""Tests for the churn-model data layer (pure cleaning + feature engineering).

Uses a tiny synthetic fixture with the real dataset's schema; never downloads
the real ``eudirectlapse`` data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from decision_sim.ml.data import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_features,
    clean,
    load_dataset,
    prepare,
    split_xy,
)

FIXTURE = Path(__file__).parent / "fixtures" / "eudirectlapse_tiny.csv"


def _raw() -> pd.DataFrame:
    return load_dataset(FIXTURE)


def test_fixture_loads_with_expected_columns() -> None:
    df = _raw()
    assert len(df) == 40
    assert TARGET in df.columns
    assert "prem_last" in df.columns and "prem_market" in df.columns


def test_clean_coerces_target_to_binary() -> None:
    cleaned = clean(_raw())
    assert set(cleaned[TARGET].unique()) <= {0, 1}
    assert cleaned[TARGET].dtype == int


def test_clean_casts_categoricals_to_str() -> None:
    cleaned = clean(_raw())
    for column in CATEGORICAL_FEATURES:
        assert cleaned[column].map(type).eq(str).all()


def test_clean_drops_nonpositive_premiums() -> None:
    df = _raw()
    df.loc[0, "prem_last"] = 0.0
    df.loc[1, "prem_market"] = -5.0
    cleaned = clean(df)
    assert len(cleaned) == len(df) - 2
    assert (cleaned["prem_last"] > 0).all()
    assert (cleaned["prem_market"] > 0).all()


def test_clean_rejects_missing_columns() -> None:
    df = _raw().drop(columns=["prem_market"])
    with pytest.raises(ValueError, match="missing required columns"):
        clean(df)


def test_clean_rejects_non_binary_target() -> None:
    df = _raw()
    df.loc[0, "lapse"] = 2
    with pytest.raises(ValueError, match="must be binary"):
        clean(df)


def test_build_features_adds_price_ratios() -> None:
    cleaned = clean(_raw())
    features = build_features(cleaned)

    assert "price_change" in features.columns
    assert "price_vs_market" in features.columns
    # Verify the ratio maths on the first row.
    first = cleaned.iloc[0]
    expected_change = first["prem_final"] / first["prem_last"] - 1.0
    assert features.iloc[0]["price_change"] == pytest.approx(expected_change)


def test_build_features_keeps_only_target_and_features() -> None:
    features = build_features(clean(_raw()))
    assert list(features.columns) == [TARGET, *FEATURES]


def test_split_xy_separates_target() -> None:
    features = prepare(FIXTURE)
    x, y = split_xy(features)
    assert list(x.columns) == FEATURES
    assert y.name == TARGET
    assert len(x) == len(y) == len(features)
    assert TARGET not in x.columns


def test_numeric_and_categorical_feature_sets_disjoint() -> None:
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)


def test_build_features_does_not_mutate_input() -> None:
    cleaned = clean(_raw())
    before = cleaned.copy()
    build_features(cleaned)
    pd.testing.assert_frame_equal(cleaned, before)
