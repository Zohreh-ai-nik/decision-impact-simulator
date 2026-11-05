"""Train the churn model on the downloaded eudirectlapse dataset.

Runs the baseline and the main model, prints ROC AUC + Brier for both, and saves
the main model to ``models/`` (git-ignored). Requires the dataset — run
``scripts/download_data.py`` first. Example:

    python scripts/train_churn.py
    python scripts/train_churn.py --track   # also log to MLflow if installed
"""

from __future__ import annotations

import argparse
from pathlib import Path

from decision_sim.ml.data import prepare
from decision_sim.ml.train import DEFAULT_MODEL_PATH, save_model, train_model

DATA_PATH = Path("data/raw/eudirectlapse.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the churn model.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--track", action="store_true", help="Log to MLflow if available.")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Dataset not found at {args.data}. Run scripts/download_data.py first.")

    features = prepare(args.data)
    print(f"Prepared {len(features):,} rows; lapse rate {features['lapse'].mean():.4f}")

    for model_name in ("logreg", "hgb"):
        result = train_model(features, model_name, track=args.track)  # type: ignore[arg-type]
        print(
            f"{model_name:>7}: ROC AUC {result.roc_auc:.4f} | Brier {result.brier:.4f} "
            f"(train {result.n_train:,} / test {result.n_test:,})"
        )
        if model_name == "hgb":
            path = save_model(result.pipeline, args.out)
            print(f"Saved main model -> {path}")


if __name__ == "__main__":
    main()
