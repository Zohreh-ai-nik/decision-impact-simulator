"""Thin, optional MLflow logging wrapper.

MLflow is an *optional* dependency (the ``ml`` extra). This module degrades
gracefully: if MLflow is not installed, the run context is a no-op so training
still works and the test suite never requires MLflow. Full experiment tracking
(scenario, model, prompt version, seed, cost, impact range) comes in Phase 12.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Protocol


class RunLogger(Protocol):
    """Minimal logging surface used by the training code."""

    def log_params(self, params: Mapping[str, Any]) -> None: ...

    def log_metrics(self, metrics: Mapping[str, float]) -> None: ...


class _NullLogger:
    """No-op logger used when MLflow is unavailable or disabled."""

    def log_params(self, params: Mapping[str, Any]) -> None:  # noqa: D102
        return None

    def log_metrics(self, metrics: Mapping[str, float]) -> None:  # noqa: D102
        return None


class _MlflowLogger:
    """Adapter that forwards to the ``mlflow`` module."""

    def __init__(self, mlflow: Any) -> None:
        self._mlflow = mlflow

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._mlflow.log_params(dict(params))

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        self._mlflow.log_metrics(dict(metrics))


@contextmanager
def start_run(run_name: str, *, enabled: bool = True) -> Iterator[RunLogger]:
    """Yield a :class:`RunLogger`, using MLflow if available and enabled.

    Args:
        run_name: Name for the MLflow run.
        enabled: When False, always yield the no-op logger (useful for tests and
            quick local runs).

    Yields:
        A logger exposing ``log_params`` and ``log_metrics``.
    """
    if not enabled:
        yield _NullLogger()
        return

    try:
        import mlflow  # noqa: PLC0415 - optional dependency, imported lazily
    except ImportError:
        yield _NullLogger()
        return

    with mlflow.start_run(run_name=run_name):
        yield _MlflowLogger(mlflow)
