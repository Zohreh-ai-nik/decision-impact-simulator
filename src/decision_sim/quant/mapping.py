"""Pure functions mapping agent stances to behavioural probabilities.

This module is intentionally self-contained: it depends only on the schemas and
the standard library, with no imports from ``simulation`` or ``ui``. That keeps
the quantitative layer easy to test and to extract into a separate service.

Churn model
-----------
``total_churn = base_churn + extra_churn`` where

* ``base_churn`` is the churn that happens anyway (no price rise), and
* ``extra_churn = max_extra_churn * dissatisfaction`` is the additional churn
  caused by a negative stance, capped at ``max_extra_churn``.
"""

from __future__ import annotations

from decision_sim.schemas import Stance

DEFAULT_BASE_CHURN = 0.08
DEFAULT_MAX_EXTRA_CHURN = 0.15


def dissatisfaction(stance: Stance) -> float:
    """Return how dissatisfied a stance is, in ``[0, 1]``.

    ``(1 - score) / 2`` maps a score in ``[-1, 1]`` to ``[0, 1]`` (supportive ->
    0, hostile -> 1). It is scaled by ``confidence``: an unsure agent moves
    fewer customers.
    """
    return (1.0 - stance.score) / 2.0 * stance.confidence


def _check_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")


def extra_churn_probability(
    stance: Stance, *, max_extra_churn: float = DEFAULT_MAX_EXTRA_CHURN
) -> float:
    """Extra per-customer churn caused by the stance, in ``[0, max_extra_churn]``."""
    _check_probability("max_extra_churn", max_extra_churn)
    return max_extra_churn * dissatisfaction(stance)


def total_churn_probability(
    stance: Stance,
    *,
    base_churn: float = DEFAULT_BASE_CHURN,
    max_extra_churn: float = DEFAULT_MAX_EXTRA_CHURN,
) -> float:
    """Per-customer churn probability: base churn plus stance-driven extra churn.

    The result lies in ``[base_churn, min(1, base_churn + max_extra_churn)]``.
    """
    _check_probability("base_churn", base_churn)
    extra = extra_churn_probability(stance, max_extra_churn=max_extra_churn)
    return min(1.0, base_churn + extra)


def aggregate_stances(
    agent_stances: dict[str, Stance],
    *,
    base_churn: float = DEFAULT_BASE_CHURN,
    max_extra_churn: float = DEFAULT_MAX_EXTRA_CHURN,
) -> dict[str, float]:
    """Summarise a set of final stances.

    Args:
        agent_stances: Mapping of agent name to that agent's final stance.
        base_churn: Passed through to :func:`total_churn_probability`.
        max_extra_churn: Passed through to :func:`total_churn_probability`.

    Returns:
        A dict with ``mean_score`` and ``mean_churn_probability``. With no
        stances, ``mean_score`` is ``0.0`` and the churn is just ``base_churn``.
    """
    if not agent_stances:
        return {"mean_score": 0.0, "mean_churn_probability": base_churn}

    scores = [s.score for s in agent_stances.values()]
    churns = [
        total_churn_probability(s, base_churn=base_churn, max_extra_churn=max_extra_churn)
        for s in agent_stances.values()
    ]
    return {
        "mean_score": sum(scores) / len(scores),
        "mean_churn_probability": sum(churns) / len(churns),
    }
