"""Pure functions mapping agent stances to behavioural probabilities.

This module is intentionally self-contained: it depends only on the schemas and
the standard library, with no imports from ``simulation`` or ``ui``. That keeps
the quantitative layer easy to test and to extract into a separate service.
"""

from __future__ import annotations

from decision_sim.schemas import Stance

DEFAULT_MAX_CHURN = 0.6


def churn_probability(stance: Stance, *, max_churn: float = DEFAULT_MAX_CHURN) -> float:
    """Map a stance to a per-customer churn probability in ``[0, max_churn]``.

    The mapping is deterministic and monotonic:

    * ``dissatisfaction = (1 - score) / 2`` turns a score in ``[-1, 1]`` into
      ``[0, 1]`` (supportive -> 0, hostile -> 1).
    * It is scaled by the agent's ``confidence`` -- an unsure agent moves fewer
      customers -- and by ``max_churn`` so churn never exceeds a plausible cap.

    Args:
        stance: The stance to convert.
        max_churn: Upper bound on churn probability (defaults to 0.6).

    Returns:
        A probability in ``[0, max_churn]``.
    """
    if not 0.0 <= max_churn <= 1.0:
        raise ValueError(f"max_churn must be in [0, 1], got {max_churn}.")
    dissatisfaction = (1.0 - stance.score) / 2.0
    return max_churn * dissatisfaction * stance.confidence


def aggregate_stances(
    agent_stances: dict[str, Stance], *, max_churn: float = DEFAULT_MAX_CHURN
) -> dict[str, float]:
    """Summarise a set of final stances.

    Args:
        agent_stances: Mapping of agent name to that agent's final stance.
        max_churn: Passed through to :func:`churn_probability`.

    Returns:
        A dict with ``mean_score`` and ``mean_churn_probability`` (both ``0.0``
        when ``agent_stances`` is empty).
    """
    if not agent_stances:
        return {"mean_score": 0.0, "mean_churn_probability": 0.0}

    scores = [s.score for s in agent_stances.values()]
    churns = [churn_probability(s, max_churn=max_churn) for s in agent_stances.values()]
    return {
        "mean_score": sum(scores) / len(scores),
        "mean_churn_probability": sum(churns) / len(churns),
    }
