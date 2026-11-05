"""Compose a single run report from the individual analyses.

Pure and decoupled: depends only on the schemas and the sibling analysis
modules, with no imports from ``simulation`` or ``ui``. The caller provides plain
inputs (posts, a ``{agent: Stance}`` mapping, and the scenario).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from decision_sim.analysis.activity import activity_counts
from decision_sim.analysis.coalitions import detect_coalitions
from decision_sim.analysis.influence import influence_ranking
from decision_sim.quant.mapping import aggregate_stances
from decision_sim.schemas import Post, Scenario, Stance


class RunReport(BaseModel):
    """A consolidated, serialisable summary of a finished run."""

    model_config = ConfigDict(extra="forbid")

    total_posts: int = Field(ge=0)
    mean_score: float
    coalitions: list[list[str]] = Field(default_factory=list)
    influence: list[tuple[str, float]] = Field(default_factory=list)
    activity: dict[str, int] = Field(default_factory=dict)
    most_influential: str | None = None


def build_report(
    posts: list[Post],
    agent_stances: dict[str, Stance],
    scenario: Scenario,
    *,
    coalition_threshold: float = 0.4,
) -> RunReport:
    """Build a :class:`RunReport` from a run's posts and final stances.

    Args:
        posts: All posts produced by the run.
        agent_stances: Each agent's final stance.
        scenario: The scenario that was run (for channel read permissions).
        coalition_threshold: Passed to :func:`detect_coalitions`.
    """
    coalitions = detect_coalitions(agent_stances, threshold=coalition_threshold)
    influence = influence_ranking(posts, scenario)
    activity = activity_counts(posts)
    aggregate = aggregate_stances(agent_stances)

    non_breaking_posts = sum(activity.values())
    most_influential = influence[0][0] if influence else None

    return RunReport(
        total_posts=non_breaking_posts,
        mean_score=aggregate["mean_score"],
        coalitions=coalitions,
        influence=influence,
        activity=activity,
        most_influential=most_influential,
    )
