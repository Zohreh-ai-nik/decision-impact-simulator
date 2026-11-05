"""Post-simulation analysis: coalitions, influence and related insights."""

from decision_sim.analysis.activity import (
    activity_counts,
    activity_dataframe,
    channel_counts,
)
from decision_sim.analysis.coalitions import detect_coalitions, final_scores
from decision_sim.analysis.influence import (
    build_interaction_graph,
    influence_ranking,
)
from decision_sim.analysis.report import RunReport, build_report

__all__ = [
    "detect_coalitions",
    "final_scores",
    "build_interaction_graph",
    "influence_ranking",
    "activity_counts",
    "channel_counts",
    "activity_dataframe",
    "RunReport",
    "build_report",
]
