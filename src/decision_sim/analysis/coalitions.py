"""Detect coalitions of agents with similar final stances.

Pure and decoupled: depends only on the schemas and NetworkX, with no imports
from ``simulation`` or ``ui``. The caller passes a plain ``{agent: Stance}``
mapping (e.g. from ``simulation.graph.final_stance_by_agent``).
"""

from __future__ import annotations

import networkx as nx

from decision_sim.schemas import Stance


def final_scores(agent_stances: dict[str, Stance]) -> dict[str, float]:
    """Return each agent's final stance score."""
    return {agent: stance.score for agent, stance in agent_stances.items()}


def detect_coalitions(
    agent_stances: dict[str, Stance], *, threshold: float = 0.4
) -> list[list[str]]:
    """Group agents into coalitions by similar, same-side final stances.

    Two agents are linked when they are on the same side of neutral (both
    supportive ``>= 0`` or both opposed ``< 0``) and their scores differ by no
    more than ``threshold``. Coalitions are the connected components of that
    graph.

    Args:
        agent_stances: Mapping of agent name to final stance.
        threshold: Maximum score gap for two agents to be linked.

    Returns:
        A list of coalitions, each a sorted list of agent names. The list is
        ordered by descending coalition size then name, for determinism.
    """
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}.")

    scores = final_scores(agent_stances)
    graph: nx.Graph = nx.Graph()
    graph.add_nodes_from(scores)

    agents = sorted(scores)
    for i, a in enumerate(agents):
        for b in agents[i + 1 :]:
            same_side = (scores[a] >= 0) == (scores[b] >= 0)
            if same_side and abs(scores[a] - scores[b]) <= threshold:
                graph.add_edge(a, b)

    components = [sorted(component) for component in nx.connected_components(graph)]
    components.sort(key=lambda group: (-len(group), group))
    return components
