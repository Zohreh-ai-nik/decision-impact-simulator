"""Pure, testable helpers that back the Streamlit UI.

Keeping this logic in ``src`` (not ``app.py``) means it can be unit-tested
offline and reused outside Streamlit. Nothing here imports ``streamlit``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd

from decision_sim.intelligence.graph import GraphStore
from decision_sim.intelligence.rag import KeywordRetriever, Retriever
from decision_sim.schemas import Evidence, Scenario

__all__ = [
    "build_evidence_by_agent",
    "graph_edges_dataframe",
]


def _agent_query(scenario: Scenario, agent_name: str) -> str:
    """Build a retrieval query describing one stakeholder."""
    agent = next((a for a in scenario.stakeholders if a.name == agent_name), None)
    if agent is None:
        return agent_name
    parts = [agent.name, agent.role, *agent.goals, *agent.concerns]
    return " ".join(p for p in parts if p)


def build_evidence_by_agent(
    scenario: Scenario,
    evidence: Sequence[Evidence],
    *,
    k: int = 3,
    retriever_factory: Callable[[], Retriever] = KeywordRetriever,
) -> dict[str, list[Evidence]]:
    """Retrieve the top-``k`` evidence chunks for each stakeholder.

    The result is ready to pass straight to
    :func:`decision_sim.simulation.graph.iter_simulation` as ``evidence_by_agent``.
    Returns an empty mapping when there is no evidence, so an ungrounded run is the
    natural default. Uses an offline :class:`KeywordRetriever` by default.
    """
    if not evidence:
        return {}
    retriever = retriever_factory()
    retriever.add(list(evidence))
    mapping: dict[str, list[Evidence]] = {}
    for agent in scenario.stakeholders:
        hits = retriever.search(_agent_query(scenario, agent.name), k=k)
        if hits:
            mapping[agent.name] = hits
    return mapping


def graph_edges_dataframe(store: GraphStore) -> pd.DataFrame:
    """Render a knowledge graph's edges as a tidy ``source/relation/target`` frame."""
    columns = ["source", "relation", "target"]
    rows = [
        {"source": e.source, "relation": e.relation, "target": e.target}
        for e in store.edges()
    ]
    return pd.DataFrame(rows, columns=columns)
