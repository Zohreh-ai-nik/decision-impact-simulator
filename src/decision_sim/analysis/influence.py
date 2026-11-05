"""Rank agents by communicative influence using a reach-weighted graph.

Pure and decoupled: depends only on the schemas and NetworkX, with no imports
from ``simulation`` or ``ui``.

Influence model: when an agent posts in a channel, it reaches every agent that
can read that channel. We build a directed graph with an edge ``poster ->
reader`` weighted by how many such posts occurred, then rank agents by PageRank.
Agents who reach many others (and are themselves reached by influential agents)
score higher. The ``Breaking News`` pseudo-agent is excluded.
"""

from __future__ import annotations

import networkx as nx

from decision_sim.schemas import Post, Scenario

BREAKING_NEWS_AGENT = "Breaking News"


def build_interaction_graph(posts: list[Post], scenario: Scenario) -> nx.DiGraph:
    """Build a directed, reach-weighted interaction graph from posts.

    An edge ``poster -> reader`` is added for every agent (other than the
    poster) who can read the channel a post was made in; its weight accumulates
    across posts. The ``Breaking News`` pseudo-agent is ignored as a poster.
    """
    readers_by_channel = {c.name: set(c.can_read) for c in scenario.channels}

    graph: nx.DiGraph = nx.DiGraph()
    graph.add_nodes_from(a.name for a in scenario.stakeholders)

    for post in posts:
        if post.agent == BREAKING_NEWS_AGENT:
            continue
        for reader in readers_by_channel.get(post.channel, set()):
            if reader == post.agent:
                continue
            if graph.has_edge(post.agent, reader):
                graph[post.agent][reader]["weight"] += 1.0
            else:
                graph.add_edge(post.agent, reader, weight=1.0)
    return graph


def influence_ranking(
    posts: list[Post], scenario: Scenario
) -> list[tuple[str, float]]:
    """Rank stakeholders by influence over the interaction graph.

    ``build_interaction_graph`` yields information-flow edges (``poster ->
    reader``). Influence runs PageRank on the *reversed* graph, so an agent that
    is read by many -- especially by other influential agents -- accumulates a
    high score.

    Returns:
        A list of ``(agent, score)`` pairs sorted by descending score then name.
        Scores sum to roughly 1.0 (PageRank). Returns a uniform ranking when
        there are no interactions.
    """
    graph = build_interaction_graph(posts, scenario)
    scores = _pagerank(graph.reverse(copy=False))
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def _pagerank(
    graph: nx.DiGraph,
    *,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> dict[str, float]:
    """Deterministic pure-Python weighted PageRank (no SciPy dependency).

    Uses power iteration with uniform teleportation and dangling-node handling,
    so it works on any directed, weighted graph without optional backends.
    """
    nodes = list(graph.nodes)
    n = len(nodes)
    if n == 0:
        return {}

    rank = {node: 1.0 / n for node in nodes}
    out_weight = {
        node: sum(data.get("weight", 1.0) for _, _, data in graph.out_edges(node, data=True))
        for node in nodes
    }

    for _ in range(max_iter):
        dangling = alpha * sum(rank[node] for node in nodes if out_weight[node] == 0.0)
        new_rank = {node: (1.0 - alpha) / n + dangling / n for node in nodes}
        for src in nodes:
            if out_weight[src] == 0.0:
                continue
            share = alpha * rank[src] / out_weight[src]
            for _, dst, data in graph.out_edges(src, data=True):
                new_rank[dst] += share * data.get("weight", 1.0)

        delta = sum(abs(new_rank[node] - rank[node]) for node in nodes)
        rank = new_rank
        if delta < tol:
            break
    return rank
