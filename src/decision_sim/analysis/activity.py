"""Post-volume activity metrics per agent and per channel.

Pure and decoupled: depends only on the schemas and pandas, with no imports from
``simulation`` or ``ui``. The ``Breaking News`` pseudo-agent is excluded from
agent counts.
"""

from __future__ import annotations

import pandas as pd

from decision_sim.schemas import Post

BREAKING_NEWS_AGENT = "Breaking News"


def activity_counts(posts: list[Post]) -> dict[str, int]:
    """Return the number of posts made by each agent (excluding Breaking News)."""
    counts: dict[str, int] = {}
    for post in posts:
        if post.agent == BREAKING_NEWS_AGENT:
            continue
        counts[post.agent] = counts.get(post.agent, 0) + 1
    return counts


def channel_counts(posts: list[Post]) -> dict[str, int]:
    """Return the number of posts made in each channel (all posts counted)."""
    counts: dict[str, int] = {}
    for post in posts:
        counts[post.channel] = counts.get(post.channel, 0) + 1
    return counts


def activity_dataframe(posts: list[Post]) -> pd.DataFrame:
    """Return a tidy ``agent``/``posts`` DataFrame sorted by descending volume."""
    columns = ["agent", "posts"]
    counts = activity_counts(posts)
    if not counts:
        return pd.DataFrame(columns=columns)
    rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return pd.DataFrame(rows, columns=columns)
