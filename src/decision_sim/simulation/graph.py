"""LangGraph-based orchestration of a decision-impact simulation.

The simulation advances one round at a time. In each round any due signal is
injected as a broadcast "Breaking News" post, then every stakeholder (in a
seeded, shuffled order) reads the recent posts from channels it is allowed to
read and responds -- via :func:`decision_sim.llm.ask` -- with a message and a
stance. The run moves through the scenario's stages and stops at the end or when
``SimConfig.max_rounds`` is reached. The final state can be saved as JSON under
``data/runs/``.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ConfigDict

from decision_sim import llm
from decision_sim.config import SimConfig
from decision_sim.schemas import Post, Scenario, Stance

# Repository root: src/decision_sim/simulation/graph.py -> parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_RUNS_DIR = _REPO_ROOT / "data" / "runs"

BREAKING_NEWS_AGENT = "Breaking News"

_SYSTEM_PROMPT = (
    "You are simulating a single stakeholder's reaction in a business decision "
    "war-game. Stay strictly in character, be concise, and respond only with the "
    "requested structured output."
)


class AgentResponse(BaseModel):
    """Structured reply returned by an agent for a single turn."""

    model_config = ConfigDict(extra="forbid")

    channel: str
    text: str
    stance: Stance


class StanceRecord(BaseModel):
    """An agent's stance captured with its stage/round context."""

    model_config = ConfigDict(extra="forbid")

    agent: str
    stage: str
    round: int
    stance: Stance


class SimState(TypedDict):
    """Mutable state threaded through the LangGraph run."""

    scenario: Scenario
    stage_index: int
    round: int
    posts: list[Post]
    stances: list[StanceRecord]
    signals_applied: list[int]
    rounds_executed: int


def recent_readable_posts(
    posts: list[Post],
    agent_name: str,
    scenario: Scenario,
    window: int,
) -> list[Post]:
    """Return the most recent posts in channels ``agent_name`` is allowed to read."""
    readable = {c.name for c in scenario.channels if agent_name in c.can_read}
    visible = [p for p in posts if p.channel in readable]
    return visible[-window:]


def _postable_channels(agent_name: str, scenario: Scenario) -> list[str]:
    return [c.name for c in scenario.channels if agent_name in c.can_post]


def _build_prompt(
    scenario: Scenario,
    agent,
    stage,
    round_no: int,
    recent: list[Post],
    postable: list[str],
) -> str:
    lines = [
        f"Decision under consideration: {scenario.decision}",
        f"Client / organisation: {scenario.client}",
        "",
        f"You are role-playing the stakeholder: {agent.name} ({agent.type}).",
        f"Role: {agent.role}",
    ]
    if agent.goals:
        lines.append("Your goals: " + "; ".join(agent.goals))
    if agent.concerns:
        lines.append("Your concerns: " + "; ".join(agent.concerns))
    if agent.style:
        lines.append(f"Your communication style: {agent.style}")
    lines += [
        "",
        f"Current stage: {stage.name} - {stage.goal}",
        f"This is round {round_no}.",
        "",
    ]
    if recent:
        lines.append("Recent messages you can see:")
        lines += [f"- [{p.channel}] {p.agent}: {p.text}" for p in recent]
    else:
        lines.append("There are no earlier messages you can see yet.")
    lines += [
        "",
        "Channels you may post to: " + ", ".join(postable) + ".",
        (
            "Write one short in-character message reacting to the situation, choose "
            "one of your allowed channels to post it to, and report your current stance."
        ),
    ]
    return "\n".join(lines)


def _apply_due_signals(
    scenario: Scenario,
    stage_name: str,
    round_no: int,
    posts: list[Post],
    signals_applied: list[int],
) -> None:
    """Inject any signal scheduled for this stage/round as broadcast posts."""
    for idx, signal in enumerate(scenario.signals):
        if idx in signals_applied:
            continue
        if signal.stage == stage_name and signal.round == round_no:
            for channel in scenario.channels:
                posts.append(
                    Post(
                        agent=BREAKING_NEWS_AGENT,
                        channel=channel.name,
                        stage=stage_name,
                        round=round_no,
                        text=f"Breaking news: {signal.text}",
                    )
                )
            signals_applied.append(idx)


def build_graph(config: SimConfig):
    """Build and compile the LangGraph state machine for ``config``."""

    def run_round(state: SimState) -> SimState:
        scenario = state["scenario"]
        stage = scenario.stages[state["stage_index"]]
        round_no = state["round"]

        posts = list(state["posts"])
        stances = list(state["stances"])
        signals_applied = list(state["signals_applied"])

        _apply_due_signals(scenario, stage.name, round_no, posts, signals_applied)

        rng = random.Random(
            config.seed * 10_000 + state["stage_index"] * 100 + round_no
        )
        agents = list(scenario.stakeholders)
        rng.shuffle(agents)

        for agent in agents:
            postable = _postable_channels(agent.name, scenario)
            if not postable:
                continue
            recent = recent_readable_posts(
                posts, agent.name, scenario, config.recent_posts_window
            )
            prompt = _build_prompt(scenario, agent, stage, round_no, recent, postable)
            reply = llm.ask(
                prompt,
                AgentResponse,
                system=_SYSTEM_PROMPT,
                model=config.model,
            )
            channel = reply.channel if reply.channel in postable else postable[0]
            posts.append(
                Post(
                    agent=agent.name,
                    channel=channel,
                    stage=stage.name,
                    round=round_no,
                    text=reply.text,
                )
            )
            stances.append(
                StanceRecord(
                    agent=agent.name,
                    stage=stage.name,
                    round=round_no,
                    stance=reply.stance,
                )
            )

        # Advance the timeline.
        if round_no < stage.rounds:
            next_stage_index = state["stage_index"]
            next_round = round_no + 1
        else:
            next_stage_index = state["stage_index"] + 1
            next_round = 1

        return {
            "scenario": scenario,
            "stage_index": next_stage_index,
            "round": next_round,
            "posts": posts,
            "stances": stances,
            "signals_applied": signals_applied,
            "rounds_executed": state["rounds_executed"] + 1,
        }

    def should_continue(state: SimState) -> str:
        if state["rounds_executed"] >= config.max_rounds:
            return "end"
        if state["stage_index"] >= len(state["scenario"].stages):
            return "end"
        return "continue"

    graph = StateGraph(SimState)
    graph.add_node("round", run_round)
    graph.set_entry_point("round")
    graph.add_conditional_edges(
        "round", should_continue, {"continue": "round", "end": END}
    )
    return graph.compile()


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "run"


def save_run(state: SimState, out_dir: Path | str = DATA_RUNS_DIR) -> Path:
    """Persist a finished run as a JSON file and return its path."""
    scenario = state["scenario"]
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    file_path = out_path / f"run_{timestamp}_{_slugify(scenario.client)}.json"

    payload = {
        "client": scenario.client,
        "decision": scenario.decision,
        "rounds_executed": state["rounds_executed"],
        "signals_applied": state["signals_applied"],
        "scenario": scenario.model_dump(mode="json"),
        "posts": [p.model_dump(mode="json") for p in state["posts"]],
        "stances": [s.model_dump(mode="json") for s in state["stances"]],
    }
    file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return file_path


def run_simulation(
    scenario: Scenario,
    config: SimConfig | None = None,
    *,
    save: bool = True,
) -> SimState:
    """Run a full simulation and optionally persist it to ``data/runs/``."""
    config = config or SimConfig()
    app = build_graph(config)
    final: SimState = app.invoke(
        _initial_state(scenario), {"recursion_limit": _step_limit(scenario, config)}
    )
    if save:
        save_run(final)
    return final


def _initial_state(scenario: Scenario) -> SimState:
    return {
        "scenario": scenario,
        "stage_index": 0,
        "round": 1,
        "posts": [],
        "stances": [],
        "signals_applied": [],
        "rounds_executed": 0,
    }


def _step_limit(scenario: Scenario, config: SimConfig) -> int:
    total_rounds = sum(s.rounds for s in scenario.stages)
    return min(config.max_rounds, total_rounds) + 5


def iter_simulation(
    scenario: Scenario,
    config: SimConfig | None = None,
) -> Iterator[SimState]:
    """Yield the simulation state after each round, for incremental UIs.

    Unlike :func:`run_simulation`, this does not save the run; the caller can
    persist the final state with :func:`save_run` if desired.
    """
    config = config or SimConfig()
    app = build_graph(config)
    for update in app.stream(
        _initial_state(scenario),
        {"recursion_limit": _step_limit(scenario, config)},
        stream_mode="values",
    ):
        # The first streamed value is the initial state (round 0 executed).
        if update["rounds_executed"] > 0:
            yield update


def stances_dataframe(stances: list[StanceRecord]) -> pd.DataFrame:
    """Build a tidy DataFrame of stance scores over time, per agent.

    Columns: ``agent``, ``stage``, ``round``, ``global_round`` (a per-agent
    sequential turn index) and ``score``.
    """
    columns = ["agent", "stage", "round", "global_round", "score"]
    if not stances:
        return pd.DataFrame(columns=columns)

    rows = []
    turn_by_agent: dict[str, int] = {}
    for record in stances:
        turn_by_agent[record.agent] = turn_by_agent.get(record.agent, 0) + 1
        rows.append(
            {
                "agent": record.agent,
                "stage": record.stage,
                "round": record.round,
                "global_round": turn_by_agent[record.agent],
                "score": record.stance.score,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def final_stance_by_agent(stances: list[StanceRecord]) -> dict[str, Stance]:
    """Return each agent's most recent stance, keyed by agent name.

    This is the hand-off point to the quantitative layer: the plain dict it
    returns can be passed straight to ``quant.simulate_impact`` without the
    quant package needing to know anything about the simulation.
    """
    latest: dict[str, Stance] = {}
    for record in stances:
        latest[record.agent] = record.stance
    return latest
