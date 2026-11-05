"""Streamlit front-end for the decision-impact simulator.

This module is intentionally thin: it only wires Streamlit widgets to the
simulation logic that lives in :mod:`decision_sim`. All scenario loading,
signal editing, simulation stepping and data shaping happen in ``src``.

Run with::

    streamlit run src/decision_sim/ui/app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from decision_sim.analysis.activity import activity_dataframe
from decision_sim.analysis.report import build_report
from decision_sim.config import SimConfig
from decision_sim.quant.formatting import (
    churn_assumption_text,
    churn_breakdown_rows,
    format_gbp_millions,
    format_pct_change,
    impact_histogram,
    impact_summary_rows,
)
from decision_sim.quant.monte_carlo import simulate_impact
from decision_sim.scenario import add_signal, list_scenarios, load_scenario
from decision_sim.schemas import Post, Scenario
from decision_sim.simulation.graph import (
    final_stance_by_agent,
    iter_simulation,
    save_run,
    stances_dataframe,
)


def _group_posts(posts: list[Post]) -> dict[str, dict[str, list[Post]]]:
    """Group posts by stage, then by channel, preserving arrival order."""
    grouped: dict[str, dict[str, list[Post]]] = {}
    for post in posts:
        grouped.setdefault(post.stage, {}).setdefault(post.channel, []).append(post)
    return grouped


def _render_posts(container, posts: list[Post]) -> None:
    container.empty()
    with container.container():
        for stage, channels in _group_posts(posts).items():
            st.subheader(f"Stage: {stage}")
            for channel, channel_posts in channels.items():
                with st.expander(f"#{channel} ({len(channel_posts)})", expanded=True):
                    for post in channel_posts:
                        st.markdown(f"**{post.agent}** (round {post.round}): {post.text}")


def _render_chart(scenario: Scenario, stances) -> None:
    df = stances_dataframe(stances)
    if df.empty:
        st.info("No stances recorded yet.")
        return
    pivot = df.pivot_table(
        index="global_round", columns="agent", values="score", aggfunc="last"
    )
    st.line_chart(pivot)


def _render_impact(scenario: Scenario, stances) -> None:
    """Compute and render the quantified revenue impact for a finished run."""
    if scenario.business is None:
        st.info("This scenario has no business section, so no impact is estimated.")
        return

    agent_stances = final_stance_by_agent(stances)
    result = simulate_impact(agent_stances, scenario.business)

    st.subheader("Estimated impact (vs. no price rise)")
    st.caption(churn_assumption_text(scenario.business))

    st.markdown("**Churn breakdown**")
    churn_cols = st.columns(3)
    for col, (label, value) in zip(churn_cols, churn_breakdown_rows(result), strict=True):
        col.metric(label, value)

    cols = st.columns(3)
    cols[0].metric(
        "Median (p50)",
        format_gbp_millions(result.revenue_impact_p50),
        format_pct_change(result.revenue_impact_pct_p50),
    )
    cols[1].metric(
        "Downside (p10)",
        format_gbp_millions(result.revenue_impact_p10),
        format_pct_change(result.revenue_impact_pct_p10),
    )
    cols[2].metric(
        "Upside (p90)",
        format_gbp_millions(result.revenue_impact_p90),
        format_pct_change(result.revenue_impact_pct_p90),
    )

    for label, value in impact_summary_rows(result):
        st.write(f"- **{label}:** {value}")

    centers, counts = impact_histogram(result, in_millions=True)
    if centers:
        axis = "Revenue impact (£m)"
        hist_df = pd.DataFrame({axis: centers, "Simulation runs": counts}).set_index(axis)
        st.bar_chart(hist_df, x_label=axis, y_label="Simulation runs")


def _render_analysis(scenario: Scenario, posts: list[Post], stances) -> None:
    """Render coalitions, influence and activity for a finished run."""
    agent_stances = final_stance_by_agent(stances)
    report = build_report(posts, agent_stances, scenario)

    st.subheader("Analysis")
    if report.most_influential:
        st.caption(f"Most influential agent: **{report.most_influential}**")

    st.markdown("**Coalitions** (agents with aligned final stances)")
    if report.coalitions:
        for group in report.coalitions:
            st.write("- " + ", ".join(group))
    else:
        st.write("- None detected.")

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Influence**")
        if report.influence:
            inf_df = pd.DataFrame(report.influence, columns=["agent", "score"]).set_index(
                "agent"
            )
            st.bar_chart(inf_df)
    with cols[1]:
        st.markdown("**Activity (posts per agent)**")
        act_df = activity_dataframe(posts)
        if not act_df.empty:
            st.bar_chart(act_df.set_index("agent"))



def main() -> None:
    st.set_page_config(page_title="Decision Impact Simulator", layout="wide")
    st.title("Decision Impact Simulator")

    scenario_files = list_scenarios()
    if not scenario_files:
        st.error("No scenario files found in the scenarios/ directory.")
        return

    with st.sidebar:
        st.header("Setup")
        choice = st.selectbox(
            "Scenario", scenario_files, format_func=lambda p: p.stem
        )
        scenario = load_scenario(choice)

        st.caption(f"**Decision:** {scenario.decision}")
        st.caption(f"**Client:** {scenario.client}")

        max_rounds = st.number_input("Max rounds", min_value=1, value=20, step=1)
        seed = st.number_input("Seed", min_value=0, value=7, step=1)

        st.divider()
        st.subheader("Add a signal")
        signal_text = st.text_input("Signal text")
        stage_names = [s.name for s in scenario.stages]
        signal_stage = st.selectbox("Signal stage", stage_names)
        signal_round = st.number_input("Signal round", min_value=1, value=1, step=1)

        if "extra_signals" not in st.session_state:
            st.session_state.extra_signals = []

        if st.button("Add signal") and signal_text.strip():
            st.session_state.extra_signals.append(
                {
                    "text": signal_text.strip(),
                    "stage": signal_stage,
                    "round": int(signal_round),
                }
            )

        if st.session_state.extra_signals:
            st.caption("Pending extra signals:")
            for sig in st.session_state.extra_signals:
                st.write(f"- [{sig['stage']} r{sig['round']}] {sig['text']}")
            if st.button("Clear extra signals"):
                st.session_state.extra_signals = []

        run = st.button("Run", type="primary")

    # Apply any user-added signals to a fresh scenario copy.
    for sig in st.session_state.get("extra_signals", []):
        scenario = add_signal(scenario, sig["text"], sig["stage"], sig["round"])

    config = SimConfig(max_rounds=int(max_rounds), seed=int(seed))

    posts_col, chart_col = st.columns([3, 2])
    posts_area = posts_col.empty()
    chart_area = chart_col.empty()

    if not run:
        posts_col.info("Configure the run in the sidebar, then press **Run**.")
        return

    final_state = None
    with st.spinner("Running simulation..."):
        for state in iter_simulation(scenario, config):
            final_state = state
            _render_posts(posts_area, state["posts"])
            with chart_area.container():
                _render_chart(scenario, state["stances"])

    if final_state is not None:
        path = save_run(final_state)
        st.success(f"Run complete ({final_state['rounds_executed']} rounds). Saved to {path}")
        _render_impact(scenario, final_state["stances"])
        _render_analysis(scenario, final_state["posts"], final_state["stances"])


if __name__ == "__main__":
    main()
