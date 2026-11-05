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
from decision_sim.analysis.interview import interview_agent
from decision_sim.analysis.report import build_report
from decision_sim.analysis.whatif import run_whatif
from decision_sim.config import SimConfig
from decision_sim.intelligence.graph import NetworkxGraphStore, build_graph
from decision_sim.intelligence.ingest import ingest_uploaded_text
from decision_sim.ml.predict import churn_inputs_or_none
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
from decision_sim.schemas import Evidence, Post, Scenario, Stance
from decision_sim.simulation.graph import (
    final_stance_by_agent,
    iter_simulation,
    save_run,
    stances_dataframe,
)
from decision_sim.ui.helpers import build_evidence_by_agent, graph_edges_dataframe


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
                        if post.sources:
                            cites = ", ".join(
                                f"[{s.title}]({s.url})" for s in post.sources
                            )
                            st.caption(f":grey[Sources: {cites}]")


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
    business = scenario.business
    model_base, model_price = churn_inputs_or_none(
        price_increase_pct=business.price_increase_pct,
        base_churn=business.base_churn,
        max_extra_churn=business.max_extra_churn,
    )
    price_from_model = model_price is not None
    result = simulate_impact(
        agent_stances,
        business,
        model_base_churn=model_base,
        model_price_extra_churn=model_price,
    )

    st.subheader("Estimated impact (vs. no price rise)")
    st.caption(churn_assumption_text(business))
    if model_base is None:
        st.caption(
            ":grey[No trained churn model found — using the FCA base and the "
            "labelled assumption for the price-change line.]"
        )
    elif not price_from_model:
        st.caption(
            ":grey[Model loaded, but its price effect was implausible — using the "
            "labelled assumption for the price-change line.]"
        )
    else:
        st.caption(
            f":grey[Price-change line from the trained model "
            f"(+{model_price:.1%} extra churn, rescaled to the FCA base).]"
        )

    st.markdown("**Churn breakdown**")
    churn_cols = st.columns(4)
    breakdown = churn_breakdown_rows(result, price_from_model=price_from_model)
    for col, (label, value) in zip(churn_cols, breakdown, strict=True):
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


def _render_knowledge_graph(scenario: Scenario, evidence: list[Evidence]) -> None:
    """Extract and show stakeholder relationships grounded in the uploaded docs."""
    if not evidence:
        return
    st.subheader("Knowledge graph")
    try:
        store = build_graph(scenario.stakeholders, evidence, NetworkxGraphStore())
    except Exception as exc:  # noqa: BLE001 - never let extraction break the run
        st.caption(f":grey[Could not extract relationships: {exc}]")
        return
    edges = graph_edges_dataframe(store)
    if edges.empty:
        st.caption(":grey[No relationships were extracted from the uploaded documents.]")
    else:
        st.dataframe(edges, hide_index=True, use_container_width=True)


def _render_interview(scenario: Scenario, posts: list[Post], stances) -> None:
    """Let the user ask a stakeholder a question, grounded in its own statements."""
    st.subheader("Interview a stakeholder")
    agent_stances = final_stance_by_agent(stances)
    names = [a.name for a in scenario.stakeholders]
    with st.form("interview_form"):
        name = st.selectbox("Stakeholder", names, key="interview_agent")
        question = st.text_input("Question", key="interview_question")
        submitted = st.form_submit_button("Ask")
    if not submitted or not question.strip():
        return
    agent = next(a for a in scenario.stakeholders if a.name == name)
    with st.spinner(f"Asking {name}..."):
        answer, cited = interview_agent(
            agent, question.strip(), posts, agent_stances.get(name)
        )
    st.markdown(f"**{name}:** {answer.answer}")
    if cited:
        st.caption("Based on their own statements:")
        for post in cited:
            st.caption(f":grey[- ({post.stage} r{post.round}) {post.text}]")


def _render_whatif(scenario: Scenario, stances) -> None:
    """Counterfactual: re-score the impact with a changed customer stance / price."""
    if scenario.business is None:
        return
    st.subheader("What-if")
    st.caption(
        "Only customer stances drive churn, so the behavioural lever is a customer "
        "group's stance. Changing the price moves revenue but not churn in this model."
    )
    business = scenario.business
    agent_stances = final_stance_by_agent(stances)
    customer_agents = [g.agent for g in business.customer_groups]
    choosable = [a for a in customer_agents if a in agent_stances]
    if not choosable:
        st.info("No customer stances available to vary.")
        return

    with st.form("whatif_form"):
        agent = st.selectbox("Customer group agent", choosable)
        new_score = st.slider(
            "Their stance (hostile -1 ... +1 supportive)",
            min_value=-1.0,
            max_value=1.0,
            value=float(agent_stances[agent].score),
            step=0.05,
        )
        new_price = st.number_input(
            "Price increase (%)",
            value=float(business.price_increase_pct),
            step=1.0,
        )
        submitted = st.form_submit_button("Compare")
    if not submitted:
        return

    override: Stance = agent_stances[agent].model_copy(update={"score": new_score})
    comparison = run_whatif(
        agent_stances,
        business,
        stance_overrides={agent: override},
        price_increase_pct=new_price,
    )
    cols = st.columns(3)
    cols[0].metric("Baseline impact", format_gbp_millions(comparison.baseline_impact_mean))
    cols[1].metric(
        "What-if impact",
        format_gbp_millions(comparison.variant_impact_mean),
        format_gbp_millions(comparison.delta_mean),
    )
    cols[2].metric(
        "Churn change",
        f"{comparison.variant_churn_rate_mean:.1%}",
        f"{comparison.delta_churn_rate_mean:+.1%}",
    )



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
        st.subheader("Ground in documents")
        uploads = st.file_uploader(
            "Upload .txt / .md documents",
            type=["txt", "md"],
            accept_multiple_files=True,
            help="Agents cite these documents during the simulation.",
        )
        evidence: list[Evidence] = []
        if uploads:
            for upload in uploads:
                text = upload.getvalue().decode("utf-8", errors="replace")
                evidence.extend(ingest_uploaded_text(upload.name, text))
            doc_names = sorted({s.title for ev in evidence for s in ev.sources})
            st.caption(
                f"{len(doc_names)} document(s), {len(evidence)} chunk(s):"
            )
            for name in doc_names:
                st.write(f"- {name}")

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

    if run:
        evidence_by_agent = build_evidence_by_agent(scenario, evidence)
        final_state = None
        with st.spinner("Running simulation..."):
            for state in iter_simulation(
                scenario, config, evidence_by_agent=evidence_by_agent
            ):
                final_state = state
                _render_posts(posts_area, state["posts"])
                with chart_area.container():
                    _render_chart(scenario, state["stances"])
        if final_state is not None:
            path = save_run(final_state)
            st.session_state["last_run"] = {
                "scenario": scenario,
                "posts": final_state["posts"],
                "stances": final_state["stances"],
                "rounds_executed": final_state["rounds_executed"],
                "evidence": evidence,
                "saved_to": str(path),
            }

    last = st.session_state.get("last_run")
    if last is None:
        posts_col.info("Configure the run in the sidebar, then press **Run**.")
        return

    run_scenario: Scenario = last["scenario"]
    posts: list[Post] = last["posts"]
    stances = last["stances"]

    # After a rerun (e.g. from the interview/what-if forms) re-render the posts.
    if not run:
        _render_posts(posts_area, posts)
        with chart_area.container():
            _render_chart(run_scenario, stances)

    st.success(
        f"Run complete ({last['rounds_executed']} rounds). Saved to {last['saved_to']}"
    )
    _render_impact(run_scenario, stances)
    _render_analysis(run_scenario, posts, stances)
    _render_knowledge_graph(run_scenario, last["evidence"])
    _render_interview(run_scenario, posts, stances)
    _render_whatif(run_scenario, stances)


if __name__ == "__main__":
    main()
