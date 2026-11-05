"""Run a scenario end-to-end and print the quantitative impact.

Runs the full staged simulation (real LLM calls), then feeds each agent's final
stance into the Monte Carlo impact model and prints the results. Example:

    python scripts/run_scenario.py insurer_price_rise
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from decision_sim.quant.formatting import (
    churn_assumption_text,
    churn_breakdown_rows,
    impact_summary_rows,
)
from decision_sim.quant.monte_carlo import MonteCarloConfig, simulate_impact
from decision_sim.scenario import load_named
from decision_sim.simulation.graph import final_stance_by_agent, run_simulation


def main(name: str) -> None:
    load_dotenv()
    scenario = load_named(name)
    if scenario.business is None:
        raise SystemExit(f"Scenario {name!r} has no business context to quantify.")

    print(f"Running scenario: {name}")
    total_rounds = sum(stage.rounds for stage in scenario.stages)
    stage_summary = [(s.name, s.rounds) for s in scenario.stages]
    print(f"Stages: {stage_summary} (total {total_rounds} rounds)")

    state = run_simulation(scenario)
    print(f"Rounds executed: {state['rounds_executed']}  |  posts: {len(state['posts'])}")

    stances = final_stance_by_agent(state["stances"])
    print("\nFinal stance by agent:")
    for agent, stance in sorted(stances.items()):
        print(f"  {agent:<22} score={stance.score:+.2f}  confidence={stance.confidence:.2f}")

    result = simulate_impact(stances, scenario.business, MonteCarloConfig())

    print("\n" + churn_assumption_text(scenario.business))
    print("\nChurn breakdown:")
    for label, value in churn_breakdown_rows(result):
        print(f"  {label:<40} {value}")

    print("\nImpact summary:")
    for label, value in impact_summary_rows(result):
        print(f"  {label:<40} {value}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "insurer_price_rise")
