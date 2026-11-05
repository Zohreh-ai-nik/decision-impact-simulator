# Decision Impact Simulator — Roadmap

A portfolio project that simulates how stakeholders react to a business decision,
then quantifies the impact. This file is the single source of truth for build
order. **Rules for every phase:** plan first, wait for approval, one phase at a
time, tests use fakes (never the network or a real LLM), append a dated entry to
`docs/devlog.md`.

## Done / in progress

1. **Source schema + provenance** — `Source`, `Evidence`; optional `sources` on
   `AgentProfile`, `Post`, `Stance`. ✅ done.
7. **Calibration with real sources** — in progress. Option C: I propose, user
   approves each source. Every figure needs the number, exact quote, URL,
   publisher and date. Never invent a figure.

## Next

2. **Evidence ingestion (documents + web)** behind a `SearchProvider` interface,
   with fake/cached versions for offline tests.
3. **RAG retrieval with Qdrant**, behind a `VectorStore` interface (in-memory
   fake for tests).
4. **Evidence-grounded agent profiles** — each profile carries its sources.
5. **Knowledge graph with Cognee + Neo4j** (kept, not optional; networkx fake for
   tests). Show uploaded documents and the graph in the Streamlit UI.
6. **Grounded simulation** — agents receive retrieved evidence and cite sources;
   the UI shows them.

8. **Churn model on real data** (base churn for the simulation).
   - Data: public `eudirectlapse` dataset (CASdatasets, GPL >= 2), 23,060
     European motor insurance renewals, target = lapse (overall lapse rate 12.8%).
   - Docs: https://dutangc.github.io/CASdatasets/reference/eudirectlapse.html
   - Raw: https://raw.githubusercontent.com/dutangc/CASdatasets/master/data/eudirectlapse.rda
   - a. `scripts/download_data.py`: download the `.rda` to `data/raw/`, convert to
     parquet with pyreadr. `data/` stays git-ignored. Record source + licence in
     `docs/data.md`.
   - b. `notebooks/01_churn_eda.ipynb`: lapse rate overall and by driver age,
     policy age (years with insurer), number of policies, price change
     (`prem_final/prem_last - 1`) and price vs market (`prem_final/prem_market - 1`).
     Quick check: under-30s lapse ~17%, over-60s ~9%; 3+ years ~9%; raw lapse rate
     does NOT rise with price change (likely confounding — bigger rises given to
     stickier customers). Investigate and explain; don't hide it.
   - c. `src/decision_sim/ml/`: `data.py` (cleaning + features), `train.py`,
     `predict.py`. Baseline: logistic regression. Main model:
     `HistGradientBoostingClassifier`. Features include price change % and price
     vs market %. Stratified split, fixed seed. Report ROC AUC, Brier score and a
     calibration plot. Save model to `models/` (git-ignored). Log to MLflow.
   - d. `predict_churn(portfolio, price_change) -> churn probability per customer`.
     Build the fictional ShieldDrive portfolio by resampling dataset rows (seeded).
   - e. In `quant/`: the model gives base churn; customer agents' stance only
     shifts it (capped). Monte Carlo also samples model uncertainty. Show "model
     base churn" and "agent adjustment" separately in the UI.
   - f. Tests use a tiny fixture dataset (no download in tests). New deps:
     scikit-learn, pyreadr, mlflow (explain each in pyproject).

9. **Compare decision options side by side** (e.g. 12% rise / 6% rise + loyalty
   discount / no change): churn, revenue range and stakeholder reaction for each,
   plus a one-page CEO summary.
10. **Agent memory with Mem0**: store what each agent said and read, retrieve
    relevant memories before posting; each agent keeps a short "current view" per
    stage.
11. **Analysis extras**: LLM executive summary report, agent interviews (one agent
    or a group answers in character), what-if (save state each round, rerun from a
    chosen round with new info, compare side by side).
12. **Sensitivity analysis** (tornado chart), **MLflow run tracking** (scenario,
    model, prompt version, seed, cost, impact range), **MCP server** (FastMCP)
    exposing `search_context` / `get_relationships` / `estimate_impact` /
    `predict_churn`, regulator agent as a separate **A2A** service.
13. **Two more fictional scenarios** to show it works for any decision:
    - Risk: a new regulation bans price walking.
    - Transaction: ShieldDrive acquires a smaller rival.
14. **Split into services, talking over gRPC**:
    - platform (FastAPI, JWT login, saved projects in PostgreSQL)
    - intelligence (upload, search, graph, agent creation)
    - simulation (rounds, memory, signals)
    - analysis + impact (coalitions, influence, Monte Carlo, churn model)
    `.proto` calls: `CreateAgents`, `RunSimulation` (server-streaming posts to the
    UI), `InjectSignal`, `PredictChurn`, `GetResults`. Docker Compose starts
    everything (UI, services, PostgreSQL, Neo4j, Qdrant, MLflow).
15. **Evaluation + showcase**: consistency across seeds, diversity (no fake
    consensus), grounding rate, churn model metrics, back-test on a public past
    event; GitHub Actions running ruff and pytest; README with architecture
    diagram, results, limitations and a demo GIF.
