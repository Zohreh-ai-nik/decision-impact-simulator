# decision-impact-simulator
Simulate how stakeholders react to a business decision using LLM agents, a knowledge graph and Monte Carlo

## Getting started

### Requirements
- Python 3.10 or newer
- An OpenAI API key (or another provider supported by LiteLLM)

### 1. Clone the repository
```bash
git clone https://github.com/Zohreh-ai-nik/decision-impact-simulator.git
cd decision-impact-simulator
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install the project
```bash
pip install -e ".[dev]"
```

### 4. Add your API key
```bash
cp .env.example .env
```
Then open `.env` and set `OPENAI_API_KEY` and `LLM_MODEL` (e.g. `gpt-4o-mini`).
`.env` is git-ignored, so keys never reach GitHub.

### 5. Check everything works
```bash
pytest -q
```

### How the setup works

| | `.venv` | `pyproject.toml` |
|---|---|---|
| What it is | A folder containing a private Python environment for this project | A file describing the project and what it needs |
| Role | **Where** packages are installed | **Which** packages to install, plus project and tool settings |
| In the repository | No (git-ignored; each person creates their own) | Yes |

- **`.venv`** keeps this project's packages separate from other projects and from the system Python, so versions never clash.
- **`pyproject.toml`** lists the project's dependencies (`litellm`, `pydantic`, `langgraph`, …), development tools (`pytest`, `ruff`) and tool settings, so the whole setup is reproducible with one command.
- **`pip install -e ".[dev]"`** reads `pyproject.toml` and installs everything into the active `.venv`:
  - `.` means "this project"
  - `[dev]` also installs the development tools
  - `-e` (editable) installs the code in `src/` as a package, so `from decision_sim.llm import ask` works anywhere, and code changes take effect without reinstalling.
