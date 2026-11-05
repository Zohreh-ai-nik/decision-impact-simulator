"""Load simulation scenarios from YAML into validated :class:`Scenario` objects."""

from __future__ import annotations

from pathlib import Path

import yaml

from decision_sim.schemas import Scenario

# Repository root is three levels up from this file: src/decision_sim/scenario.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIOS_DIR = _REPO_ROOT / "scenarios"


def load_scenario(path: str | Path) -> Scenario:
    """Load and validate a scenario from a YAML file.

    Args:
        path: Path to a YAML file describing a scenario.

    Returns:
        A validated :class:`Scenario`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        pydantic.ValidationError: If the data does not satisfy the schema.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Scenario file not found: {file_path}")
    data = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    return Scenario.model_validate(data)


def load_named(name: str) -> Scenario:
    """Load a scenario by name from the bundled ``scenarios/`` directory.

    Args:
        name: Scenario file name, with or without the ``.yaml`` suffix.
    """
    stem = name[:-5] if name.endswith(".yaml") else name
    return load_scenario(SCENARIOS_DIR / f"{stem}.yaml")


def list_scenarios(directory: str | Path = SCENARIOS_DIR) -> list[Path]:
    """Return the YAML scenario files available in ``directory``, sorted by name."""
    folder = Path(directory)
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.yaml"))


def add_signal(scenario: Scenario, text: str, stage: str, round: int) -> Scenario:
    """Return a new validated :class:`Scenario` with an extra signal appended.

    Args:
        scenario: The scenario to extend (left unmodified).
        text: The signal text.
        stage: Name of the stage the signal fires in; must exist in the scenario.
        round: Round number within the stage (1-based).

    Returns:
        A new :class:`Scenario` including the added signal.

    Raises:
        pydantic.ValidationError: If the resulting scenario is invalid (e.g. the
            stage does not exist).
    """
    data = scenario.model_dump()
    data["signals"].append({"text": text, "stage": stage, "round": round})
    return Scenario.model_validate(data)

