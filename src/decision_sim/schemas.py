"""Pydantic models describing a decision-impact simulation scenario.

These schemas define the structured vocabulary used throughout the simulator:
the stakeholders (:class:`AgentProfile`), the timeline (:class:`Stage`), the
communication topology (:class:`Channel`), the messages exchanged
(:class:`Post`), an agent's reaction (:class:`Stance`), external events
(:class:`Signal`) and the overall :class:`Scenario` that ties them together.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

AgentType = Literal["person", "organisation", "group"]


class AgentProfile(BaseModel):
    """A stakeholder participating in the simulation."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    type: AgentType
    role: str = Field(min_length=1)
    goals: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    style: str = Field(default="", description="How this agent tends to communicate.")


class Stage(BaseModel):
    """A phase of the simulation with a goal and a number of rounds."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    rounds: int = Field(ge=1)


class Channel(BaseModel):
    """A communication channel and which agents may post to / read from it."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    can_post: list[str] = Field(default_factory=list)
    can_read: list[str] = Field(default_factory=list)


class Post(BaseModel):
    """A single message posted by an agent to a channel during a round."""

    model_config = ConfigDict(extra="forbid")

    agent: str = Field(min_length=1)
    channel: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    round: int = Field(ge=1)
    text: str = Field(min_length=1)


class Stance(BaseModel):
    """An agent's reaction to the decision at a point in time."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(
        ge=-1.0, le=1.0, description="Sentiment from -1 (hostile) to 1 (supportive)."
    )
    likely_action: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)


class Signal(BaseModel):
    """An external event injected into the simulation at a given stage/round."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    round: int = Field(ge=1)


class CustomerGroup(BaseModel):
    """A segment of the customer base represented by a single stakeholder agent."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    agent: str = Field(min_length=1, description="Stakeholder name that represents this group.")
    share: float = Field(gt=0.0, le=1.0, description="Fraction of the customer base in [0, 1].")


class BusinessContext(BaseModel):
    """Quantitative business inputs used to turn stances into a revenue impact."""

    model_config = ConfigDict(extra="forbid")

    customer_base: int = Field(ge=1, description="Total number of customers.")
    annual_premium: float = Field(gt=0.0, description="Current average annual premium.")
    price_increase_pct: float = Field(description="Percentage change applied to the premium.")
    customer_groups: list[CustomerGroup] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_groups(self) -> BusinessContext:
        names = [g.name for g in self.customer_groups]
        seen: set[str] = set()
        for name in names:
            if name in seen:
                raise ValueError(f"Duplicate customer group name: {name!r}.")
            seen.add(name)

        total_share = sum(g.share for g in self.customer_groups)
        if total_share > 1.0 + 1e-9:
            raise ValueError(
                f"Customer group shares sum to {total_share:.4f}, which exceeds 1.0."
            )
        return self


class ImpactResult(BaseModel):
    """Monte Carlo summary of the revenue impact of a decision."""

    model_config = ConfigDict(extra="forbid")

    n_runs: int = Field(ge=1)
    baseline_revenue: float
    new_annual_premium: float
    churn_rate_mean: float = Field(ge=0.0, le=1.0)
    lost_customers_mean: float = Field(ge=0.0)
    revenue_impact_mean: float
    revenue_impact_std: float = Field(ge=0.0)
    revenue_impact_p10: float
    revenue_impact_p50: float
    revenue_impact_p90: float
    samples: list[float] = Field(default_factory=list)


class Scenario(BaseModel):
    """A complete decision-impact scenario."""

    model_config = ConfigDict(extra="forbid")

    decision: str = Field(min_length=1)
    client: str = Field(min_length=1)
    stakeholders: list[AgentProfile] = Field(min_length=1)
    stages: list[Stage] = Field(min_length=1)
    channels: list[Channel] = Field(min_length=1)
    signals: list[Signal] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    business: BusinessContext | None = None

    @model_validator(mode="after")
    def _check_references(self) -> Scenario:
        agent_names = [a.name for a in self.stakeholders]
        stage_names = [s.name for s in self.stages]
        channel_names = [c.name for c in self.channels]

        self._require_unique(agent_names, "stakeholder")
        self._require_unique(stage_names, "stage")
        self._require_unique(channel_names, "channel")

        agent_set = set(agent_names)
        for channel in self.channels:
            for who in (*channel.can_post, *channel.can_read):
                if who not in agent_set:
                    raise ValueError(
                        f"Channel {channel.name!r} references unknown agent {who!r}."
                    )

        stage_set = set(stage_names)
        for signal in self.signals:
            if signal.stage not in stage_set:
                raise ValueError(
                    f"Signal references unknown stage {signal.stage!r}."
                )

        if self.business is not None:
            for group in self.business.customer_groups:
                if group.agent not in agent_set:
                    raise ValueError(
                        f"Customer group {group.name!r} references unknown agent "
                        f"{group.agent!r}."
                    )

        return self

    @staticmethod
    def _require_unique(names: list[str], label: str) -> None:
        seen: set[str] = set()
        for name in names:
            if name in seen:
                raise ValueError(f"Duplicate {label} name: {name!r}.")
            seen.add(name)
