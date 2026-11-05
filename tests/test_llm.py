"""Tests for :mod:`decision_sim.llm` using the FakeLLM fixture."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from decision_sim import llm
from tests.conftest import FakeLLM


class Sentiment(BaseModel):
    label: str
    score: float


@pytest.fixture(autouse=True)
def _set_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a model via env so ask() never needs a real provider."""
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-4o-mini")


def _patch(monkeypatch: pytest.MonkeyPatch, fake: FakeLLM) -> None:
    monkeypatch.setattr(llm, "completion", fake)


def test_ask_returns_validated_instance(
    monkeypatch: pytest.MonkeyPatch, fake_llm: FakeLLM
) -> None:
    fake_llm.queue('{"label": "positive", "score": 0.9}')
    _patch(monkeypatch, fake_llm)

    result = llm.ask("How do you feel?", Sentiment)

    assert isinstance(result, Sentiment)
    assert result.label == "positive"
    assert result.score == pytest.approx(0.9)
    assert len(fake_llm.calls) == 1


def test_ask_passes_temperature_model_and_system(
    monkeypatch: pytest.MonkeyPatch, fake_llm: FakeLLM
) -> None:
    fake_llm.queue('{"label": "neutral", "score": 0.5}')
    _patch(monkeypatch, fake_llm)

    llm.ask("rate this", Sentiment, system="You are a rater.")

    call = fake_llm.calls[0]
    assert call.temperature == 0
    assert call.model == "openai/gpt-4o-mini"
    assert call.response_format is Sentiment
    assert call.messages[0] == {"role": "system", "content": "You are a rater."}
    assert call.messages[1] == {"role": "user", "content": "rate this"}


def test_ask_omits_system_when_not_provided(
    monkeypatch: pytest.MonkeyPatch, fake_llm: FakeLLM
) -> None:
    fake_llm.queue('{"label": "neutral", "score": 0.5}')
    _patch(monkeypatch, fake_llm)

    llm.ask("rate this", Sentiment)

    call = fake_llm.calls[0]
    assert all(m["role"] != "system" for m in call.messages)


def test_ask_retries_then_succeeds(
    monkeypatch: pytest.MonkeyPatch, fake_llm: FakeLLM
) -> None:
    fake_llm.queue(
        "not json at all",
        '{"label": "positive", "score": 0.8}',
    )
    _patch(monkeypatch, fake_llm)

    result = llm.ask("How do you feel?", Sentiment)

    assert result.label == "positive"
    assert len(fake_llm.calls) == 2
    # The retry should feed the validation error back to the model.
    assert any("did not match the required schema" in m["content"]
               for m in fake_llm.calls[1].messages)


def test_ask_raises_after_exhausting_retries(
    monkeypatch: pytest.MonkeyPatch, fake_llm: FakeLLM
) -> None:
    fake_llm.queue(
        '{"label": "x"}',  # missing required 'score'
        '{"label": "x"}',
        '{"label": "x"}',
    )
    _patch(monkeypatch, fake_llm)

    with pytest.raises(ValidationError):
        llm.ask("How do you feel?", Sentiment, max_retries=2)

    # 1 initial attempt + 2 retries == 3 calls.
    assert len(fake_llm.calls) == 3


def test_ask_requires_model(monkeypatch: pytest.MonkeyPatch, fake_llm: FakeLLM) -> None:
    monkeypatch.delenv("LLM_MODEL", raising=False)
    _patch(monkeypatch, fake_llm)

    with pytest.raises(RuntimeError):
        llm.ask("hi", Sentiment)
