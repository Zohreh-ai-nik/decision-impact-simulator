"""Shared pytest fixtures, including a fake LLM for offline testing."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import pytest


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    """Minimal stand-in for a LiteLLM completion response."""

    choices: list[_Choice]


@dataclass
class _Call:
    """Record of a single call made to the fake LLM."""

    model: str
    messages: list[dict[str, str]]
    temperature: float | None
    response_format: Any


@dataclass
class FakeLLM:
    """Drop-in replacement for ``litellm.completion`` used in tests.

    Returns each queued string as a completion response in order, recording the
    arguments of every call for later assertions. If the queue is empty and a
    ``default`` is set, that value is returned for every further call -- handy
    for round-based simulations that make many identical requests.
    """

    responses: list[str] = field(default_factory=list)
    calls: list[_Call] = field(default_factory=list)
    default: str | None = None

    def queue(self, *contents: str) -> FakeLLM:
        """Append one or more response bodies to the queue."""
        self.responses.extend(contents)
        return self

    def __call__(
        self,
        *,
        model: str,
        messages: Iterable[dict[str, str]],
        temperature: float | None = None,
        response_format: Any = None,
        **_: Any,
    ) -> _Response:
        self.calls.append(
            _Call(
                model=model,
                messages=list(messages),
                temperature=temperature,
                response_format=response_format,
            )
        )
        if self.responses:
            content = self.responses.pop(0)
        elif self.default is not None:
            content = self.default
        else:
            raise AssertionError("FakeLLM called more times than responses were queued.")
        return _Response(choices=[_Choice(message=_Message(content=content))])


@pytest.fixture
def fake_llm() -> FakeLLM:
    """Provide an empty :class:`FakeLLM`; queue responses in the test."""
    return FakeLLM()
