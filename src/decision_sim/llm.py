"""LLM access layer backed by LiteLLM with Pydantic-validated structured output.

The public entry point is :func:`ask`, which sends a prompt to the configured
model and returns a validated instance of a Pydantic schema. If the model's
response fails validation, the call is retried (with the validation error fed
back to the model) up to ``max_retries`` times.

``completion`` is exposed at module level so tests can monkeypatch it with a
fake instead of hitting a real provider.
"""

from __future__ import annotations

import json
import os
from typing import TypeVar

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

# Indirection point so tests can swap in a fake LLM via monkeypatch.
completion = litellm.completion

T = TypeVar("T", bound=BaseModel)

_ENV_LOADED = False


def _ensure_env_loaded() -> None:
    """Load variables from a local ``.env`` file once per process."""
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_dotenv()
        _ENV_LOADED = True


def _resolve_model(model: str | None) -> str:
    """Return the explicit model or fall back to the ``LLM_MODEL`` env var."""
    resolved = model or os.environ.get("LLM_MODEL")
    if not resolved:
        raise RuntimeError(
            "No model configured: pass model=... or set the LLM_MODEL environment variable."
        )
    return resolved


def _extract_content(response: object) -> str:
    """Pull the text content out of a LiteLLM-style completion response."""
    try:
        content = response.choices[0].message.content  # type: ignore[attr-defined]
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError(f"Malformed completion response: {response!r}") from exc
    if content is None:
        raise ValueError("Completion response contained no content.")
    return content


def ask(
    prompt: str,
    schema: type[T],
    system: str | None = None,
    *,
    model: str | None = None,
    max_retries: int = 2,
) -> T:
    """Ask the LLM for structured output and return it validated against ``schema``.

    Args:
        prompt: The user prompt.
        schema: A Pydantic model class the response must validate against.
        system: Optional system prompt prepended to the conversation.
        model: Model identifier (LiteLLM format). Defaults to the ``LLM_MODEL``
            environment variable.
        max_retries: Number of additional attempts if validation fails. With the
            default of 2 the model is called at most 3 times.

    Returns:
        A validated instance of ``schema``.

    Raises:
        ValidationError: If the response still fails validation after all retries.
        RuntimeError: If no model is configured.
    """
    _ensure_env_loaded()
    resolved_model = _resolve_model(model)

    messages: list[dict[str, str]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_error: ValidationError | None = None

    for attempt in range(max_retries + 1):
        response = completion(
            model=resolved_model,
            messages=messages,
            temperature=0,
            response_format=schema,
        )
        content = _extract_content(response)

        try:
            return schema.model_validate_json(content)
        except ValidationError as exc:
            last_error = exc
            if attempt < max_retries:
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response did not match the required schema. "
                            f"Validation errors:\n{json.dumps(exc.errors(), default=str)}\n"
                            "Return only valid JSON that satisfies the schema."
                        ),
                    }
                )

    assert last_error is not None  # loop runs at least once
    raise last_error
