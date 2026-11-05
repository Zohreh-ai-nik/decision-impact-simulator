"""Tests for the embedders. FastEmbed is heavy/optional, so it is skipped unless
installed; the model download happens on first use."""

from __future__ import annotations

import pytest

from decision_sim.intelligence.embeddings import LiteLLMEmbedder
from decision_sim.intelligence.rag import EmbeddingFn

# --- LiteLLMEmbedder (no network needed for these checks) --------------------


def test_litellm_embedder_requires_model(monkeypatch):
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    with pytest.raises(ValueError, match="embedding model"):
        LiteLLMEmbedder()


def test_litellm_embedder_reads_env(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL", "text-embedding-3-small")
    embedder = LiteLLMEmbedder()
    assert isinstance(embedder, EmbeddingFn)


# --- FastEmbedEmbedder (optional, downloads a small model on first use) ------


def test_fastembed_produces_consistent_vectors():
    pytest.importorskip("fastembed")
    from decision_sim.intelligence.embeddings import FastEmbedEmbedder

    embedder = FastEmbedEmbedder()
    vectors = embedder(["insurance price rise", "insurance price rise"])
    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1]) > 0
    assert vectors[0] == pytest.approx(vectors[1])  # deterministic for same text
    assert isinstance(embedder, EmbeddingFn)
