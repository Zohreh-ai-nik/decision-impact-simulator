"""Offline tests for document ingestion."""

from __future__ import annotations

import pytest

from decision_sim.intelligence.ingest import (
    chunk_text,
    ingest_dir,
    ingest_file,
    load_text,
)

# --- load_text --------------------------------------------------------------


def test_load_text_reads_utf8(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("héllo wörld", encoding="utf-8")
    assert load_text(p) == "héllo wörld"


def test_load_text_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_text(tmp_path / "nope.md")


def test_load_text_rejects_unsupported_suffix(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_text(p)


# --- chunk_text -------------------------------------------------------------


def test_chunk_text_short_is_single_chunk():
    assert chunk_text("short text", chunk_size=800) == ["short text"]


def test_chunk_text_empty_is_no_chunks():
    assert chunk_text("   \n  ") == []


def test_chunk_text_respects_size_and_overlap():
    text = "abcdefghij"  # 10 chars
    chunks = chunk_text(text, chunk_size=4, overlap=1)
    # step = 3 -> starts 0,3,6; the chunk at 6 reaches the end, so no tiny tail
    assert chunks == ["abcd", "defg", "ghij"]


def test_chunk_text_overlap_preserves_boundary_text():
    text = "0123456789"
    chunks = chunk_text(text, chunk_size=5, overlap=2)
    assert chunks[0][-2:] == chunks[1][:2]  # overlap region matches


def test_chunk_text_rejects_bad_params():
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size=4, overlap=4)


# --- ingest_file ------------------------------------------------------------


def test_ingest_file_one_evidence_per_chunk(tmp_path):
    p = tmp_path / "report.md"
    p.write_text("0123456789", encoding="utf-8")
    evidence = ingest_file(p, chunk_size=4, overlap=1)
    assert len(evidence) == len(chunk_text("0123456789", chunk_size=4, overlap=1))
    assert evidence[0].text == "0123"


def test_ingest_file_sources_are_file_uris(tmp_path):
    p = tmp_path / "policy.md"
    p.write_text("some grounded fact", encoding="utf-8")
    ev = ingest_file(p)[0]
    src = ev.sources[0]
    assert src.url.startswith("file://")
    assert src.title == "policy.md"
    assert src.snippet  # preview populated


def test_ingest_file_empty_document_yields_nothing(tmp_path):
    p = tmp_path / "empty.md"
    p.write_text("   ", encoding="utf-8")
    assert ingest_file(p) == []


# --- ingest_dir -------------------------------------------------------------


def test_ingest_dir_picks_up_matching_files(tmp_path):
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.md").write_text("beta", encoding="utf-8")
    (tmp_path / "skip.txt").write_text("ignored", encoding="utf-8")
    evidence = ingest_dir(tmp_path, pattern="*.md")
    texts = [e.text for e in evidence]
    assert texts == ["alpha", "beta"]  # sorted, .txt skipped


def test_ingest_dir_missing_directory(tmp_path):
    with pytest.raises(NotADirectoryError):
        ingest_dir(tmp_path / "nope")
