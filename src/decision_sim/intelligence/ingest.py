"""Document ingestion: turn local files into citable :class:`Evidence` chunks.

This is deliberately dependency-light (stdlib + Pydantic). It reads plain-text
and Markdown files, splits them into overlapping chunks, and wraps each chunk in
an :class:`Evidence` with a ``file://`` :class:`Source`, so an uploaded document
is cited exactly like a web result. Vector indexing / retrieval comes later
(Phase 2d).

Security note: document text is treated strictly as **data**. It is only ever
stored in ``Evidence.text`` and never interpreted as instructions here.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from decision_sim.schemas import Evidence, Source

__all__ = [
    "SUPPORTED_SUFFIXES",
    "chunk_text",
    "ingest_dir",
    "ingest_file",
    "ingest_uploaded_text",
    "load_text",
]

#: File types this module can read today. PDF/doc parsing is deferred.
SUPPORTED_SUFFIXES = (".txt", ".md")


def load_text(path: str | Path) -> str:
    """Read a supported text file as UTF-8.

    Raises :class:`FileNotFoundError` if the file is missing and
    :class:`ValueError` for an unsupported extension (an explicit boundary so
    callers get a clear message instead of garbled bytes).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No such file: {p}")
    if p.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type {p.suffix!r}; supported: {', '.join(SUPPORTED_SUFFIXES)}."
        )
    return p.read_text(encoding="utf-8")


def chunk_text(text: str, *, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping character windows.

    Overlap keeps a fact that straddles a boundary intact for later retrieval.
    Returns an empty list for blank text. ``chunk_size`` must be positive and
    ``overlap`` must be smaller than it.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
    if not 0 <= overlap < chunk_size:
        raise ValueError(f"overlap must be in [0, chunk_size), got {overlap}.")
    stripped = text.strip()
    if not stripped:
        return []
    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(stripped), step):
        chunk = stripped[start : start + chunk_size]
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(stripped):
            break
    return chunks


def _source_for(path: Path, snippet: str) -> Source:
    return Source(
        url=path.resolve().as_uri(),  # file:// absolute URI
        title=path.name,
        snippet=snippet[:200],
    )


def ingest_file(
    path: str | Path, *, chunk_size: int = 800, overlap: int = 100
) -> list[Evidence]:
    """Read one file and return one :class:`Evidence` per chunk, cited to it."""
    p = Path(path)
    text = load_text(p)
    return [
        Evidence(text=chunk, sources=[_source_for(p, chunk)])
        for chunk in chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    ]


def ingest_uploaded_text(
    name: str,
    text: str,
    *,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[Evidence]:
    """Turn an in-memory uploaded document into citable :class:`Evidence` chunks.

    Unlike :func:`ingest_file` this never touches the filesystem, so it suits a
    web upload. Each chunk cites a synthetic ``file://uploads/<name>`` source, so
    uploaded documents are cited exactly like files and web results. ``name`` is
    sanitised to a safe, path-free filename for the URL and title.
    """
    safe_name = Path(name).name.strip() or "document"
    quoted = quote(safe_name)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    return [
        Evidence(
            text=chunk,
            sources=[
                Source(
                    url=f"file://uploads/{quoted}",
                    title=safe_name,
                    snippet=chunk[:200],
                )
            ],
        )
        for chunk in chunks
    ]


def ingest_dir(
    directory: str | Path,
    *,
    pattern: str = "*.md",
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[Evidence]:
    """Ingest every file in ``directory`` matching ``pattern`` (sorted by name)."""
    d = Path(directory)
    if not d.is_dir():
        raise NotADirectoryError(f"No such directory: {d}")
    evidence: list[Evidence] = []
    for file in sorted(d.glob(pattern)):
        if file.is_file():
            evidence.extend(
                ingest_file(file, chunk_size=chunk_size, overlap=overlap)
            )
    return evidence
