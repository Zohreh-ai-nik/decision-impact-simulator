"""Download the public eudirectlapse churn dataset and convert it to parquet.

Source: CASdatasets (Christophe Dutang & Arthur Charpentier), GPL (>= 2).
    https://dutangc.github.io/CASdatasets/reference/eudirectlapse.html

The dataset is a European *direct* motor insurer's renewal data (23,060 rows,
19 columns, binary ``lapse`` target). It is used only for churn *patterns*
(relative risk between customers and the price-change effect), never for the UK
base churn level — see ``docs/calibration.md`` and ``docs/data.md``.

The raw and processed files live under ``data/`` which is git-ignored; the data
is never committed or redistributed. Run this once before training:

    python scripts/download_data.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pyreadr

RDA_URL = "https://raw.githubusercontent.com/dutangc/CASdatasets/master/data/eudirectlapse.rda"
DATA_RAW = Path("data/raw")
RDA_PATH = DATA_RAW / "eudirectlapse.rda"
PARQUET_PATH = DATA_RAW / "eudirectlapse.parquet"

EXPECTED_ROWS = 23_060
EXPECTED_COLS = 19


def download(url: str = RDA_URL, dest: Path = RDA_PATH) -> Path:
    """Download the ``.rda`` file to ``dest`` (idempotent)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Already downloaded: {dest} ({dest.stat().st_size:,} bytes)")
        return dest
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 - trusted CASdatasets URL
    print(f"Downloaded {dest.stat().st_size:,} bytes")
    return dest


def convert(rda_path: Path = RDA_PATH, parquet_path: Path = PARQUET_PATH) -> Path:
    """Read the ``.rda`` into a DataFrame and write it as parquet."""
    result = pyreadr.read_r(str(rda_path))
    # An .rda holds named objects; take the single data frame it contains.
    if not result:
        raise ValueError(f"No objects found in {rda_path}.")
    name, df = next(iter(result.items()))
    print(f"Loaded R object {name!r}: {df.shape[0]:,} rows x {df.shape[1]} columns")

    if (df.shape[0], df.shape[1]) != (EXPECTED_ROWS, EXPECTED_COLS):
        raise ValueError(
            f"Unexpected shape {df.shape}; expected "
            f"({EXPECTED_ROWS}, {EXPECTED_COLS}). Has the source changed?"
        )

    df.to_parquet(parquet_path)
    print(f"Wrote {parquet_path} ({parquet_path.stat().st_size:,} bytes)")
    return parquet_path


def main() -> None:
    download()
    convert()
    print("Done. Dataset ready for training (data/ is git-ignored).")


if __name__ == "__main__":
    main()
