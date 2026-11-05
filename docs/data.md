# Data sources and licences

Datasets used by the project. Raw and processed data live under `data/`, which is
git-ignored and never committed.

## eudirectlapse (CASdatasets)

- **What:** European motor insurance renewal dataset from the direct channel.
  23,060 policies (one-year renewal quotes), binary `lapse` target. Confirmed from
  the reference page: "a data frame of 19 columns and 23,060 rows".
- **Publisher:** CASdatasets — Christophe Dutang & Arthur Charpentier.
- **Reference:** https://dutangc.github.io/CASdatasets/reference/eudirectlapse.html
- **Raw file:** https://raw.githubusercontent.com/dutangc/CASdatasets/master/data/eudirectlapse.rda
- **Licence:** GPL (>= 2), per the CASdatasets package.
- **Use in this project:** churn *patterns* only (Phase 8), not the UK base churn
  level. See `docs/calibration.md` for why (it is an unknown European direct
  insurer with ~12.8% lapse, far below the UK motor ~40% attrition).
- **Handling:** downloaded by `scripts/download_data.py` to `data/raw/` and
  converted to parquet. The dataset is **not** redistributed in this repository;
  users download it themselves from the source above.
