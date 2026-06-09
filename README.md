# Oxford Domestic EPC: Energy-Rating Prediction and Retrofit Analysis

Predicting the Energy Performance Certificate (EPC) band of Oxford's domestic
housing stock from physical building fabric, identifying the drivers of energy
efficiency, and turning the findings into prioritised retrofit recommendations.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/parsecsion/oxford-epc/blob/main/notebooks/EPC_Oxford_Analysis.ipynb)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: OGL v3.0](https://img.shields.io/badge/data-OGL%20v3.0-brightgreen.svg)](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)

---

## Overview

Energy Performance Certificates rate dwellings from **A** (most efficient) to
**G** (least). The rating is a deterministic discretisation of an underlying
**SAP score** (1–100). This project learns that relationship from *physical
fabric only*, meaning walls, glazing, heating, age and floor area, while rigorously
excluding the SAP-engine outputs that would otherwise leak the target.

The result is a model that ranks dwellings for retrofit prioritisation across
the Oxford stock, accompanied by an interpretable account of *why* a property
scores as it does and *which* improvement measures the evidence supports.

## Key results

On a strict post-2022 temporal hold-out (*n* = 12,095):

| Metric | Value |
| --- | --- |
| Quadratic Weighted Kappa (primary) | **0.7696**  (0.7714 ± 0.0019 over 3 seeds) |
| Within 15 SAP points | **97.7%** (exceeds the ONS Data Science Campus 2021 UK benchmark of 93%) |
| Within 10 / 5 SAP points | 92.5% / 71.4% |
| Score MAE | 4.04 SAP points (≈ the inter-assessor noise floor) |
| MAE in band units | 0.32 (under a third of a band) |

The champion, **`SapStratifiedRegressor`**, is a hybrid regression-to-band
model: a `HistGradientBoostingRegressor` predicts the continuous SAP score,
which is mapped to a band via the official DLUHC thresholds, with a dedicated
specialist head for the SAP-assessed cohort (`REPORT_TYPE == 101`). Every
headline figure is enforced by **95 automated checks** (`scripts/verify_*.py`).

## Quickstart

### Run in the browser (no setup)

Click the **Open in Colab** badge above and choose *Runtime → Run all*. The
notebook's first cell fetches this repository automatically.

### Run locally

```bash
git clone https://github.com/parsecsion/oxford-epc.git
cd oxford-epc
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab notebooks/EPC_Oxford_Analysis.ipynb       # Kernel → Restart & Run All
```

Reproduce the production pipeline and re-validate every metric:

```bash
python scripts/sap_stratified.py        # hybrid experiment + multi-seed stability
python scripts/freeze_champion.py       # write artefacts/champion.pkl (+ sha256)
python scripts/predict_oxford.py        # per-certificate predictions
python scripts/champion_robustness.py   # robustness metrics + figures
python scripts/verify_data_pipeline.py  # 25/25 checks
python scripts/verify_model_pipeline.py # 70/70 checks
```

See [`HOW_TO_RUN.md`](HOW_TO_RUN.md) for the full operator runbook.

## Methodology

- **Leakage control.** SAP-engine outputs (potential ratings, CO₂, costs,
  per-element efficiency stars) are retained in the published dataset for
  analysis but stripped from the feature matrix before training.
- **Group-aware temporal split.** Train on certificates lodged ≤ 2022-12-31,
  test on later ones; any dwelling (`UPRN`) appearing after the cutoff is held
  out entirely, so repeat lodgements cannot leak across folds.
- **Regression-to-band.** Modelling the continuous SAP score and thresholding
  is a consistent estimator under known cut-offs, and supports counterfactual
  queries (e.g. the SAP impact of added insulation) that a discrete classifier
  cannot express.
- **Interpretability.** Permutation importance scored directly on QWK (avoiding
  the cardinality bias of impurity importance) plus SHAP on the regression head.
- **Segmentation & recommendations.** K-Prototypes clustering of the mixed-type
  stock, and a retrofit recommender combining association rules with a
  content-based k-NN, cross-checked against the SAP engine's own suggestions.

## Repository structure

```
.
├── notebooks/EPC_Oxford_Analysis.ipynb   # end-to-end analysis walk-through
├── src/                                  # importable package
│   ├── data.py            # typed load, leakage filter, dedup, group split
│   ├── features.py        # feature engineering
│   ├── models.py          # estimator factories + SapStratifiedRegressor
│   ├── evaluation.py      # QWK, MAE-bands, linear kappa, bootstrap CIs
│   ├── clustering.py      # K-Prototypes segmentation
│   ├── plots.py           # publication figures
│   └── recs.py            # recommendations analysis
├── scripts/                              # reproducible pipeline + verifiers
├── data/processed/oxford_epc_clean.csv   # cleaned, GDPR-stripped dataset
├── artefacts/champion.pkl                # frozen, sha256-pinned model
├── reports/                              # metrics (JSON), figures, written report
├── docs/                                 # data card & model card
├── certificates.csv, recommendations.csv # raw EPC source data
├── DATA_PIPELINE.md, HOW_TO_RUN.md       # documentation
└── requirements.txt
```

## Data & licensing

Source: the **Energy Performance of Buildings Register** (England & Wales),
DLUHC / MHCLG, filtered to local authority **E07000178 (Oxford)**. Non-address
fields are released under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/);
address-level fields fall under a restricted Ordnance Survey / Royal Mail
licence and are excluded from the published dataset. Personal identifiers are
removed in line with GDPR data-minimisation. See [`docs/data_card.md`](docs/data_card.md).

## Reproducibility

- Python 3.11 with pinned dependencies (`requirements.txt`).
- Global seed `42`; the champion is validated across seeds 42, 123, 2026.
- All preprocessing lives inside a single `sklearn.Pipeline` fit only on the
  training fold.
- The frozen model is content-hashed (sha256) and re-validated on every load.

## Author

**Parsec**, COM6003 Data Science, Buckinghamshire New University.
