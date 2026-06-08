# How to run this project

Two workflows are supported:

- **A. Production script pipeline** — the canonical workflow. Produces every
  numeric artefact, hashes the champion, and emits the per-UPRN deliverable.
  All numbers in the report and the model card trace to JSON files produced
  here, and both verifiers exit 0 (95/95 checks) on a clean run.
- **B. Notebook walk-through** — `notebooks/EPC_Oxford_Analysis.ipynb`, the
  graded walk-through. Imports the same `src/` modules, can be `Run All`-ed
  on a fresh kernel, and is the artefact you submit on Blackboard.

## 0. Prerequisites

- Python 3.11 (other 3.10+ versions should work but are not pinned).
- ~4 GB free RAM during model training and SHAP computation.

## 1. Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If TensorFlow is awkward on your platform you can comment it out of
`requirements.txt` — the production champion is gradient-boosted trees, not
the optional CORN ordinal NN.

## 2. Workflow A — production pipeline

Each step is independent and re-runnable. Run from the project root.

### 2.1 Data side — verify the cleaned dataset is reproducible

```bash
python scripts/verify_data_pipeline.py
```

Re-loads raw `certificates.csv`, re-cleans, re-deduplicates, re-writes
`data/processed/oxford_epc_clean.csv`, regenerates
`reports/summary_stats.csv` and `reports/data_quality_report.csv`, and runs
25 completion checks (no PII, no UPRN leakage between train/test, no zero
floor areas, reproducibility hashes match, etc.). Should exit 0.

### 2.2 Model selection — comparison narrative

```bash
python scripts/cv_compare.py            # baseline 5-fold CV across estimators
python scripts/cv_groupkfold.py         # group-aware 5-fold CV (no UPRN leakage)
python scripts/cv_champions.py          # regression-to-band + ensemble + LightGBM
python scripts/model_panel.py           # holdout panel across every variant
python scripts/tune_histgb.py           # Optuna on HistGB classifier (legacy)
```

Outputs: `reports/cv_compare.json`, `cv_groupkfold.json`, `cv_champions.json`,
`model_panel.json`, `tune_histgb.json`. Used to construct the model-selection
narrative in §3.4 of the report.

### 2.3 Champion training, freezing, and inference

```bash
python scripts/sap_stratified.py        # produces sap_stratified.json (unified vs stratified vs hybrid + 3-seed stability) -- ~10-15 min
python scripts/freeze_champion.py       # writes artefacts/champion.pkl + reports/champion_artefact.json
python scripts/predict_oxford.py        # writes reports/predictions_oxford.csv (76,400 rows)
```

`freeze_champion.py` computes the sha256 of the pickle and records it in
`champion_artefact.json`; `predict_oxford.py` validates the hash before
inference, so silent artefact replacement is detected.

### 2.4 Robustness and explainability

```bash
python scripts/champion_robustness.py   # MAE-bands, linear kappa, ONS DSC comparator, per-class bootstrap CIs, confidence-proxy validation, score reliability
python scripts/champion_explanations.py # permutation importance + SHAP for the current champion -- ~10 min
python scripts/diagnostics.py           # legacy diagnostics on the HGB classifier
```

### 2.5 Verify everything ties together

```bash
python scripts/verify_model_pipeline.py
```

70 checks across: required artefacts on disk, baseline numbers reasonable,
leakage guards, group-split integrity, CV stability, SAP fairness gap
addressed, hybrid section measured-vs-docstring, frozen champion
sha256 valid, per-UPRN CSV schema correct, ONS DSC benchmark met,
permutation importance scored on QWK, SHAP generated, etc. Should exit 0.

## 3. Workflow B — notebook walk-through

```bash
jupyter lab notebooks/EPC_Oxford_Analysis.ipynb
```

In the notebook UI: **Kernel → Restart Kernel and Run All Cells**.

The notebook is 44 cells, structured to follow the assignment rubric:

| Section | Cells | Output |
|---|---|---|
| Acquisition (LO1, 7%) | 2-3 | inline narrative |
| Data wrangling (LO3, 20%) | 4-6 | `data/processed/oxford_epc_clean.csv` |
| Descriptive (LO2, 15%) | 7-11 | Figures 1-4 |
| Diagnostic (LO3, 15%) | 12-17 | Figures 5-7, Spearman ρ, Cramér's V |
| Feature engineering (LO3, 5%) | 18 | inline narrative (features built in §3.3) |
| Predictive (LO3, 20%) | 19-37 | Figures 8-11; CV comparison; **production champion subsection at cells 30-37** that loads `artefacts/champion.pkl`, verifies its sha256, and displays current numbers from `champion_robustness.json` |
| Recommendations (LO1, 8%) | 38-42 | `recommendations.csv` analysis, `fig_top_recommendations.png` |
| CORN NN (optional) | 43 | skipped if TensorFlow unavailable |
| Save | 44 | `models/champion.joblib` |

Long-running cells:

- `cell baselines` (5 estimators × 5-fold CV) — ~5 min
- `cell optuna` (30 trials × 3-fold CV) — ~25 min
- `cell permimp` (5,000 sample × 3 repeats) — ~5 min
- `cell shap` (4,000 sample TreeExplainer) — ~2 min
- `cell corn-nn` (TensorFlow) — ~5 min if available

Total wall time on a typical laptop: 40-60 min. The notebook is shipped
**pre-executed with outputs** (verified to run cleanly end-to-end via
`jupyter nbconvert --execute`); `Kernel -> Restart and Run All` reproduces
every figure and metric from scratch.

## 4. Additional reproducible artefacts

```bash
python scripts/champion_significance.py  # bootstrap 95% CI for QWK, R2/RMSE, error-direction breakdown, cohort + postcode counts
python scripts/train_corn_nn.py          # TensorFlow CORN ordinal-NN baseline -> reports/corn_nn_result.json
python scripts/regen_descriptive_figs.py # regenerates the 3.2/3.3 descriptive + diagnostic figures and statistics
python scripts/run_clustering.py          # K-Prototypes segmentation -> clustering.json + figure
python scripts/run_recommender.py         # Apriori + content-kNN recommender -> recommender.json + figure
python scripts/run_property_analysis.py   # property-type vs efficiency + era timeline -> figures
python scripts/run_extra_charts.py       # histogram, violin, donut, treemap, stacked-area, bubble, word cloud
```

The report (`reports/COM6003_Oxford_Report.docx`) and the presentation are
the final written deliverables; an optional interactive Streamlit
dashboard is provided (`streamlit run app.py`). Every numeric claim is reproducible
from the JSON artefacts above and cross-checked by the two verifiers.

## 5. Submission checklist

- [ ] `python scripts/verify_data_pipeline.py` exits 0 (25/25).
- [ ] `python scripts/verify_model_pipeline.py` exits 0 (70/70).
- [ ] `data/processed/oxford_epc_clean.csv` present.
- [ ] `artefacts/champion.pkl` present and sha256 matches `champion_artefact.json`.
- [ ] `reports/predictions_oxford.csv` present (76,400 rows).
- [ ] Notebook runs end-to-end on a fresh kernel.
- [ ] Report ≤ 2,750 body words; Harvard references validated against
      citethemrightonline.com.
- [ ] Presentation opens and renders correctly.
- [ ] All four files submitted on Blackboard:
  - `notebooks/EPC_Oxford_Analysis.ipynb`
  - `data/processed/oxford_epc_clean.csv`
  - `reports/COM6003_Oxford_Report.docx`
  - `presentation/COM6003_Oxford_Presentation.pptx`
- [ ] Same package emailed to **mohammed.ahmed@bucks.ac.uk** before
      **Thursday 18 June 2026, 14:00**.

## Troubleshooting

**`champion.pkl sha256 mismatch`** — somebody re-trained the pickle without
re-running `freeze_champion.py`. Re-run `python scripts/freeze_champion.py`
to refresh both the pickle and the recorded hash in
`reports/champion_artefact.json`.

**Notebook timeouts on the Optuna cell** — drop `n_trials` from 30 to 10 in
the `optuna` cell; results degrade gracefully.

**TensorFlow fails to install** — comment out the `tensorflow` line in
`requirements.txt` and skip the CORN-NN cell. The production champion is
HistGB regression-to-band, not the NN.

**SHAP shape error** — SHAP ≥ 0.43 returns multiclass values as a 3-D
array; older returns a list of 2-D arrays. The current champion is a
single-output regressor so this doesn't apply, but if you re-enable the
notebook's classifier-SHAP cell, see the SHAP version note in
`scripts/champion_explanations.py`.
