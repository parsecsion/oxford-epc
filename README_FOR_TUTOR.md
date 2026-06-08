# How to run this project — guide for the marker

**Module:** COM6003 Data Science · **Local authority:** Oxford (E07000178)
**Main artefact:** `notebooks/EPC_Oxford_Analysis.ipynb`

The notebook is **shipped pre-executed** — every figure, table and metric is
already saved in the file, so you can read it end-to-end without running
anything. If you would like to reproduce it yourself, there are two ways.

---

## Option A — Run locally (recommended, most robust)

Requires Python 3.11 (3.10–3.13 also work). From this folder:

```bash
# 1. create an isolated environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 2. install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. open the notebook
jupyter lab notebooks/EPC_Oxford_Analysis.ipynb
```

Then in Jupyter: **Kernel → Restart Kernel and Run All Cells**.

- Expected wall time: **40–60 min** (the Optuna tuning cell is ~25 min; SHAP
  and the optional TensorFlow CORN-NN add a few minutes each).
- If TensorFlow will not install on your machine, the CORN-NN cell is wrapped
  in `try/except` and skips cleanly — the production champion is gradient-boosted
  trees, not the neural network.

### One-command integrity check (optional, fast)

```bash
python scripts/verify_data_pipeline.py     # 25/25 checks — data side
python scripts/verify_model_pipeline.py    # 70/70 checks — model side
```

Both print `RESULT: N/N checks PASSED` and exit 0.

---

## Option B — Run on Google Colab (zero local setup)

The notebook's first cell auto-detects Colab and clones the project for you.

1. Open Colab → **File → Open notebook → GitHub**, paste the repository URL,
   and select `notebooks/EPC_Oxford_Analysis.ipynb`
   (or use the **Open in Colab** badge in `README.md`).
2. **Runtime → Run all.**

The first cell runs `git clone` to pull `src/`, the data, and the artefacts,
then continues normally. Note that the heavy training cells (Optuna, SHAP,
TensorFlow) run **slower on free Colab CPUs** than locally — budget extra time.

---

## What's in this bundle

| Path | Purpose |
|---|---|
| `notebooks/EPC_Oxford_Analysis.ipynb` | The graded walk-through (pre-executed) |
| `src/` | Reusable package the notebook imports (data, features, models, evaluation, plots) |
| `certificates.csv`, `recommendations.csv` | Raw EPC source data |
| `data/processed/oxford_epc_clean.csv` | Cleaned, de-duplicated, GDPR-stripped dataset (deliverable) |
| `artefacts/champion.pkl` | Frozen champion model (sha256 pinned in `reports/champion_artefact.json`) |
| `reports/` | Precomputed JSON metrics, figures, and the written report |
| `scripts/` | Pipeline + the two verifier scripts |
| `docs/` | Data card and model card |
| `presentation/` | The slide deck |

**Headline result:** champion `SapStratifiedRegressor`, temporal hold-out (n = 12,095)
QWK = **0.7696**, 97.7% of SAP-score predictions within 15 points (exceeds the
ONS Data Science Campus 2021 benchmark of 93%).
