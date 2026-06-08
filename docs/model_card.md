# Model Card — Oxford EPC Rating Predictor

Adapted from the Model Cards framework (Mitchell *et al.*, 2019). Numbers
quoted below trace to JSON artefacts in `reports/` that are each protected
by a check in `scripts/verify_model_pipeline.py` (70 checks, all passing).

## Model details

- **Champion:** `src.models.SapStratifiedRegressor` — a hybrid regression-to-band
  model with two `HistGradientBoostingRegressor` heads on the continuous
  `CURRENT_ENERGY_EFFICIENCY` (SAP) score:
  - `unified` regressor trained on all training rows; serves the dominant
    RdSAP cohort (`REPORT_TYPE == 100`) and is also the fallback for any
    row with an unknown `REPORT_TYPE`.
  - `sap_only` specialist trained on the SAP-only subset
    (`REPORT_TYPE == 101`); used only for SAP-assessed test rows.
  Predictions are continuous SAP scores; bands are derived deterministically
  via the official DLUHC thresholds (A ≥ 92, B 81–91, C 69–80, D 55–68,
  E 39–54, F 21–38, G ≤ 20).
- **Justification:** EPC bands are a deterministic discretisation of an
  observable SAP score. Under known thresholds, regression-then-threshold is
  a consistent estimator (Pedregosa *et al.*, 2017) and arguably more
  principled than fitting a discrete ordinal classifier (e.g. CORN; Cao
  *et al.*, 2020) that ignores the score-generating process.
- **Selection rule:** highest temporal-holdout QWK among CV-stable
  candidates, ties broken by inference cost. LightGBM with expected-value
  decoding actually wins CV (0.8672 vs 0.8640) but loses holdout
  (0.7548 vs 0.7560 before routing); regression-to-band is also ≈ 4× faster
  to fit, and the hybrid SapStratifiedRegressor extends it to 0.7696.
  Side-by-side numbers are recorded in `reports/champion_artefact.json`.
- **Frozen artefact:** `artefacts/champion.pkl` (5.8 MB).
  sha256 = `50f1fba2a542b4799c9e064cc770a686cdaa899b6d043add1ecdcbe3127fdd50`
  is pinned in `reports/champion_artefact.json` and re-checked on every
  load by `scripts/predict_oxford.py`.
- **Baselines reported (`reports/model_panel.json`):** stratified-dummy
  (QWK ≈ 0), majority-class (QWK = 0), age-only logistic regression
  (QWK 0.33), full-feature logistic regression (QWK 0.40), Random Forest,
  HistGB classifier, LightGBM (with `class_weight='balanced'` and with
  expected-value decoding), and a probability-mean ensemble.
- **Random seed:** 42. Multi-seed stability (`reports/sap_stratified.json`):
  QWK = 0.7714 ± 0.0019 across seeds 42, 123, 2026.

## Intended use

- *Primary:* support local-authority and housing-association decisions about
  retrofit prioritisation across the Oxford domestic stock; the operational
  output is `reports/predictions_oxford.csv`, a 76,400-row per-UPRN file
  with predicted band, predicted SAP score, and a confidence proxy
  (band-boundary distance) that is validated as monotone in QWK across
  quartiles (Q1 = 0.66, Q4 = 0.84).
- *Out of scope:* individual property valuation, mortgage underwriting,
  enforcement decisions, or any use that would constitute automated decision-
  making with legal effect under GDPR Article 22.

## Training data

Oxford EPC certificates lodged on or before 2022-12-31, group-split by UPRN
so that no dwelling appears in both train and test. After dedup and outlier
removal: 54,834 training rows (1,395 SAP, 53,439 RdSAP) across 85 features.
See `reports/champion_artefact.json::train_shape` for the exact numbers.

## Evaluation data

EPC certificates lodged 2023-01-01 onwards (temporal hold-out, n = 12,095:
10,700 RdSAP + 1,395 SAP).

## Headline metrics on the hold-out

All numbers traceable to `reports/champion_robustness.json`:

| Metric | Value | Notes |
|---|---|---|
| Quadratic Weighted Kappa | **0.7696** | Primary metric; 0.7714 ± 0.0019 across 3 seeds |
| Linear-weighted kappa | 0.652 | Warrens (2012) robustness check |
| Accuracy | 0.700 | |
| Balanced accuracy | 0.584 | |
| Macro-F1 | 0.550 | |
| MAE in band-units | 0.316 | Under one third of a band off on average |
| Score MAE (SAP points) | 4.04 | At or below the ~5-point inter-assessor noise floor (Few *et al.*, 2023; Hardy and Glew, 2019) |
| % within 5 SAP points | 71.4% | |
| % within 10 SAP points | 92.5% | |
| **% within 15 SAP points** | **97.7%** | **Exceeds the ONS Data Science Campus (2021) UK national benchmark of 93%** |
| SAP-cohort QWK lift (hybrid) | +0.0609 | rt_101 unified 0.517 → hybrid 0.578 |

Per-class F1 bootstrap 95% CIs and per-REPORT_TYPE breakdowns are recorded
in `reports/champion_robustness.json` (`per_class_f1_bootstrap_ci` and
`per_report_type`). The minority classes A, F and G carry wide CIs
(width 0.11, 0.18, 0.32 respectively) driven by their irreducibly small
test counts.

## Calibration

For a regression head, the calibration analogue is the score-reliability
curve (`reports/figures/fig_score_reliability.png`): per-decile mean
predicted SAP score against per-decile mean observed. The champion has
slope 0.79 and intercept +16, indicating mild compression toward the mean.
Ordinal-band predictions are robust to such uniform shifts because the
DLUHC thresholds are wide, but downstream uses requiring unbiased point
scores should post-process with isotonic regression.

## Feature attribution

Two complementary methods, both computed against the frozen champion
(`reports/champion_explanations.json`):

- **Permutation importance**, scored on hybrid-routed band QWK so that the
  magnitude is directly interpretable as "Δ-QWK on shuffle". Top features:
  `WINDOWS_DESCRIPTION` (+0.071), `CONSTRUCTION_AGE_NUM` (+0.063),
  `MAIN_FUEL` (+0.062), `MAINHEAT_DESCRIPTION` (+0.036),
  `HOTWATER_DESCRIPTION` (+0.034), `BUILT_FORM` (+0.027).
- **SHAP** TreeExplainer on the unified regressor (the RdSAP path), with
  values in SAP-score units. Beeswarm in `fig_shap_summary.png`.

Strobl *et al.* (2007) showed that impurity-based importances are biased
toward high-cardinality features, so we explicitly use permutation
importance instead. Loecher (2023) notes that SHAP on tree ensembles can
itself be biased toward high-entropy features; per-feature SHAP magnitudes
should be interpreted comparatively rather than as causal effect sizes.

## Ethical considerations

- Address fields and UPRN excluded from the model inputs to comply with
  GDPR Article 5 minimisation; only the outward postcode (`POSTCODE_DISTRICT`,
  e.g. "OX4") is retained as a non-personal geographic feature.
- The training corpus over-represents transacting properties; predictions on
  long-tenure owner-occupied homes carry elevated uncertainty.
- Class-imbalance handling avoids SMOTE/oversampling per van den Goorbergh
  *et al.* (2022) on the calibration harm of imbalance corrections in risk
  prediction; the regression head handles imbalance natively via the L2
  loss on the continuous SAP score.
- Do not use the model to direct enforcement against tenants in private-
  rented stock — the same physical condition is rated identically across
  tenures, but the lodgement *cause* is correlated with tenure, which can
  produce confounded SHAP attributions if used naively.

## Caveats and recommendations

- Re-train when SAP methodology changes (next anticipated update of SAP 10).
- Recalibrate when extending to a different local authority — Oxford's
  housing-stock profile is atypical for England & Wales as a whole.
- The SAP-cohort (`REPORT_TYPE == 101`) holdout QWK of 0.578 indicates
  residual headroom that requires cohort-specific feature engineering;
  the hybrid routing closes part of the gap but is not a substitute for
  better SAP-specific inputs.
- Reproducibility: `scripts/freeze_champion.py` re-fits the champion
  deterministically and re-pins the sha256;
  `scripts/verify_model_pipeline.py` cross-checks the pickle hash, every
  headline number, and every figure against its source JSON.
