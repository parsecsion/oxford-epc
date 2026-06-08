# Data Pipeline Specification — Oxford EPC (E07000178)

This document is the **definition of done** for the data side of the
COM6003 assignment. It records every cleaning, engineering, splitting, and
encoding decision, the rationale, and the completion criteria that determine
when this work is finished.

It is intentionally *not* a model-tuning document; everything below is data
work in the §1, §2, and §3.1 sense of the assignment brief.

---

## 1. Source

| Field | Value |
|---|---|
| Dataset name | Domestic EPC certificates |
| Source authority | MHCLG / DLUHC Energy Performance of Buildings Register |
| Local authority code | `E07000178` (City of Oxford) |
| File on disk | `certificates.csv` (project root) |
| Auxiliary file | `recommendations.csv` (improvement measures) |
| Raw shape | 76,430 rows × 93 columns |
| Date range | 2007-08-09 → 2026-02-28 |

## 2. Canonical pipeline

```
certificates.csv
    │  load_certificates (dtype hints, parse dates)
    ▼
filter_oxford          (defensive)
coerce_numeric         (32 columns; "NO DATA!" → NaN)
validate_consistency   (repair illegal values; flag PV via HAS_ONSITE_GENERATION)
deduplicate_latest     (quality-aware, one row per UPRN)
cap_outliers           (winsorise floor area at 99.5%, drop <10 m²)
drop_fully_missing     (>99.9% NaN columns)
dropna(target)
filter A-G valid bands
    │
    ▼
publishable_clean_frame   (drop PII + admin)
    │
    ▼
data/processed/oxford_epc_clean.csv     (DELIVERABLE, 49,442 × 77)
    │
    ▼
engineer                  (12 derived features)
    │
    ▼
to_model_matrix           (drop leakage, add missingness indicators)
    │
    ▼
X (49,442 × 83), y_class, y_reg
```

## 3. Cleaning decisions and rationale

| Step | Rule | Rows affected | Rationale |
|---|---|---|---|
| validate_consistency | `heated > habitable` → clip | 2 | physically impossible |
| validate_consistency | `heated_rooms == 0` → NaN | 156 | impossible for heated dwelling |
| validate_consistency | `floor_area == 0` → NaN | 75 | impossible value |
| validate_consistency | negative CO2 or energy → flag `HAS_ONSITE_GENERATION` | 185 | preserves PV signal lost when leakage cols dropped |
| validate_consistency | HMO outliers (>30 rooms) | 6 | kept; tracked only |
| deduplicate_latest | one row per UPRN, prefer recency then completeness | 26,969 dropped | unit of analysis is the dwelling; 14.5% UPRNs would otherwise leak across temporal split |
| cap_outliers | floor area ≥ 99.5th pct → 306 m² | 247 | trims commercial mis-classifications |
| cap_outliers | floor area < 10 m² → drop | 41 | smaller than any plausible studio |
| drop_fully_missing | columns ≥ 99.9% NaN | 2 cols dropped | `SHEATING_*_EFF` 100% NaN in Oxford |
| target filter | drop null + non A-G | 0 rows in this dataset | guarantees valid ordinal target |

**Audit trail:** `reports/data_quality_report.csv` is regenerated on every
pipeline run.

## 4. Feature engineering

| Feature | Derived from | Type |
|---|---|---|
| `CONSTRUCTION_AGE_NUM` | `CONSTRUCTION_AGE_BAND` (regex + band-midpoint map) | int year |
| `INSPECTION_YEAR`, `INSPECTION_QUARTER` | `INSPECTION_DATE` | int |
| `POSTCODE_DISTRICT` | `POSTCODE` outward code only | string (non-PII) |
| `LOFT_INSULATION_MM` | regex on `ROOF_DESCRIPTION` | int mm |
| `FLAG_CAVITY_INSULATED` | regex on `WALLS_DESCRIPTION` | binary |
| `FLAG_SOLID_BRICK_BARE` | regex on `WALLS_DESCRIPTION` | binary |
| `FLAG_HEAT_PUMP` | regex on `MAINHEAT_DESCRIPTION` | binary |
| `FLAG_TRIPLE_GLAZED`, `FLAG_SINGLE_GLAZED` | regex on `WINDOWS_DESCRIPTION` | binary |
| `FLAG_LOFT_270MM`, `FLAG_NO_LOFT_INSULATION` | from `LOFT_INSULATION_MM` | binary |
| `FLAG_GAS_BOILER`, `FLAG_ELECTRIC_HEATING` | regex on `MAINHEAT_DESCRIPTION` | binary |
| `LOW_ENERGY_LIGHTING_RATIO` | `LOW_ENERGY_FIXED_LIGHT_COUNT / FIXED_LIGHTING_OUTLETS_COUNT` | float ∈[0,1] |
| `HEATED_ROOM_RATIO` | `NUMBER_HEATED_ROOMS / NUMBER_HABITABLE_ROOMS` | float ∈[0,1] |
| `AREA_PER_ROOM` | `TOTAL_FLOOR_AREA / NUMBER_HABITABLE_ROOMS` | float |
| `LOG_FLOOR_AREA` | `log1p(TOTAL_FLOOR_AREA)` | float |
| `HAS_ONSITE_GENERATION` | flag added during `validate_consistency` | binary |
| `<col>_IS_MISSING` indicators | added by `to_model_matrix(add_missingness=True)` | binary |

**Empirical validation:** permutation importance on the holdout test
(`reports/permutation_importance.csv`) ranks `CONSTRUCTION_AGE_NUM`,
`FLAG_NO_LOFT_INSULATION`, and `FLAG_CAVITY_INSULATED` in the top 6 features,
confirming the engineered features carry real signal.

## 5. Train/test split

Two splitters are provided. Use the one that matches your modelling intent.

### 5.1 `temporal_split` (default)
- Input: deduplicated frame (49,442 rows)
- Output: train 37,347 / test 12,095
- Use when: training a single-cert-per-dwelling model

### 5.2 `group_temporal_split` (advanced)
- Input: **non-deduplicated** frame (76,400 rows, available via
  `scripts/cv_compare.py:build_nondedup_frame`)
- Output: train 54,834 / test 12,095
- Use when: you want longitudinal training observations (15,011 dwellings
  contribute multiple certificates showing pre-/post-improvement state)
- Guarantee: zero UPRN overlap between train and test (verified)
- Empirical: +0.017 QWK (RF) and +0.011 QWK (HistGB) in 5-fold CV vs flat
  split on the dedup'd frame

Both splitters use cutoff 2022-12-31.

## 6. Leakage and PII exclusion

Three sets of columns are removed for principled reasons:

1. **`LEAKAGE_COLS` (16)** — SAP-engine outputs whose values are computed
   *from* the target. Removed in `to_model_matrix`. Includes
   `CURRENT_ENERGY_EFFICIENCY`, all `POTENTIAL_*`, `ENVIRONMENTAL_IMPACT_*`,
   `ENERGY_CONSUMPTION_*`, `CO2_EMISSIONS_*`, cost columns.
2. **`ELEMENT_EFF_COLS` (18)** — Element-level 1–5 star ratings produced by
   SAP. Kept in the published clean CSV for diagnostic plots, removed from
   the model feature matrix.
3. **`PII_COLS` (7)** — `ADDRESS*`, `POSTCODE`, `UPRN`,
   `BUILDING_REFERENCE_NUMBER`. Removed in `publishable_clean_frame` for
   GDPR Article 5 (data minimisation).

## 7. Sensitivity-tested defaults

The defaults below were chosen by ablation (`reports/ablation_study.json`)
and 5-fold CV (`reports/cv_compare.json`), not by assumption.

| Parameter | Default | Why |
|---|---|---|
| `drop_raw_descriptions` | **False** | costs −0.058 QWK; descriptions are feature #2 by permutation importance |
| `drop_methodology` | **False** | costs −0.005 QWK; small empirical penalty for cleaner design |
| `add_missingness` | **True** | +0.014 macro F1; especially helps minority bands A/F/G with RF |
| `missingness_threshold` | 0.05 | adds indicators only for materially missing columns |
| `quality_aware` (dedup) | **True** | picks a different cert for 545 dwellings where latest is sparser |
| `cap_outliers` upper | 99.5th percentile | trims long right tail of likely commercial mis-classifications |
| `cap_outliers` lower | 10 m² | smaller than any plausible studio bedsit |
| `drop_fully_missing` threshold | 99.9% | catches Oxford-specific empties like `SHEATING_*_EFF` |

## 8. Definition of done

This pipeline is **complete** when every check in
`reports/data_completeness_checks.json` passes. The current 25-check
list covers:

- **Acquisition (2 checks)**: raw loadable, LA filter applied
- **Auditability (2)**: quality report exists and records rules
- **Consistency (4)**: no impossible values survive; PV signal preserved
- **Deduplication (2)**: UPRN unique, LMK_KEY unique
- **GDPR (2)**: no PII or admin columns in the published CSV
- **Target (2)**: only A–G values, no nulls
- **Leakage (2)**: no leakage / element-eff columns in the model matrix
- **Splits (3)**: temporal cleanliness for both splitters, zero UPRN leakage
- **Feature engineering (3)**: flags present, construction-age engineered,
  non-PII postcode district
- **Reproducibility (2)**: identical shape and content on re-run
- **Documentation (1)**: this file exists

When all checks pass and no new data-source updates are received, the data
work is finished for this iteration.

## 9. What is explicitly out of scope (handled elsewhere)

| Concern | Lives in | Why not here |
|---|---|---|
| Hyperparameter tuning | `scripts/tune_histgb.py`, Optuna outputs | model work, not data work |
| Model selection | `scripts/cv_compare.py`, `reports/champion_decision.json` | model work |
| Per-class CI reporting | `scripts/diagnostics.py`, `reports/diagnostics.json` | reporting/statistics |
| Calibration | not yet implemented | model post-processing |
| Methodology-stratified modelling | recommended next step | model architecture |
| Longitudinal retrofit-impact analysis | recommended next step | analytical exercise (§3.3) |

## 10. Reproducibility

Run `python scripts/verify_data_pipeline.py` to regenerate every artefact
and verify the checklist. Exit code is 0 if all checks pass, 1 otherwise.

Run `python scripts/cv_compare.py` for 5-fold CV across pipeline configs.
Run `python scripts/diagnostics.py` for feature importance, REPORT_TYPE
stratification, and bootstrap CIs.
