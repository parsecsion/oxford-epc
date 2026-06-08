# Data Card — Oxford EPC Cleaned Dataset

**File:** `data/processed/oxford_epc_clean.csv`
**Source:** EPB Register (England & Wales), DLUHC / MHCLG.
**Local authority:** Oxford (ONS code **E07000178**).
**Licence:** Open Government Licence v3.0 (non-address fields). Address fields
*excluded* from this cleaned export to satisfy the Ordnance Survey / Royal Mail
restricted licence and GDPR/DPA 2018 minimisation principles.

## Removed for privacy
`ADDRESS1`, `ADDRESS2`, `ADDRESS3`, `ADDRESS`, `POSTCODE`,
`UPRN`, `BUILDING_REFERENCE_NUMBER`, `LMK_KEY` (kept only as index, not as feature).
A coarse `POSTCODE_DISTRICT` (e.g., `OX4`) is retained as a non-PII geographic feature.

## Removed as administrative or constant within Oxford
`LOCAL_AUTHORITY` (constant = E07000178), `LOCAL_AUTHORITY_LABEL`,
`CONSTITUENCY`, `CONSTITUENCY_LABEL`, `COUNTY`, `POSTTOWN`, `LODGEMENT_DATETIME`
(redundant with `LODGEMENT_DATE`), `UPRN_SOURCE`.

## SAP-engine *output* columns — present in the CSV, removed from the model

Every column in this group is a **derived output** of the same SAP/RdSAP
calculation that produces the target `CURRENT_ENERGY_RATING`. Including any
of them as a feature is total target leakage. The cleaned CSV
(`oxford_epc_clean.csv`) **retains** them because they are needed for the
diagnostic plots in §3.2-3.3 of the report and for downstream analyses by
other users. They are explicitly **stripped from the feature matrix** by
`src.data.to_model_matrix` before the model is trained:

- `POTENTIAL_ENERGY_RATING`, `POTENTIAL_ENERGY_EFFICIENCY`
- `CURRENT_ENERGY_EFFICIENCY` (numeric form of the target — kept as the
  *secondary regression target only*)
- `ENVIRONMENTAL_IMPACT_CURRENT`, `ENVIRONMENTAL_IMPACT_POTENTIAL`
- `ENERGY_CONSUMPTION_CURRENT`, `ENERGY_CONSUMPTION_POTENTIAL`
- `CO2_EMISSIONS_CURRENT`, `CO2_EMISS_CURR_PER_FLOOR_AREA`, `CO2_EMISSIONS_POTENTIAL`
- `LIGHTING_COST_CURRENT`, `LIGHTING_COST_POTENTIAL`
- `HEATING_COST_CURRENT`, `HEATING_COST_POTENTIAL`
- `HOT_WATER_COST_CURRENT`, `HOT_WATER_COST_POTENTIAL`
- per-element `*_ENERGY_EFF` and `*_ENV_EFF` star ratings (e.g.
  `WALLS_ENERGY_EFF`), which are *sub-outputs* of the same engine.

The corresponding **physical** description fields — `WALLS_DESCRIPTION`,
`ROOF_DESCRIPTION`, `MAINHEAT_DESCRIPTION`, `WINDOWS_DESCRIPTION`, etc. — are
**retained as features** because they describe building reality, not engine
output.

## Auto-dropped columns (≥ 99.9% NaN)

`SHEATING_ENERGY_EFF` and `SHEATING_ENV_EFF` are entirely empty in the Oxford
subset (no dwelling has a recorded second-heating system) and are dropped
in `src.data.drop_fully_missing`.

## Deduplication
Multiple EPCs may exist per dwelling. We keep only the **most recent EPC per
`UPRN`** to prevent the same property leaking between train and test.

## Outlier handling
- `TOTAL_FLOOR_AREA` capped at 99.5th percentile.
- Rows with `TOTAL_FLOOR_AREA < 10` m² dropped as data errors.
- Negative cost or consumption values dropped.

## Targets
- **Primary:** `CURRENT_ENERGY_RATING` (ordinal, A–G).
- **Secondary:** `CURRENT_ENERGY_EFFICIENCY` (integer 1–100).

## Splits
- Train: `INSPECTION_DATE <= 2022-12-31`.
- Test (hold-out): `INSPECTION_DATE >= 2023-01-01`.
- 5-fold stratified CV inside the training portion.

## Known biases / limitations
1. **Lodgement-event selection bias.** EPCs only generated at sale, rental,
   new-build, or grant assessment. Long-tenure owner-occupied homes
   under-represented.
2. **Methodology drift.** SAP/RdSAP conventions revised (2009, 2012, SAP 10
   in 2022) — temporal split mitigates but does not eliminate.
3. **Assessor variance.** Field measurements partly inferred from defaults;
   `*_DESCRIPTION` strings often contain "(assumed)".
4. **Geographic generalisation.** Findings calibrated to Oxford's housing
   stock (~57% Victorian/Edwardian, dense terraces and student conversions);
   may not transfer to coastal, rural, or post-war estate stock.
