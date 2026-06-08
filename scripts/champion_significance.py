"""Bootstrap statistical significance of the champion's holdout QWK.

Reads ``reports/predictions_oxford.csv``, resamples the holdout with
replacement (1000 times by default), recomputes QWK on each resample, and
saves:

* The bootstrap 95% CI for the champion's QWK
* R² and RMSE for the underlying SAP-score regression head
* The error-direction breakdown (under-predict vs over-predict)
* Off-by-N analysis (how serious are the model's mistakes?)

Output: ``reports/champion_significance.json``.

This addresses the academic-rigour gap that the headline 0.7741 QWK had
no formal uncertainty interval and the regression head's R²/RMSE were
not reported alongside the band-classification metrics.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, r2_score, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
PRED_CSV = ROOT / "reports" / "predictions_oxford.csv"
OUT_JSON = ROOT / "reports" / "champion_significance.json"
RATING_ORDER = list("ABCDEFG")
RATING_TO_INT = {b: i for i, b in enumerate(RATING_ORDER)}


def main() -> int:
    if not PRED_CSV.exists():
        print(f"ERROR: {PRED_CSV} missing", file=sys.stderr)
        return 2
    df = pd.read_csv(PRED_CSV)
    te = df[df["split"] == "test"].reset_index(drop=True)
    n = len(te)
    print(f"Holdout: {n:,} rows")

    yt = te["actual_band"].map(RATING_TO_INT).to_numpy()
    yp = te["predicted_band"].map(RATING_TO_INT).to_numpy()
    score_true = te["actual_sap_score"].to_numpy(dtype=float)
    score_pred = te["predicted_sap_score"].to_numpy(dtype=float)

    # --- Point estimates ---
    qwk_point = float(cohen_kappa_score(yt, yp,
                                          labels=list(range(7)),
                                          weights="quadratic"))
    r2_point = float(r2_score(score_true, score_pred))
    rmse_point = float(np.sqrt(mean_squared_error(score_true, score_pred)))
    print(f"\nPoint estimates:")
    print(f"  QWK   = {qwk_point:.4f}")
    print(f"  R²    = {r2_point:.4f}  (SAP score regression)")
    print(f"  RMSE  = {rmse_point:.3f}  SAP points")

    # --- Bootstrap 95% CI for QWK ---
    print(f"\nBootstrap (n_resamples=1000, seed=42) ...")
    rng = np.random.default_rng(42)
    n_resamples = 1000
    qwk_boot = np.empty(n_resamples)
    r2_boot = np.empty(n_resamples)
    rmse_boot = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt_b = yt[idx]; yp_b = yp[idx]
        st_b = score_true[idx]; sp_b = score_pred[idx]
        qwk_boot[i] = cohen_kappa_score(yt_b, yp_b,
                                          labels=list(range(7)),
                                          weights="quadratic")
        r2_boot[i] = r2_score(st_b, sp_b)
        rmse_boot[i] = np.sqrt(mean_squared_error(st_b, sp_b))
    qwk_ci = (float(np.quantile(qwk_boot, 0.025)),
              float(np.quantile(qwk_boot, 0.975)))
    r2_ci = (float(np.quantile(r2_boot, 0.025)),
             float(np.quantile(r2_boot, 0.975)))
    rmse_ci = (float(np.quantile(rmse_boot, 0.025)),
               float(np.quantile(rmse_boot, 0.975)))
    print(f"  QWK   95% CI = [{qwk_ci[0]:.4f}, {qwk_ci[1]:.4f}]")
    print(f"  R²    95% CI = [{r2_ci[0]:.4f}, {r2_ci[1]:.4f}]")
    print(f"  RMSE  95% CI = [{rmse_ci[0]:.3f}, {rmse_ci[1]:.3f}]")

    # --- Compare to next-best model (lgbm_expv) ---
    # From reports/model_panel.json: 3c_lgbm_expv holdout QWK = 0.7548
    next_best_qwk = 0.7548
    diff = qwk_point - next_best_qwk
    is_significant = next_best_qwk < qwk_ci[0]
    print(f"\nChampion vs next-best (lgbm_expected_value, model_panel.json):")
    print(f"  champion QWK     = {qwk_point:.4f}")
    print(f"  next-best QWK    = {next_best_qwk:.4f}")
    print(f"  difference       = {diff:+.4f}")
    print(f"  significant?     = "
          f"{'YES' if is_significant else 'no'} "
          f"(next-best falls {'below' if is_significant else 'within'} "
          f"champion's 95% CI lower bound {qwk_ci[0]:.4f})")

    # --- Error-direction breakdown ---
    err = np.abs(yt - yp)
    over = int((yp < yt).sum())     # predicted better (lower idx) than truth
    under = int((yp > yt).sum())    # predicted worse (higher idx) than truth
    correct = int((yp == yt).sum())
    off_by_1 = int((err == 1).sum())
    off_by_2 = int((err == 2).sum())
    off_by_3plus = int((err >= 3).sum())
    print(f"\nError direction and magnitude:")
    print(f"  Correct          : {correct:>6,} ({100*correct/n:.2f}%)")
    print(f"  Off-by-1         : {off_by_1:>6,} ({100*off_by_1/n:.2f}%)")
    print(f"  Off-by-2         : {off_by_2:>6,} ({100*off_by_2/n:.2f}%)")
    print(f"  Off-by-3+        : {off_by_3plus:>6,} ({100*off_by_3plus/n:.2f}%)")
    print(f"  Predicted better : {over:>6,} ({100*over/n:.2f}%)")
    print(f"  Predicted worse  : {under:>6,} ({100*under/n:.2f}%)  "
          f"-- conservative bias, ratio {under/max(over,1):.1f}:1")

    # --- Cohort counts for §4 recommendations ---
    # Re-derive engineered features to count cohorts; load via the same pipeline.
    print(f"\nLoading cleaned frame for cohort counts ...")
    sys.path.insert(0, str(ROOT))
    from src.data import (load_certificates, filter_oxford, coerce_numeric,
                            validate_consistency, cap_outliers,
                            drop_fully_missing)
    from src.features import engineer
    raw = load_certificates(ROOT / "certificates.csv")
    eng = (engineer(raw.pipe(filter_oxford)
                       .pipe(coerce_numeric, cols=[
                           "CURRENT_ENERGY_EFFICIENCY", "TOTAL_FLOOR_AREA",
                           "NUMBER_HABITABLE_ROOMS", "NUMBER_HEATED_ROOMS",
                           "CO2_EMISSIONS_CURRENT",
                           "ENERGY_CONSUMPTION_CURRENT", "REPORT_TYPE",
                       ])
                       .pipe(validate_consistency).pipe(cap_outliers)
                       .pipe(drop_fully_missing, threshold=0.999)
                       .dropna(subset=["CURRENT_ENERGY_RATING"])
                       .loc[lambda d: d["CURRENT_ENERGY_RATING"]
                              .isin(RATING_ORDER)]
                       .reset_index(drop=True)))
    n_eng = len(eng)
    age = eng.get("CONSTRUCTION_AGE_NUM")
    cohorts = {
        "n_certificates_total": int(n_eng),
        "pre_1929_count": int((age < 1930).sum()) if age is not None else None,
        "flag_solid_brick_bare_count":
            int(eng.get("FLAG_SOLID_BRICK_BARE", pd.Series()).sum()),
        "flag_no_loft_insulation_count":
            int(eng.get("FLAG_NO_LOFT_INSULATION", pd.Series()).sum()),
        "flag_single_glazed_count":
            int(eng.get("FLAG_SINGLE_GLAZED", pd.Series()).sum()),
        "single_glazed_1900_1949_count": (
            int(((age >= 1900) & (age < 1950) &
                  eng.get("FLAG_SINGLE_GLAZED",
                            pd.Series(False, index=eng.index)).astype(bool))
                .sum())
            if age is not None else None),
    }
    print("Cohort counts:")
    for k, v in cohorts.items():
        print(f"  {k}: {v}")

    # Postcode-district pre-1900 density vs absolute
    pd_pre1900 = []
    if "POSTCODE_DISTRICT" in eng.columns and age is not None:
        for district, sub in eng.groupby("POSTCODE_DISTRICT"):
            n_dist = len(sub)
            if n_dist < 500:
                continue
            pre1900 = int((sub["CONSTRUCTION_AGE_NUM"] < 1900).sum())
            pd_pre1900.append({
                "district": district,
                "n_certificates": n_dist,
                "pre_1900_count": pre1900,
                "pre_1900_pct": float(100 * pre1900 / n_dist),
            })
        pd_pre1900.sort(key=lambda x: -x["pre_1900_count"])
    print("\nPostcode-district pre-1900 (sorted by absolute count):")
    for r in pd_pre1900:
        print(f"  {r['district']:<4s}  n={r['n_certificates']:>6,}  "
              f"pre-1900 = {r['pre_1900_count']:>5,} "
              f"({r['pre_1900_pct']:>5.1f}%)")

    # --- Write ---
    out = {
        "holdout_n": int(n),
        "point_estimates": {
            "qwk": qwk_point,
            "r2": r2_point,
            "rmse_sap_points": rmse_point,
        },
        "bootstrap_95_ci": {
            "qwk": list(qwk_ci),
            "r2": list(r2_ci),
            "rmse_sap_points": list(rmse_ci),
            "n_resamples": n_resamples,
            "seed": 42,
        },
        "vs_next_best": {
            "next_best_model": "lgbm_expected_value (from model_panel.json)",
            "next_best_qwk": next_best_qwk,
            "champion_qwk": qwk_point,
            "difference": diff,
            "next_best_below_champion_ci_lower_bound": is_significant,
        },
        "error_breakdown": {
            "n": int(n),
            "correct": correct,
            "off_by_1": off_by_1,
            "off_by_2": off_by_2,
            "off_by_3_plus": off_by_3plus,
            "predicted_better_than_actual": over,
            "predicted_worse_than_actual": under,
            "conservative_bias_ratio": float(under / max(over, 1)),
        },
        "cohorts_for_recommendations": cohorts,
        "postcode_districts": pd_pre1900,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
