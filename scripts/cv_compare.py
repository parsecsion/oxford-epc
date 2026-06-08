"""5-fold CV comparison of pipeline configurations.

Runs four configurations (RF/HistGB × dedup/group split) plus a
missingness-indicator ablation for HistGB. Reports mean ± std for each
metric across folds, plus bootstrap CIs on per-class F1 of the champion.
"""
from __future__ import annotations
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.data import (
    load_certificates, build_clean_frame, to_model_matrix,
    filter_oxford, coerce_numeric, validate_consistency,
    cap_outliers, drop_fully_missing,
    temporal_split, group_temporal_split,
)
from src.features import engineer
from src.models import build_pipeline, rf_classifier, hist_gradient_boosting
from src.evaluation import evaluate_classifier

REPORT_PATH = Path("reports/cv_compare.json")
RATING_ORDER = list("ABCDEFG")


def build_nondedup_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Build the cleaned-but-not-deduplicated frame for group split."""
    return (
        raw.pipe(filter_oxford)
           .pipe(coerce_numeric, cols=[
               "CURRENT_ENERGY_EFFICIENCY", "POTENTIAL_ENERGY_EFFICIENCY",
               "ENVIRONMENTAL_IMPACT_CURRENT", "ENVIRONMENTAL_IMPACT_POTENTIAL",
               "ENERGY_CONSUMPTION_CURRENT", "ENERGY_CONSUMPTION_POTENTIAL",
               "CO2_EMISSIONS_CURRENT", "CO2_EMISS_CURR_PER_FLOOR_AREA",
               "CO2_EMISSIONS_POTENTIAL",
               "TOTAL_FLOOR_AREA", "MULTI_GLAZE_PROPORTION",
               "EXTENSION_COUNT", "NUMBER_HABITABLE_ROOMS", "NUMBER_HEATED_ROOMS",
               "LOW_ENERGY_LIGHTING", "NUMBER_OPEN_FIREPLACES",
               "WIND_TURBINE_COUNT", "UNHEATED_CORRIDOR_LENGTH",
               "FLOOR_HEIGHT", "PHOTO_SUPPLY", "FLAT_STOREY_COUNT",
               "FIXED_LIGHTING_OUTLETS_COUNT", "LOW_ENERGY_FIXED_LIGHT_COUNT",
               "REPORT_TYPE",
               "LIGHTING_COST_CURRENT", "LIGHTING_COST_POTENTIAL",
               "HEATING_COST_CURRENT", "HEATING_COST_POTENTIAL",
               "HOT_WATER_COST_CURRENT", "HOT_WATER_COST_POTENTIAL",
           ])
           .pipe(validate_consistency)
           .pipe(cap_outliers)
           .pipe(drop_fully_missing, threshold=0.999)
           .dropna(subset=["CURRENT_ENERGY_RATING"])
           .loc[lambda d: d["CURRENT_ENERGY_RATING"].isin(RATING_ORDER)]
           .reset_index(drop=True)
    )


def build_histgb_pipeline(X_train: pd.DataFrame, model):
    """HistGB needs no imputation; OneHot/Ordinal encode categoricals only."""
    from src.models import infer_column_groups
    groups = infer_column_groups(X_train)
    prep = ColumnTransformer([
        ("num", "passthrough", groups["numeric"]),
        ("lo", Pipeline([("oh", OneHotEncoder(handle_unknown="ignore",
                                               min_frequency=20,
                                               sparse_output=False))]),
         groups["low_card"]),
        ("hi", Pipeline([("ord", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                 unknown_value=-1))]),
         groups["high_card"]),
    ], remainder="drop", verbose_feature_names_out=False)
    return Pipeline([("prep", prep), ("clf", model)])


def fill_categorical_nans(X: pd.DataFrame) -> pd.DataFrame:
    """For HistGB path: OneHot/OrdinalEncoder cannot accept NaN in string cols."""
    out = X.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(object).where(out[c].notna(), "__MISSING__")
    return out


def cv_fold_metrics(X: pd.DataFrame, y: pd.Series,
                    fold_idx_pairs, model_builder, fold_label: str):
    """Run a model_builder across folds, return list-of-dicts of metrics."""
    out = []
    for k, (tr_i, va_i) in enumerate(fold_idx_pairs):
        t0 = time.time()
        Xtr = X.iloc[tr_i].copy()
        Xva = X.iloc[va_i].copy()
        ytr = y.iloc[tr_i]
        yva = y.iloc[va_i]
        pipe = model_builder(Xtr)
        pipe.fit(Xtr, ytr)
        pred = pd.Series(pipe.predict(Xva), index=Xva.index)
        rep = evaluate_classifier(yva, pred)
        out.append({
            "fold": k,
            "n_train": len(tr_i), "n_val": len(va_i),
            "acc": rep.accuracy, "bal_acc": rep.balanced_accuracy,
            "macro_f1": rep.macro_f1, "qwk": rep.qwk,
            "per_class_f1": {r: rep.per_class[r]["f1-score"] for r in RATING_ORDER},
            "fit_seconds": time.time() - t0,
        })
        print(f"  {fold_label} fold {k}: QWK={rep.qwk:.4f}  bal={rep.balanced_accuracy:.4f}  "
              f"f1={rep.macro_f1:.4f}  ({time.time()-t0:.0f}s)")
    return out


def aggregate(metric_dicts):
    df = pd.DataFrame(metric_dicts)
    out = {}
    for m in ["acc", "bal_acc", "macro_f1", "qwk"]:
        out[m] = {"mean": float(df[m].mean()), "std": float(df[m].std(ddof=1))}
    pcf = pd.DataFrame([d["per_class_f1"] for d in metric_dicts])
    out["per_class_f1_mean"] = {r: float(pcf[r].mean()) for r in RATING_ORDER}
    out["per_class_f1_std"] = {r: float(pcf[r].std(ddof=1)) for r in RATING_ORDER}
    return out


def main():
    print("Loading & cleaning ...")
    raw = load_certificates("certificates.csv")
    clean_dedup = build_clean_frame(raw)
    clean_non = build_nondedup_frame(raw)
    eng_dedup = engineer(clean_dedup)
    eng_non = engineer(clean_non)

    # === Build training matrices for each (frame, split) pair ===
    # Dedup + flat split: training fold is rows where INSPECTION_DATE <= cutoff
    tr_idx_dedup, _ = temporal_split(eng_dedup)
    tr_idx_group, _ = group_temporal_split(eng_non)

    # 5-fold StratifiedKFold on the training data only
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- model matrices ---
    X_dedup, y_dedup, _ = to_model_matrix(eng_dedup)
    Xtr_dedup = X_dedup.loc[tr_idx_dedup].reset_index(drop=True)
    ytr_dedup = y_dedup.loc[tr_idx_dedup].reset_index(drop=True)
    folds_dedup = list(skf.split(Xtr_dedup, ytr_dedup))

    X_group, y_group, _ = to_model_matrix(eng_non)
    Xtr_group = X_group.loc[tr_idx_group].reset_index(drop=True)
    ytr_group = y_group.loc[tr_idx_group].reset_index(drop=True)
    folds_group = list(skf.split(Xtr_group, ytr_group))

    # Also a non-missingness matrix for HistGB ablation
    X_group_nomiss, _, _ = to_model_matrix(eng_non, add_missingness=False)
    Xtr_group_nomiss = X_group_nomiss.loc[tr_idx_group].reset_index(drop=True)

    print(f"\nTrain shapes: dedup={Xtr_dedup.shape}  group={Xtr_group.shape}  "
          f"group_nomiss={Xtr_group_nomiss.shape}")

    # === Run all 5 configs across folds ===
    results = {}

    def rf_builder(X):
        return build_pipeline(rf_classifier(n_estimators=200), X)

    def hgb_builder(X):
        # HistGB path: fill categorical NaN, no imputer
        return build_histgb_pipeline(fill_categorical_nans(X),
                                     hist_gradient_boosting(max_iter=400))

    def hgb_call(X_train_outer, fold_idx_pairs, label):
        # Wrap to fill NaN within each fold's train slice; but easier to fill once
        Xf = fill_categorical_nans(X_train_outer)
        builder = lambda x: build_histgb_pipeline(x, hist_gradient_boosting(max_iter=400))
        return cv_fold_metrics(Xf, ytr_group if "group" in label else ytr_dedup,
                               fold_idx_pairs, builder, label)

    print("\n=== A. RF + dedup ===")
    results["A_rf_dedup"] = cv_fold_metrics(Xtr_dedup, ytr_dedup, folds_dedup,
                                            rf_builder, "A_rf_dedup")
    print("\n=== B. RF + group ===")
    results["B_rf_group"] = cv_fold_metrics(Xtr_group, ytr_group, folds_group,
                                            rf_builder, "B_rf_group")
    print("\n=== C. HistGB + dedup ===")
    Xf = fill_categorical_nans(Xtr_dedup)
    results["C_hgb_dedup"] = cv_fold_metrics(
        Xf, ytr_dedup, folds_dedup,
        lambda x: build_histgb_pipeline(x, hist_gradient_boosting(max_iter=400)),
        "C_hgb_dedup")
    print("\n=== D. HistGB + group (with missingness) ===")
    Xf = fill_categorical_nans(Xtr_group)
    results["D_hgb_group"] = cv_fold_metrics(
        Xf, ytr_group, folds_group,
        lambda x: build_histgb_pipeline(x, hist_gradient_boosting(max_iter=400)),
        "D_hgb_group")
    print("\n=== E. HistGB + group (NO missingness indicators) ===")
    Xf = fill_categorical_nans(Xtr_group_nomiss)
    # Note: must regenerate folds because Xtr_group_nomiss has different shape if rows changed; same row count here
    folds_group_nomiss = list(skf.split(Xtr_group_nomiss, ytr_group))
    results["E_hgb_group_nomiss"] = cv_fold_metrics(
        Xf, ytr_group, folds_group_nomiss,
        lambda x: build_histgb_pipeline(x, hist_gradient_boosting(max_iter=400)),
        "E_hgb_group_nomiss")

    # === Aggregate ===
    print("\n" + "=" * 78)
    print("CV SUMMARY (mean +/- std across 5 folds)")
    print("=" * 78)
    summary = {}
    for label, metrics in results.items():
        agg = aggregate(metrics)
        summary[label] = agg
        print(f"\n{label}")
        for m in ["acc", "bal_acc", "macro_f1", "qwk"]:
            print(f"  {m:10s}: {agg[m]['mean']:.4f} +/- {agg[m]['std']:.4f}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump({"per_fold": results, "summary": summary}, f, indent=2)
    print(f"\nSaved -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
