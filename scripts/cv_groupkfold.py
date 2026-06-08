"""GroupKFold CV — corrects the methodological flaw in cv_compare.py.

The previous cv_compare.py used StratifiedKFold on a training set that
contains multiple certificates per UPRN. Inner folds could therefore
put the same UPRN in both train and validation — the very leakage that
group_temporal_split was designed to prevent. This script fixes that by
using GroupKFold(by UPRN).
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

from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.data import (load_certificates, build_clean_frame, to_model_matrix,
                      filter_oxford, coerce_numeric, validate_consistency,
                      cap_outliers, drop_fully_missing,
                      temporal_split, group_temporal_split)
from src.features import engineer
from src.models import build_pipeline, rf_classifier, hist_gradient_boosting, infer_column_groups
from src.evaluation import evaluate_classifier

RATING_ORDER = list("ABCDEFG")
OUT = Path("reports/cv_groupkfold.json")


def build_nondedup_frame(raw):
    return (
        raw.pipe(filter_oxford)
           .pipe(coerce_numeric, cols=[
               "CURRENT_ENERGY_EFFICIENCY", "TOTAL_FLOOR_AREA",
               "MULTI_GLAZE_PROPORTION", "EXTENSION_COUNT",
               "NUMBER_HABITABLE_ROOMS", "NUMBER_HEATED_ROOMS",
               "LOW_ENERGY_LIGHTING", "NUMBER_OPEN_FIREPLACES",
               "WIND_TURBINE_COUNT", "UNHEATED_CORRIDOR_LENGTH",
               "FLOOR_HEIGHT", "PHOTO_SUPPLY", "FLAT_STOREY_COUNT",
               "FIXED_LIGHTING_OUTLETS_COUNT", "LOW_ENERGY_FIXED_LIGHT_COUNT",
               "REPORT_TYPE", "CO2_EMISSIONS_CURRENT", "ENERGY_CONSUMPTION_CURRENT",
           ])
           .pipe(validate_consistency)
           .pipe(cap_outliers)
           .pipe(drop_fully_missing, threshold=0.999)
           .dropna(subset=["CURRENT_ENERGY_RATING"])
           .loc[lambda d: d["CURRENT_ENERGY_RATING"].isin(RATING_ORDER)]
           .reset_index(drop=True)
    )


def fill_cat_nans(X):
    out = X.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(object).where(out[c].notna(), "__MISSING__")
    return out


def hgb_builder(X_train):
    g = infer_column_groups(X_train)
    prep = ColumnTransformer([
        ("num", "passthrough", g["numeric"]),
        ("lo", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                             sparse_output=False), g["low_card"]),
        ("hi", OrdinalEncoder(handle_unknown="use_encoded_value",
                              unknown_value=-1), g["high_card"]),
    ], remainder="drop", verbose_feature_names_out=False)
    return Pipeline([("prep", prep), ("clf", hist_gradient_boosting(max_iter=400))])


def cv_run(X, y, groups, builder, label, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    results = []
    for k, (tr, va) in enumerate(gkf.split(X, y, groups=groups)):
        t0 = time.time()
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]
        # Verify no leakage within fold
        tr_g = set(groups.iloc[tr].dropna())
        va_g = set(groups.iloc[va].dropna())
        overlap = len(tr_g & va_g)
        pipe = builder(Xtr)
        pipe.fit(Xtr, ytr)
        pred = pd.Series(pipe.predict(Xva), index=Xva.index)
        rep = evaluate_classifier(yva, pred)
        results.append({
            "fold": k, "n_train": len(tr), "n_val": len(va),
            "group_overlap": overlap,
            "acc": rep.accuracy, "bal_acc": rep.balanced_accuracy,
            "macro_f1": rep.macro_f1, "qwk": rep.qwk,
            "per_class_f1": {r: rep.per_class[r]["f1-score"] for r in RATING_ORDER},
            "fit_seconds": time.time() - t0,
        })
        print(f"  {label} fold {k}: QWK={rep.qwk:.4f} bal={rep.balanced_accuracy:.4f} "
              f"f1={rep.macro_f1:.4f} grp_overlap={overlap} ({time.time()-t0:.0f}s)")
    return results


def aggregate(metrics):
    df = pd.DataFrame(metrics)
    return {m: {"mean": float(df[m].mean()), "std": float(df[m].std(ddof=1))}
            for m in ["acc", "bal_acc", "macro_f1", "qwk"]}


def main():
    raw = load_certificates("certificates.csv")
    clean_dedup = build_clean_frame(raw)
    clean_non = build_nondedup_frame(raw)
    eng_dedup = engineer(clean_dedup)
    eng_non = engineer(clean_non)

    # Configurations
    # NOTE: GroupKFold needs a 'groups' array. For un-keyed rows we fabricate
    # a unique pseudo-group per row so they end up in different folds (no
    # opportunity for self-leakage).
    def make_groups(df):
        g = df["UPRN"].astype("string").copy()
        no_key = g.isna()
        # synthesise unique pseudo-IDs for the un-keyed rows
        g[no_key] = pd.Series([f"_nokey_{i}" for i in range(no_key.sum())],
                              index=df.index[no_key])
        return g

    # --- A. RF + dedup (training set) ---
    tr_d, _ = temporal_split(eng_dedup)
    X_d, y_d, _ = to_model_matrix(eng_dedup)
    X_d_tr = X_d.loc[tr_d].reset_index(drop=True)
    y_d_tr = y_d.loc[tr_d].reset_index(drop=True)
    g_d_tr = make_groups(eng_dedup.loc[tr_d]).reset_index(drop=True)

    # --- B. RF + group (training set on non-dedup frame) ---
    tr_g, _ = group_temporal_split(eng_non)
    X_g, y_g, _ = to_model_matrix(eng_non)
    X_g_tr = X_g.loc[tr_g].reset_index(drop=True)
    y_g_tr = y_g.loc[tr_g].reset_index(drop=True)
    g_g_tr = make_groups(eng_non.loc[tr_g]).reset_index(drop=True)

    print(f"\nA shape: {X_d_tr.shape}, B shape: {X_g_tr.shape}")
    print(f"A groups: {g_d_tr.nunique()} unique, B groups: {g_g_tr.nunique()} unique")

    results = {}

    print("\n=== A. RF + dedup ===")
    results["A_rf_dedup"] = cv_run(
        X_d_tr, y_d_tr, g_d_tr,
        lambda X: build_pipeline(rf_classifier(n_estimators=200), X),
        "A_rf_dedup")

    print("\n=== B. RF + group (GroupKFold) ===")
    results["B_rf_group"] = cv_run(
        X_g_tr, y_g_tr, g_g_tr,
        lambda X: build_pipeline(rf_classifier(n_estimators=200), X),
        "B_rf_group")

    print("\n=== D. HistGB + group (GroupKFold) ===")
    X_g_filled = fill_cat_nans(X_g_tr)
    results["D_hgb_group"] = cv_run(
        X_g_filled, y_g_tr, g_g_tr,
        hgb_builder, "D_hgb_group")

    # Aggregate
    summary = {k: aggregate(v) for k, v in results.items()}

    print("\n" + "=" * 70)
    print("GROUPKFOLD CV SUMMARY (mean +/- std across 5 folds)")
    print("=" * 70)
    for label, agg in summary.items():
        print(f"\n{label}")
        for m in ["acc", "bal_acc", "macro_f1", "qwk"]:
            print(f"  {m:10s}: {agg[m]['mean']:.4f} +/- {agg[m]['std']:.4f}")

    # Compare to stratified-CV results
    sk_path = Path("reports/cv_compare.json")
    if sk_path.exists():
        sk = json.load(open(sk_path))["summary"]
        print("\n=== STRATIFIED (old) vs GROUPKFOLD (new) ===")
        for k in ["A_rf_dedup", "B_rf_group", "D_hgb_group"]:
            if k in sk and k in summary:
                d = summary[k]["qwk"]["mean"] - sk[k]["qwk"]["mean"]
                print(f"  {k}: stratified QWK={sk[k]['qwk']['mean']:.4f}, "
                      f"groupkfold QWK={summary[k]['qwk']['mean']:.4f}, delta={d:+.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"per_fold": results, "summary": summary}, f, indent=2)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
