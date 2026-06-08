"""GroupKFold CV for the new candidates: regression head, ensemble, and
LightGBM with expected-value decoding. Tells us whether the single-split
ranking from model_panel.py is stable.
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
from sklearn.impute import SimpleImputer

from src.data import (load_certificates, filter_oxford, coerce_numeric,
                      validate_consistency, cap_outliers, drop_fully_missing,
                      group_temporal_split, to_model_matrix)
from src.features import engineer
from src.models import (rf_classifier, hist_gradient_boosting, lgbm_classifier,
                        infer_column_groups, build_pipeline,
                        regression_to_band_estimator, sap_score_to_band)
from src.evaluation import evaluate_classifier
from src import RATING_TO_INT, INT_TO_RATING, RATING_ORDER

OUT = Path("reports/cv_champions.json")


def build_nondedup_frame(raw):
    return (
        raw.pipe(filter_oxford)
           .pipe(coerce_numeric, cols=[
               "CURRENT_ENERGY_EFFICIENCY", "POTENTIAL_ENERGY_EFFICIENCY",
               "TOTAL_FLOOR_AREA", "MULTI_GLAZE_PROPORTION",
               "EXTENSION_COUNT", "NUMBER_HABITABLE_ROOMS", "NUMBER_HEATED_ROOMS",
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


def make_groups(df):
    g = df["UPRN"].astype("string").copy()
    no_key = g.isna()
    g[no_key] = pd.Series([f"_nokey_{i}" for i in range(no_key.sum())],
                          index=df.index[no_key])
    return g


# NOTE: score->band mapping now imported as `sap_score_to_band` from
# src.models to eliminate drift risk (DLUHC thresholds live in exactly one
# place). The previous local copy was removed in the 2026-05 review pass.


def hgb_pipeline(X_train):
    g = infer_column_groups(X_train)
    prep = ColumnTransformer([
        ("num", "passthrough", g["numeric"]),
        ("lo", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                             sparse_output=False), g["low_card"]),
        ("hi", OrdinalEncoder(handle_unknown="use_encoded_value",
                              unknown_value=-1), g["high_card"]),
    ], remainder="drop", verbose_feature_names_out=False)
    return Pipeline([("prep", prep), ("clf", hist_gradient_boosting(max_iter=400))])


def lgbm_pipeline(X_train):
    g = infer_column_groups(X_train)
    num_p = Pipeline([("imp", SimpleImputer(strategy="median"))])
    low_p = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore",
                                            min_frequency=20, sparse_output=False))])
    hi_p = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("ord", OrdinalEncoder(handle_unknown="use_encoded_value",
                                             unknown_value=-1))])
    prep = ColumnTransformer([
        ("num", num_p, g["numeric"]),
        ("lo", low_p, g["low_card"]),
        ("hi", hi_p, g["high_card"]),
    ], remainder="drop", verbose_feature_names_out=False)
    return Pipeline([("prep", prep),
                     ("clf", lgbm_classifier(class_weight="balanced"))])


def reg_pipeline(X_train):
    g = infer_column_groups(X_train)
    prep = ColumnTransformer([
        ("num", "passthrough", g["numeric"]),
        ("lo", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                             sparse_output=False), g["low_card"]),
        ("hi", OrdinalEncoder(handle_unknown="use_encoded_value",
                              unknown_value=-1), g["high_card"]),
    ], remainder="drop", verbose_feature_names_out=False)
    return Pipeline([("prep", prep), ("reg", regression_to_band_estimator(seed=42))])


def main():
    raw = load_certificates("certificates.csv")
    eng = engineer(build_nondedup_frame(raw))
    tr, _ = group_temporal_split(eng)
    X, y, y_reg = to_model_matrix(eng)
    Xtr = X.loc[tr].reset_index(drop=True)
    ytr = y.loc[tr].reset_index(drop=True)
    yreg_tr = y_reg.loc[tr].reset_index(drop=True)
    groups = make_groups(eng.loc[tr]).reset_index(drop=True)
    Xtr_f = fill_cat_nans(Xtr)
    print(f"Train: {Xtr.shape}, unique groups: {groups.nunique()}")

    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(Xtr, ytr, groups=groups))

    def cv_metric(fn, label):
        """fn(tr_idx, va_idx) -> predictions for the val slice."""
        rows = []
        for k, (tr_i, va_i) in enumerate(folds):
            t0 = time.time()
            pred = fn(tr_i, va_i)
            rep = evaluate_classifier(ytr.iloc[va_i], pd.Series(pred, index=ytr.iloc[va_i].index))
            rows.append({"fold": k, "qwk": rep.qwk, "bal_acc": rep.balanced_accuracy,
                         "macro_f1": rep.macro_f1, "acc": rep.accuracy,
                         "fit_seconds": time.time() - t0})
            print(f"  {label} fold {k}: QWK={rep.qwk:.4f} bal={rep.balanced_accuracy:.4f} "
                  f"f1={rep.macro_f1:.4f} ({time.time()-t0:.0f}s)")
        df = pd.DataFrame(rows)
        return rows, {m: {"mean": float(df[m].mean()), "std": float(df[m].std(ddof=1))}
                       for m in ["qwk", "bal_acc", "macro_f1", "acc"]}

    results = {}

    # --- 1. Regression -> band ---
    print("\n=== 1. Regression head -> band mapping ===")
    def reg_fn(tr_i, va_i):
        pipe = reg_pipeline(Xtr_f.iloc[tr_i])
        pipe.fit(Xtr_f.iloc[tr_i], yreg_tr.iloc[tr_i])
        score = pipe.predict(Xtr_f.iloc[va_i])
        return sap_score_to_band(score)
    folds_r, summary_r = cv_metric(reg_fn, "regression")
    results["regression_to_band"] = {"folds": folds_r, "summary": summary_r}

    # --- 2. Ensemble argmax (RF + HGB + LGBM mean proba) ---
    print("\n=== 2. Ensemble (RF + HGB + LGBM mean proba, argmax) ===")
    def ens_fn(tr_i, va_i):
        rf = build_pipeline(rf_classifier(n_estimators=200), Xtr.iloc[tr_i])
        rf.fit(Xtr.iloc[tr_i], ytr.iloc[tr_i])
        hgb = hgb_pipeline(Xtr_f.iloc[tr_i])
        hgb.fit(Xtr_f.iloc[tr_i], ytr.iloc[tr_i])
        lgbm = lgbm_pipeline(Xtr.iloc[tr_i])
        lgbm.fit(Xtr.iloc[tr_i], ytr.iloc[tr_i])
        p_rf = rf.predict_proba(Xtr.iloc[va_i])
        p_hgb = hgb.predict_proba(Xtr_f.iloc[va_i])
        p_lgbm = lgbm.predict_proba(Xtr.iloc[va_i])
        # ensure class order matches
        if not (list(rf.classes_) == list(hgb.classes_) == list(lgbm.classes_)):
            # Reorder
            classes = list(rf.classes_)
            def reorder(pipe, p):
                idx = [list(pipe.classes_).index(c) for c in classes]
                return p[:, idx]
            p_hgb = reorder(hgb, p_hgb)
            p_lgbm = reorder(lgbm, p_lgbm)
        else:
            classes = list(rf.classes_)
        mean_p = (p_rf + p_hgb + p_lgbm) / 3
        return np.array(classes)[mean_p.argmax(axis=1)]
    folds_e, summary_e = cv_metric(ens_fn, "ensemble")
    results["ensemble_argmax"] = {"folds": folds_e, "summary": summary_e}

    # --- 3. LightGBM with expected-value decoding ---
    print("\n=== 3. LightGBM + expected-value decoding ===")
    def lgbm_expv_fn(tr_i, va_i):
        pipe = lgbm_pipeline(Xtr.iloc[tr_i])
        pipe.fit(Xtr.iloc[tr_i], ytr.iloc[tr_i])
        proba = pipe.predict_proba(Xtr.iloc[va_i])
        cls_int = np.array([RATING_TO_INT[c] for c in pipe.classes_])
        expv = proba @ cls_int
        rounded = np.clip(np.round(expv).astype(int), 0, 6)
        return np.array([INT_TO_RATING[i] for i in rounded])
    folds_l, summary_l = cv_metric(lgbm_expv_fn, "lgbm_expv")
    results["lgbm_expected_value"] = {"folds": folds_l, "summary": summary_l}

    # === Summary ===
    print("\n" + "=" * 78)
    print("GROUPKFOLD CV — NEW CHAMPIONS (mean +/- std)")
    print("=" * 78)
    for label, data in results.items():
        print(f"\n{label}")
        for m in ["qwk", "bal_acc", "macro_f1", "acc"]:
            print(f"  {m:10s}: {data['summary'][m]['mean']:.4f} +/- {data['summary'][m]['std']:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
