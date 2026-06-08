"""Simple-model baselines to justify (or challenge) the gradient-boosted
champion's complexity — addresses the audit gap that the panel jumped from a
linear model straight to ensembles.

* A depth-5 single DecisionTree (interpretable, one tree).
* A hand-coded deterministic heuristic: median SAP per (construction-era ×
  wall-type) lookup learned on train, applied to test, thresholded to a band.

If the GBM only narrowly beats these, the complexity is not earned.
Writes reports/simple_baselines.json.
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import (load_certificates, filter_oxford, coerce_numeric,
                      validate_consistency, cap_outliers, drop_fully_missing,
                      group_temporal_split, to_model_matrix)
from src.features import engineer
from src.models import infer_column_groups, sap_score_to_band
from src.clustering import wall_type
from src import RATING_TO_INT


def build_nondedup(raw):
    return (raw.pipe(filter_oxford)
            .pipe(coerce_numeric, cols=["CURRENT_ENERGY_EFFICIENCY", "TOTAL_FLOOR_AREA",
                  "NUMBER_HABITABLE_ROOMS", "NUMBER_HEATED_ROOMS",
                  "CO2_EMISSIONS_CURRENT", "ENERGY_CONSUMPTION_CURRENT", "REPORT_TYPE"])
            .pipe(validate_consistency).pipe(cap_outliers)
            .pipe(drop_fully_missing, threshold=0.999)
            .dropna(subset=["CURRENT_ENERGY_RATING"])
            .loc[lambda d: d["CURRENT_ENERGY_RATING"].isin(list("ABCDEFG"))]
            .reset_index(drop=True))


def fill_cat(X):
    out = X.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(object).where(out[c].notna(), "__MISSING__")
    return out


def qwk(yt, yp):
    return float(cohen_kappa_score(yt.map(RATING_TO_INT), pd.Series(yp, index=yt.index).map(RATING_TO_INT),
                                   labels=list(range(7)), weights="quadratic"))


def main():
    print("Loading ...")
    eng = engineer(build_nondedup(load_certificates("certificates.csv")))
    tr, te = group_temporal_split(eng)
    X, y, _ = to_model_matrix(eng)
    Xtr = fill_cat(X.loc[tr]); Xte = fill_cat(X.loc[te])
    ytr, yte = y.loc[tr], y.loc[te]

    # --- depth-5 decision tree ---
    g = infer_column_groups(Xtr)
    prep = ColumnTransformer([
        ("num", "passthrough", g["numeric"]),
        ("lo", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), g["low_card"]),
        ("hi", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), g["high_card"]),
    ], remainder="drop", verbose_feature_names_out=False)
    dt = Pipeline([("prep", prep), ("dt", DecisionTreeClassifier(max_depth=5, random_state=42))])
    dt.fit(Xtr, ytr)
    q_dt = qwk(yte, dt.predict(Xte))
    print(f"DecisionTree(max_depth=5):       QWK = {q_dt:.4f}")

    # --- deterministic heuristic: median SAP per (era-bin x wall-type) ---
    age = pd.to_numeric(eng["CONSTRUCTION_AGE_NUM"], errors="coerce")
    era = pd.cut(age, [0, 1900, 1950, 1980, 2010, 2100],
                 labels=["<1900", "1900-49", "1950-79", "1980-09", "2010+"])
    wt = wall_type(eng)
    sap = pd.to_numeric(eng["CURRENT_ENERGY_EFFICIENCY"], errors="coerce")
    key = era.astype(str) + "|" + wt.astype(str)
    lut = sap.loc[tr].groupby(key.loc[tr]).median()
    global_med = sap.loc[tr].median()
    pred_sap = key.loc[te].map(lut).fillna(global_med).to_numpy()
    q_heur = qwk(yte, sap_score_to_band(pred_sap))
    print(f"Heuristic (era x wall lookup):   QWK = {q_heur:.4f}")

    champ = 0.7696
    out = {
        "decision_tree_depth5_qwk": round(q_dt, 4),
        "heuristic_era_wall_qwk": round(q_heur, 4),
        "champion_qwk": champ,
        "champion_gain_over_tree": round(champ - q_dt, 4),
        "champion_gain_over_heuristic": round(champ - q_heur, 4),
        "verdict": ("GBM complexity justified" if champ - q_dt > 0.03
                    else "GBM gain over a single tree is marginal"),
    }
    (ROOT / "reports" / "simple_baselines.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nChampion 0.7696 vs tree {q_dt:.4f} (+{champ-q_dt:.4f}) vs heuristic {q_heur:.4f} (+{champ-q_heur:.4f})")
    print("verdict:", out["verdict"])


if __name__ == "__main__":
    main()
