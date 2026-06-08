"""Optuna tuning of HistGradientBoostingClassifier on the group split.

Optimises QWK directly via 3-fold inner CV on the training set. Compares
against the untuned baseline reported in reports/cv_compare.json.
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

import optuna
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import cohen_kappa_score

from src.data import (load_certificates, filter_oxford, coerce_numeric,
                      validate_consistency, cap_outliers, drop_fully_missing,
                      group_temporal_split, to_model_matrix)
from src.features import engineer
from src.models import infer_column_groups
from src.evaluation import evaluate_classifier
from src import RATING_TO_INT

OUT_PATH = Path("reports/tune_histgb.json")
N_TRIALS = 20
TIMEOUT_SECONDS = 1800  # 30 min hard cap


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
               "REPORT_TYPE",
               "CO2_EMISSIONS_CURRENT", "ENERGY_CONSUMPTION_CURRENT",
           ])
           .pipe(validate_consistency)
           .pipe(cap_outliers)
           .pipe(drop_fully_missing, threshold=0.999)
           .dropna(subset=["CURRENT_ENERGY_RATING"])
           .loc[lambda d: d["CURRENT_ENERGY_RATING"].isin(list("ABCDEFG"))]
           .reset_index(drop=True)
    )


def fill_cat_nans(X):
    out = X.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(object).where(out[c].notna(), "__MISSING__")
    return out


def make_pipeline(X_train, params):
    g = infer_column_groups(X_train)
    prep = ColumnTransformer([
        ("num", "passthrough", g["numeric"]),
        ("lo", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                             sparse_output=False), g["low_card"]),
        ("hi", OrdinalEncoder(handle_unknown="use_encoded_value",
                              unknown_value=-1), g["high_card"]),
    ], remainder="drop", verbose_feature_names_out=False)
    clf = HistGradientBoostingClassifier(
        max_iter=params["max_iter"],
        learning_rate=params["learning_rate"],
        max_leaf_nodes=params["max_leaf_nodes"],
        min_samples_leaf=params["min_samples_leaf"],
        l2_regularization=params["l2_regularization"],
        max_features=params["max_features"],
        class_weight="balanced",
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
    )
    return Pipeline([("prep", prep), ("clf", clf)])


def main():
    print("Loading & cleaning ...")
    raw = load_certificates("certificates.csv")
    eng = engineer(build_nondedup_frame(raw))
    tr_idx, te_idx = group_temporal_split(eng)
    X, y, _ = to_model_matrix(eng)
    Xtr = fill_cat_nans(X.loc[tr_idx]).reset_index(drop=True)
    ytr = y.loc[tr_idx].reset_index(drop=True)
    Xte = fill_cat_nans(X.loc[te_idx]).reset_index(drop=True)
    yte = y.loc[te_idx].reset_index(drop=True)
    print(f"Train: {Xtr.shape}, Test: {Xte.shape}")

    # 3-fold inner CV
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    folds = list(skf.split(Xtr, ytr))

    def objective(trial):
        params = {
            "max_iter": trial.suggest_int("max_iter", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 5.0),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
        }
        qwks = []
        for tr_i, va_i in folds:
            pipe = make_pipeline(Xtr.iloc[tr_i], params)
            pipe.fit(Xtr.iloc[tr_i], ytr.iloc[tr_i])
            pred = pipe.predict(Xtr.iloc[va_i])
            yt = ytr.iloc[va_i].map(RATING_TO_INT).to_numpy()
            yp = pd.Series(pred).map(RATING_TO_INT).to_numpy()
            qwks.append(cohen_kappa_score(yt, yp, weights="quadratic",
                                          labels=list(range(7))))
        return float(np.mean(qwks))

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS,
                   gc_after_trial=True, catch=(Exception,),
                   show_progress_bar=False)
    elapsed = time.time() - t0
    print(f"\nTuning done in {elapsed:.0f}s. Best inner-CV QWK: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Refit on full train, evaluate on holdout test
    best_pipe = make_pipeline(Xtr, study.best_params)
    best_pipe.fit(Xtr, ytr)
    pred = pd.Series(best_pipe.predict(Xte), index=Xte.index)
    rep = evaluate_classifier(yte, pred)
    print("\n=== HOLDOUT METRICS (tuned HistGB on group split) ===")
    print(f"  accuracy           : {rep.accuracy:.4f}")
    print(f"  balanced_accuracy  : {rep.balanced_accuracy:.4f}")
    print(f"  macro_f1           : {rep.macro_f1:.4f}")
    print(f"  QWK                : {rep.qwk:.4f}")
    print("  per-class F1:")
    for r in "ABCDEFG":
        print(f"    {r}: {rep.per_class[r]['f1-score']:.3f}  "
              f"(support={int(rep.per_class[r]['support'])})")

    out = {
        "n_trials": len(study.trials),
        "best_inner_cv_qwk": study.best_value,
        "best_params": study.best_params,
        "elapsed_seconds": elapsed,
        "holdout": rep.to_dict(),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
