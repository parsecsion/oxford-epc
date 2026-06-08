"""Diagnostic analyses on the champion HistGB + group-split pipeline.

- Permutation importance: which features actually drive predictions?
- REPORT_TYPE stratified eval: are gains uniform across SAP vs RdSAP?
- Bootstrap CIs on per-class F1 to gauge minority-class uncertainty.
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.inspection import permutation_importance
from sklearn.metrics import cohen_kappa_score, f1_score

from src.data import (load_certificates, filter_oxford, coerce_numeric,
                      validate_consistency, cap_outliers, drop_fully_missing,
                      group_temporal_split, to_model_matrix)
from src.features import engineer
from src.models import hist_gradient_boosting, infer_column_groups
from src.evaluation import evaluate_classifier
from src import RATING_TO_INT

OUT_PATH = Path("reports/diagnostics.json")
TUNED_PARAMS_PATH = Path("reports/tune_histgb.json")
N_PERM_SAMPLES = 3000
N_BOOTSTRAP = 500


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
    from sklearn.ensemble import HistGradientBoostingClassifier
    prep = ColumnTransformer([
        ("num", "passthrough", g["numeric"]),
        ("lo", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                             sparse_output=False), g["low_card"]),
        ("hi", OrdinalEncoder(handle_unknown="use_encoded_value",
                              unknown_value=-1), g["high_card"]),
    ], remainder="drop", verbose_feature_names_out=False)
    clf = HistGradientBoostingClassifier(
        class_weight="balanced", random_state=42,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
        **{k: v for k, v in params.items()}
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

    # Use tuned params if available, else defaults
    params = {
        "max_iter": 400, "learning_rate": 0.07, "max_leaf_nodes": 63,
        "min_samples_leaf": 20, "l2_regularization": 1.0, "max_features": 1.0,
    }
    if TUNED_PARAMS_PATH.exists():
        with open(TUNED_PARAMS_PATH) as f:
            t = json.load(f)
        if "best_params" in t:
            params.update(t["best_params"])
            print(f"Using tuned params from {TUNED_PARAMS_PATH}")
    print(f"Params: {params}")

    pipe = make_pipeline(Xtr, params)
    pipe.fit(Xtr, ytr)
    pred = pd.Series(pipe.predict(Xte), index=Xte.index)
    rep = evaluate_classifier(yte, pred)
    print(f"\nHoldout metrics: QWK={rep.qwk:.4f}  bal={rep.balanced_accuracy:.4f}  "
          f"f1={rep.macro_f1:.4f}  acc={rep.accuracy:.4f}")

    out = {"holdout_overall": rep.to_dict()}

    # === Permutation importance ===
    print("\n=== Permutation importance (sample n=3000, repeats=5) ===")
    rng = np.random.RandomState(42)
    samp = rng.choice(len(Xte), size=min(N_PERM_SAMPLES, len(Xte)), replace=False)
    Xs, ys = Xte.iloc[samp], yte.iloc[samp]
    yi = ys.map(RATING_TO_INT).to_numpy()

    def qwk_score(estimator, Xv, _):
        yp = pd.Series(estimator.predict(Xv)).map(RATING_TO_INT).to_numpy()
        return cohen_kappa_score(yi, yp, weights="quadratic", labels=list(range(7)))

    pi = permutation_importance(pipe, Xs, ys, n_repeats=5, random_state=42,
                                 scoring=qwk_score, n_jobs=1)
    imp_df = pd.DataFrame({
        "feature": Xs.columns,
        "importance_mean": pi.importances_mean,
        "importance_std": pi.importances_std,
    }).sort_values("importance_mean", ascending=False)
    print("Top 20 features (by mean QWK drop when shuffled):")
    print(imp_df.head(20).to_string(index=False))
    out["permutation_importance_top20"] = imp_df.head(20).to_dict(orient="records")
    imp_df.to_csv("reports/permutation_importance.csv", index=False)

    # === REPORT_TYPE stratified evaluation ===
    print("\n=== Stratified by REPORT_TYPE ===")
    if "REPORT_TYPE" in eng.columns:
        # Get REPORT_TYPE aligned to te_idx
        rt = eng.loc[te_idx, "REPORT_TYPE"].reset_index(drop=True)
        strat = {}
        for rt_val in sorted(rt.dropna().unique()):
            mask = (rt == rt_val)
            if mask.sum() < 100:
                continue
            r = evaluate_classifier(yte[mask], pred[mask])
            strat[str(int(rt_val))] = {
                "n": int(mask.sum()),
                "qwk": r.qwk, "bal_acc": r.balanced_accuracy,
                "macro_f1": r.macro_f1, "acc": r.accuracy,
            }
            print(f"  REPORT_TYPE={int(rt_val)} (n={mask.sum()}): "
                  f"QWK={r.qwk:.4f}  bal={r.balanced_accuracy:.4f}  f1={r.macro_f1:.4f}")
        out["report_type_stratified"] = strat

    # === Bootstrap CIs on per-class F1 ===
    print(f"\n=== Bootstrap CIs on per-class F1 (n={N_BOOTSTRAP}) ===")
    yi_te = yte.map(RATING_TO_INT).to_numpy()
    yp_te = pred.map(RATING_TO_INT).to_numpy()
    n = len(yi_te)
    rng = np.random.RandomState(42)
    bootstraps = {r: [] for r in "ABCDEFG"}
    for _ in range(N_BOOTSTRAP):
        idx = rng.randint(0, n, size=n)
        f1s = f1_score(yi_te[idx], yp_te[idx], labels=list(range(7)),
                       average=None, zero_division=0)
        for r, f in zip("ABCDEFG", f1s):
            bootstraps[r].append(float(f))
    cis = {}
    for r in "ABCDEFG":
        arr = np.array(bootstraps[r])
        cis[r] = {
            "mean": float(arr.mean()),
            "ci_low_2.5": float(np.percentile(arr, 2.5)),
            "ci_high_97.5": float(np.percentile(arr, 97.5)),
            "ci_width": float(np.percentile(arr, 97.5) - np.percentile(arr, 2.5)),
        }
        print(f"  {r}: F1 = {cis[r]['mean']:.3f}  "
              f"[{cis[r]['ci_low_2.5']:.3f}, {cis[r]['ci_high_97.5']:.3f}]  "
              f"width={cis[r]['ci_width']:.3f}")
    out["per_class_f1_bootstrap_ci"] = cis

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
