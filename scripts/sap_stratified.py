"""SAP/RdSAP routing experiments for the regression-to-band champion.

Three configurations are evaluated on the temporal holdout:

* ``1_unified``    — one HistGradientBoostingRegressor trained on the full
                     training set.
* ``2_stratified`` — one regressor per ``REPORT_TYPE`` (no cross-cohort
                     transfer). Documents the strict-stratification baseline.
* ``3_hybrid``     — the actual production champion: a unified regressor for
                     RdSAP rows (REPORT_TYPE 100, the dominant cohort) plus
                     a SAP-only specialist (REPORT_TYPE 101) layered on top.
                     This is the configuration ``SapStratifiedRegressor``
                     implements; its numbers feed the class docstring.

Goal: close the 0.21 QWK fairness gap on SAP-assessed dwellings without
materially regressing overall QWK.
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

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.data import (load_certificates, filter_oxford, coerce_numeric,
                      validate_consistency, cap_outliers, drop_fully_missing,
                      group_temporal_split, to_model_matrix)
from src.features import engineer
from src.models import (infer_column_groups, regression_to_band_estimator,
                        sap_score_to_band, SapStratifiedRegressor)
from src.evaluation import evaluate_classifier

OUT = Path("reports/sap_stratified.json")
RATING_ORDER = list("ABCDEFG")


def build_nondedup(raw):
    return (raw.pipe(filter_oxford)
                .pipe(coerce_numeric, cols=[
                    "CURRENT_ENERGY_EFFICIENCY", "TOTAL_FLOOR_AREA",
                    "NUMBER_HABITABLE_ROOMS", "NUMBER_HEATED_ROOMS",
                    "CO2_EMISSIONS_CURRENT", "ENERGY_CONSUMPTION_CURRENT",
                    "REPORT_TYPE",
                ])
                .pipe(validate_consistency)
                .pipe(cap_outliers)
                .pipe(drop_fully_missing, threshold=0.999)
                .dropna(subset=["CURRENT_ENERGY_RATING"])
                .loc[lambda d: d["CURRENT_ENERGY_RATING"].isin(RATING_ORDER)]
                .reset_index(drop=True))


def fill_cat(X):
    out = X.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(object).where(out[c].notna(), "__MISSING__")
    return out


def make_pipeline(X_train, seed=42):
    """Preprocessor + regression_to_band_estimator (single source of truth)."""
    g = infer_column_groups(X_train)
    prep = ColumnTransformer([
        ("num", "passthrough", g["numeric"]),
        ("lo", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                             sparse_output=False), g["low_card"]),
        ("hi", OrdinalEncoder(handle_unknown="use_encoded_value",
                              unknown_value=-1), g["high_card"]),
    ], remainder="drop", verbose_feature_names_out=False)
    return Pipeline([("prep", prep), ("reg", regression_to_band_estimator(seed=seed))])


def stratified_eval(yte, pred, rt_te, label):
    """Return dict with overall + per-REPORT_TYPE QWK/F1/bal_acc/n."""
    overall = evaluate_classifier(yte, pred)
    out = {
        "label": label,
        "overall": {
            "n": int(len(yte)),
            "qwk": overall.qwk, "bal_acc": overall.balanced_accuracy,
            "macro_f1": overall.macro_f1, "acc": overall.accuracy,
        }
    }
    for rt_val in sorted(rt_te.dropna().unique()):
        mask = (rt_te == rt_val)
        if mask.sum() < 100:
            continue
        rep = evaluate_classifier(yte[mask], pred[mask])
        out[f"rt_{int(rt_val)}"] = {
            "n": int(mask.sum()),
            "qwk": rep.qwk, "bal_acc": rep.balanced_accuracy,
            "macro_f1": rep.macro_f1, "acc": rep.accuracy,
        }
    return out


def main():
    print("Loading & cleaning ...")
    raw = load_certificates("certificates.csv")
    eng = engineer(build_nondedup(raw))
    tr, te = group_temporal_split(eng)
    X, y, y_reg = to_model_matrix(eng, missingness_ref=tr)  # C3: train-fold missingness
    Xtr = fill_cat(X.loc[tr])
    Xte = fill_cat(X.loc[te])
    ytr, yte = y.loc[tr], y.loc[te]
    yreg_tr = y_reg.loc[tr]
    rt_tr = eng.loc[tr, "REPORT_TYPE"].astype("Int64").reset_index(drop=True)
    rt_te = eng.loc[te, "REPORT_TYPE"].astype("Int64").reset_index(drop=True)
    _uprn_tr = eng.loc[tr, "UPRN"].reset_index(drop=True)
    w_tr = (1.0 / _uprn_tr.map(_uprn_tr.value_counts()).fillna(1.0)).to_numpy()  # M1
    Xtr = Xtr.reset_index(drop=True); Xte = Xte.reset_index(drop=True)
    ytr = ytr.reset_index(drop=True); yte = yte.reset_index(drop=True)
    yreg_tr = yreg_tr.reset_index(drop=True)

    print(f"Train: {Xtr.shape}, Test: {Xte.shape}")
    print(f"Train REPORT_TYPE: {rt_tr.value_counts().to_dict()}")
    print(f"Test  REPORT_TYPE: {rt_te.value_counts().to_dict()}")

    results = {}

    # === 1. UNIFIED MODEL (baseline for comparison) ===
    print("\n=== 1. UNIFIED regression-to-band ===")
    t0 = time.time()
    unified = make_pipeline(Xtr, seed=42)
    unified.fit(Xtr, yreg_tr)
    score_un = unified.predict(Xte)
    pred_un = pd.Series(sap_score_to_band(score_un), index=Xte.index)
    results["1_unified"] = stratified_eval(yte, pred_un, rt_te, "unified")
    print(f"  Unified fit: {time.time()-t0:.0f}s")
    print(f"  Overall    : QWK={results['1_unified']['overall']['qwk']:.4f}  "
          f"bal={results['1_unified']['overall']['bal_acc']:.4f}")
    for k in sorted(results['1_unified'].keys()):
        if k.startswith('rt_'):
            r = results['1_unified'][k]
            print(f"  {k} (n={r['n']}): QWK={r['qwk']:.4f}  bal={r['bal_acc']:.4f}")

    # === 2. STRATIFIED MODELS (one per REPORT_TYPE) ===
    print("\n=== 2. STRATIFIED regression-to-band (one model per REPORT_TYPE) ===")
    models = {}
    for rt_val in sorted(rt_tr.dropna().unique()):
        mask_tr = (rt_tr == rt_val).values
        n = int(mask_tr.sum())
        if n < 500:
            print(f"  REPORT_TYPE={int(rt_val)} skipped (only {n} training rows)")
            continue
        t0 = time.time()
        pipe = make_pipeline(Xtr.loc[mask_tr], seed=42)
        pipe.fit(Xtr.loc[mask_tr], yreg_tr.loc[mask_tr])
        models[int(rt_val)] = pipe
        print(f"  REPORT_TYPE={int(rt_val)} model fit on n={n} ({time.time()-t0:.0f}s)")

    # Route each test row to its REPORT_TYPE model
    score_strat = score_un.copy()
    routed_count = {}
    for rt_val, pipe in models.items():
        mask = (rt_te == rt_val).values
        n_routed = int(mask.sum())
        routed_count[rt_val] = n_routed
        if n_routed > 0:
            score_strat[mask] = pipe.predict(Xte.loc[mask])
    fallback_count = int(len(Xte) - sum(routed_count.values()))
    print(f"  Routing: {routed_count}, fallback (unified): {fallback_count}")
    pred_st = pd.Series(sap_score_to_band(score_strat), index=Xte.index)
    results["2_stratified"] = stratified_eval(yte, pred_st, rt_te, "stratified")
    print(f"\n  Stratified overall: QWK={results['2_stratified']['overall']['qwk']:.4f}  "
          f"bal={results['2_stratified']['overall']['bal_acc']:.4f}")
    for k in sorted(results['2_stratified'].keys()):
        if k.startswith('rt_'):
            r = results['2_stratified'][k]
            print(f"  {k} (n={r['n']}): QWK={r['qwk']:.4f}  bal={r['bal_acc']:.4f}")

    # === 3. HYBRID MODEL (unified + SAP specialist) — THE PRODUCTION CHAMPION ===
    print("\n=== 3. HYBRID (unified + SAP specialist) — production champion ===")
    t0 = time.time()
    pipe_u_hybrid = make_pipeline(Xtr, seed=42)
    pipe_s_hybrid = make_pipeline(Xtr.loc[(rt_tr == 101).values], seed=42)
    champion = SapStratifiedRegressor(seed=42).fit(
        pipe_u_hybrid, pipe_s_hybrid, Xtr, yreg_tr, rt_tr)
    pred_hy = pd.Series(champion.predict_band(Xte, rt_te), index=Xte.index)
    results["3_hybrid"] = stratified_eval(yte, pred_hy, rt_te, "hybrid")
    print(f"  Hybrid fit + predict: {time.time()-t0:.0f}s")
    print(f"  Overall    : QWK={results['3_hybrid']['overall']['qwk']:.4f}  "
          f"bal={results['3_hybrid']['overall']['bal_acc']:.4f}")
    for k in sorted(results['3_hybrid'].keys()):
        if k.startswith('rt_'):
            r = results['3_hybrid'][k]
            print(f"  {k} (n={r['n']}): QWK={r['qwk']:.4f}  bal={r['bal_acc']:.4f}")

    # === 4. DELTAS ===
    print("\n=== 4. UNIFIED -> {STRATIFIED, HYBRID} DELTAS ===")
    for k in sorted(results['1_unified'].keys()):
        if k == 'label': continue
        u = results['1_unified'][k]
        s = results['2_stratified'][k]
        h = results['3_hybrid'][k]
        ds_qwk = s['qwk'] - u['qwk']
        dh_qwk = h['qwk'] - u['qwk']
        flag = ""
        if k.startswith('rt_'):
            flag = "  <-- SAP fairness target" if k == 'rt_101' else ""
        print(f"  {k:12s} (n={u['n']}): "
              f"strat dQWK={ds_qwk:+.4f}  hybrid dQWK={dh_qwk:+.4f}{flag}")

    # === 5. MULTI-SEED STABILITY OF THE HYBRID CHAMPION ===
    print("\n=== 5. Multi-seed stability of hybrid champion (seeds 42, 123, 2026) ===")
    seed_results = []
    for seed in (42, 123, 2026):
        t0 = time.time()
        pipe_u = make_pipeline(Xtr, seed=seed)
        pipe_s = make_pipeline(Xtr.loc[(rt_tr == 101).values], seed=seed)
        ch = SapStratifiedRegressor(seed=seed).fit(
            pipe_u, pipe_s, Xtr, yreg_tr, rt_tr)
        pred = pd.Series(ch.predict_band(Xte, rt_te), index=Xte.index)
        rep = evaluate_classifier(yte, pred)
        seed_results.append({
            "seed": seed, "qwk": rep.qwk, "bal_acc": rep.balanced_accuracy,
            "macro_f1": rep.macro_f1, "acc": rep.accuracy,
            "fit_seconds": time.time() - t0,
        })
        print(f"  seed={seed}: QWK={rep.qwk:.4f}  bal={rep.balanced_accuracy:.4f}  "
              f"f1={rep.macro_f1:.4f}  ({time.time()-t0:.0f}s)")
    qwks = np.array([r["qwk"] for r in seed_results])
    results["4_seed_stability"] = {
        "seeds": seed_results,
        "qwk_mean": float(qwks.mean()),
        "qwk_std": float(qwks.std(ddof=1)),
        "qwk_min": float(qwks.min()),
        "qwk_max": float(qwks.max()),
    }
    print(f"  QWK mean +/- std: {qwks.mean():.4f} +/- {qwks.std(ddof=1):.4f}  "
          f"(range {qwks.min():.4f}..{qwks.max():.4f})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
