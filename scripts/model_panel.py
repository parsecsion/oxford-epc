"""Comprehensive model panel: baselines, regression head, ensemble,
threshold tuning, calibration, REPORT_TYPE weighting.

Trained on the group-split training fold; evaluated on the holdout test fold.
GroupKFold CV from cv_groupkfold.py provides the stability estimates.

All models report the same metrics (acc, balanced_accuracy, macro_f1, qwk)
plus per-class F1 with bootstrap CIs.
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
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import cohen_kappa_score, f1_score

from src.data import (load_certificates, filter_oxford, coerce_numeric,
                      validate_consistency, cap_outliers, drop_fully_missing,
                      group_temporal_split, to_model_matrix)
from src.features import engineer
from src.models import (build_pipeline, rf_classifier, hist_gradient_boosting,
                        lgbm_classifier, infer_column_groups)
from src.evaluation import evaluate_classifier
from src import RATING_ORDER, RATING_TO_INT, INT_TO_RATING

OUT = Path("reports/model_panel.json")
N_BOOTSTRAP = 500


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


# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------

def full_metrics(y_true, y_pred):
    rep = evaluate_classifier(y_true, y_pred)
    return {
        "acc": rep.accuracy, "bal_acc": rep.balanced_accuracy,
        "macro_f1": rep.macro_f1, "qwk": rep.qwk,
        "per_class_f1": {r: rep.per_class[r]["f1-score"] for r in RATING_ORDER},
    }


def bootstrap_per_class_f1(y_true, y_pred, n=N_BOOTSTRAP, seed=42):
    yi = y_true.map(RATING_TO_INT).to_numpy()
    yp = pd.Series(y_pred, index=y_true.index).map(RATING_TO_INT).to_numpy()
    rng = np.random.RandomState(seed)
    nrow = len(yi)
    samples = {r: [] for r in RATING_ORDER}
    for _ in range(n):
        idx = rng.randint(0, nrow, size=nrow)
        f1s = f1_score(yi[idx], yp[idx], labels=list(range(7)),
                       average=None, zero_division=0)
        for r, f in zip(RATING_ORDER, f1s):
            samples[r].append(float(f))
    return {
        r: {
            "mean": float(np.mean(samples[r])),
            "ci_low": float(np.percentile(samples[r], 2.5)),
            "ci_high": float(np.percentile(samples[r], 97.5)),
        } for r in RATING_ORDER
    }


# -------------------------------------------------------------------
# Pipeline builders
# -------------------------------------------------------------------

def rf_pipeline(X_train):
    return build_pipeline(rf_classifier(n_estimators=400), X_train)


def hgb_pipeline(X_train, **kw):
    g = infer_column_groups(X_train)
    prep = ColumnTransformer([
        ("num", "passthrough", g["numeric"]),
        ("lo", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                             sparse_output=False), g["low_card"]),
        ("hi", OrdinalEncoder(handle_unknown="use_encoded_value",
                              unknown_value=-1), g["high_card"]),
    ], remainder="drop", verbose_feature_names_out=False)
    clf = hist_gradient_boosting(max_iter=400, **kw)
    return Pipeline([("prep", prep), ("clf", clf)])


def lgbm_pipeline(X_train):
    g = infer_column_groups(X_train)
    from sklearn.impute import SimpleImputer
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


# -------------------------------------------------------------------
# Approaches
# -------------------------------------------------------------------

def predict_via_argmax(pipe, X):
    return pd.Series(pipe.predict(X), index=X.index)


def predict_via_expected_value(pipe, X):
    """Take the expected ordinal value of the predicted class distribution."""
    proba = pipe.predict_proba(X)  # (n, 7)
    classes = pipe.classes_  # e.g. ['A','B',...,'G']
    cls_int = np.array([RATING_TO_INT[c] for c in classes])
    expv = proba @ cls_int  # (n,)
    rounded = np.clip(np.round(expv).astype(int), 0, 6)
    return pd.Series([INT_TO_RATING[i] for i in rounded], index=X.index)


def tune_thresholds_for_qwk(pipe, X_train, y_train, X_val):
    """Tune 6 thresholds on training probabilities to maximise QWK on val.

    Predicts class k where cumulative prob first exceeds threshold_k.
    A simple coordinate-descent search over each threshold in [0.1, 0.9].
    """
    yi_tr = y_train.map(RATING_TO_INT).to_numpy()
    p_tr = pipe.predict_proba(X_train)  # (n_tr, 7)
    classes = pipe.classes_
    cls_idx = [list(classes).index(r) for r in RATING_ORDER]
    p_tr = p_tr[:, cls_idx]  # reordered to A..G

    # Use cumulative prob from highest band (G) downwards; threshold_k =
    # the minimum cumprob to declare class <= k
    cum = np.cumsum(p_tr, axis=1)  # cum[i, k] = P(Y <= k+1)

    def predict_with_thresh(cum_arr, thresh):
        # For each row, the predicted class is the smallest k such that
        # cum[i, k] >= thresh[k], else 6 (G)
        out = np.full(len(cum_arr), 6, dtype=int)
        for k in range(7):
            mask = (out == 6) & (cum_arr[:, k] >= thresh[k])
            out[mask] = k
        return out

    # Start from defaults derived from class priors
    init = np.array([0.5] * 7)
    best = init.copy()
    yp = predict_with_thresh(cum, best)
    best_qwk = cohen_kappa_score(yi_tr, yp, weights="quadratic",
                                  labels=list(range(7)))
    grid = np.linspace(0.1, 0.9, 17)
    improved = True
    iters = 0
    while improved and iters < 3:
        improved = False
        for k in range(7):
            for cand in grid:
                t = best.copy(); t[k] = cand
                yp = predict_with_thresh(cum, t)
                q = cohen_kappa_score(yi_tr, yp, weights="quadratic",
                                       labels=list(range(7)))
                if q > best_qwk + 1e-5:
                    best, best_qwk = t, q
                    improved = True
        iters += 1

    # Apply to validation
    p_va = pipe.predict_proba(X_val)[:, cls_idx]
    cum_va = np.cumsum(p_va, axis=1)
    yp_va_int = predict_with_thresh(cum_va, best)
    return pd.Series([INT_TO_RATING[i] for i in yp_va_int], index=X_val.index), best.tolist()


def ensemble_average(pipes, X):
    """Average predict_proba across pipelines; argmax the mean."""
    probs = []
    classes = None
    for p in pipes:
        pr = p.predict_proba(X)
        if classes is None:
            classes = list(p.classes_)
        else:
            assert list(p.classes_) == classes, "Class orders differ"
        probs.append(pr)
    mean_p = np.mean(probs, axis=0)
    preds = np.array(classes)[mean_p.argmax(axis=1)]
    return pd.Series(preds, index=X.index), mean_p, classes


def regression_to_band(score: np.ndarray) -> np.ndarray:
    """SAP score -> band mapping per DLUHC EPC rating boundaries.

    A: 92+, B: 81-91, C: 69-80, D: 55-68, E: 39-54, F: 21-38, G: 1-20
    """
    def _one(s):
        if s >= 92: return "A"
        if s >= 81: return "B"
        if s >= 69: return "C"
        if s >= 55: return "D"
        if s >= 39: return "E"
        if s >= 21: return "F"
        return "G"
    return np.array([_one(s) for s in score])


def regression_pipeline(X_train):
    from sklearn.ensemble import HistGradientBoostingRegressor
    g = infer_column_groups(X_train)
    prep = ColumnTransformer([
        ("num", "passthrough", g["numeric"]),
        ("lo", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                             sparse_output=False), g["low_card"]),
        ("hi", OrdinalEncoder(handle_unknown="use_encoded_value",
                              unknown_value=-1), g["high_card"]),
    ], remainder="drop", verbose_feature_names_out=False)
    reg = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.07, max_leaf_nodes=63,
        min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=20, random_state=42,
    )
    return Pipeline([("prep", prep), ("reg", reg)])


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    print("Loading & cleaning ...")
    raw = load_certificates("certificates.csv")
    eng = engineer(build_nondedup_frame(raw))
    tr, te = group_temporal_split(eng)
    X, y, y_reg_full = to_model_matrix(eng)
    Xtr = X.loc[tr].reset_index(drop=True)
    ytr = y.loc[tr].reset_index(drop=True)
    Xte = X.loc[te].reset_index(drop=True)
    yte = y.loc[te].reset_index(drop=True)
    y_reg_tr = y_reg_full.loc[tr].reset_index(drop=True)
    print(f"Train: {Xtr.shape}, Test: {Xte.shape}")

    Xtr_f = fill_cat_nans(Xtr)
    Xte_f = fill_cat_nans(Xte)

    results = {}

    # ===============================================
    # 1. Baselines
    # ===============================================
    print("\n=== 1. Baselines ===")
    dummy = DummyClassifier(strategy="stratified", random_state=42)
    dummy.fit(Xtr, ytr)
    pred = pd.Series(dummy.predict(Xte), index=Xte.index)
    results["1a_dummy_stratified"] = full_metrics(yte, pred)
    print(f"  Stratified-dummy:        QWK={results['1a_dummy_stratified']['qwk']:.4f}")

    most_freq = DummyClassifier(strategy="most_frequent")
    most_freq.fit(Xtr, ytr)
    pred = pd.Series(most_freq.predict(Xte), index=Xte.index)
    results["1b_dummy_most_frequent"] = full_metrics(yte, pred)
    print(f"  Most-frequent:           QWK={results['1b_dummy_most_frequent']['qwk']:.4f}")

    # Age-only LogReg
    age = Xtr["CONSTRUCTION_AGE_NUM"].astype("float64").fillna(Xtr["CONSTRUCTION_AGE_NUM"].median())
    age_te = Xte["CONSTRUCTION_AGE_NUM"].astype("float64").fillna(age.median())
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler().fit(age.values.reshape(-1, 1))
    age_lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    age_lr.fit(scaler.transform(age.values.reshape(-1, 1)), ytr)
    pred = pd.Series(age_lr.predict(scaler.transform(age_te.values.reshape(-1, 1))),
                     index=Xte.index)
    results["1c_age_only_lr"] = full_metrics(yte, pred)
    print(f"  Age-only LogReg:         QWK={results['1c_age_only_lr']['qwk']:.4f}")

    # Full LogReg baseline
    from src.models import logistic_baseline
    lr_pipe = build_pipeline(logistic_baseline(), Xtr)
    t0 = time.time()
    lr_pipe.fit(Xtr, ytr)
    pred = pd.Series(lr_pipe.predict(Xte), index=Xte.index)
    results["1d_full_logreg"] = full_metrics(yte, pred)
    print(f"  Full LogReg:             QWK={results['1d_full_logreg']['qwk']:.4f}  ({time.time()-t0:.0f}s)")

    # ===============================================
    # 2. Individual classifiers (champion candidates)
    # ===============================================
    print("\n=== 2. Individual classifiers ===")
    t0 = time.time()
    rf = rf_pipeline(Xtr); rf.fit(Xtr, ytr)
    print(f"  RF trained ({time.time()-t0:.0f}s)")

    t0 = time.time()
    hgb = hgb_pipeline(Xtr_f); hgb.fit(Xtr_f, ytr)
    print(f"  HistGB trained ({time.time()-t0:.0f}s)")

    t0 = time.time()
    lgbm = lgbm_pipeline(Xtr); lgbm.fit(Xtr, ytr)
    print(f"  LightGBM trained ({time.time()-t0:.0f}s)")

    results["2a_rf"] = full_metrics(yte, predict_via_argmax(rf, Xte))
    results["2b_hgb"] = full_metrics(yte, predict_via_argmax(hgb, Xte_f))
    results["2c_lgbm"] = full_metrics(yte, predict_via_argmax(lgbm, Xte))
    print(f"  RF:                      QWK={results['2a_rf']['qwk']:.4f}")
    print(f"  HistGB:                  QWK={results['2b_hgb']['qwk']:.4f}")
    print(f"  LightGBM:                QWK={results['2c_lgbm']['qwk']:.4f}")

    # ===============================================
    # 3. Expected-value prediction
    # ===============================================
    print("\n=== 3. Expected-value prediction (ordinal-aware decode) ===")
    results["3a_rf_expv"] = full_metrics(yte, predict_via_expected_value(rf, Xte))
    results["3b_hgb_expv"] = full_metrics(yte, predict_via_expected_value(hgb, Xte_f))
    results["3c_lgbm_expv"] = full_metrics(yte, predict_via_expected_value(lgbm, Xte))
    print(f"  RF expected:             QWK={results['3a_rf_expv']['qwk']:.4f}  (vs argmax {results['2a_rf']['qwk']:.4f})")
    print(f"  HistGB expected:         QWK={results['3b_hgb_expv']['qwk']:.4f}  (vs argmax {results['2b_hgb']['qwk']:.4f})")
    print(f"  LightGBM expected:       QWK={results['3c_lgbm_expv']['qwk']:.4f}  (vs argmax {results['2c_lgbm']['qwk']:.4f})")

    # ===============================================
    # 4. Ensemble
    # ===============================================
    print("\n=== 4. Ensemble (mean of probabilities) ===")
    # Need all pipes to predict_proba on the same X — use the unfilled Xte for RF/LGBM
    # and Xte_f for HGB. So we wrap them.
    def get_proba(model, X, X_filled):
        if model is hgb:
            return model.predict_proba(X_filled)
        return model.predict_proba(X)

    proba_rf = get_proba(rf, Xte, Xte_f)
    proba_hgb = get_proba(hgb, Xte, Xte_f)
    proba_lgbm = get_proba(lgbm, Xte, Xte_f)
    # Verify class order matches
    assert list(rf.classes_) == list(hgb.classes_) == list(lgbm.classes_), "class order mismatch"
    mean_p = (proba_rf + proba_hgb + proba_lgbm) / 3
    pred = pd.Series(np.array(rf.classes_)[mean_p.argmax(axis=1)], index=Xte.index)
    results["4a_ensemble_argmax"] = full_metrics(yte, pred)
    # Expected value of the ensemble
    cls_int = np.array([RATING_TO_INT[c] for c in rf.classes_])
    expv = mean_p @ cls_int
    pred_expv = pd.Series([INT_TO_RATING[int(round(np.clip(v, 0, 6)))] for v in expv],
                          index=Xte.index)
    results["4b_ensemble_expv"] = full_metrics(yte, pred_expv)
    print(f"  Ensemble argmax:         QWK={results['4a_ensemble_argmax']['qwk']:.4f}")
    print(f"  Ensemble expected-value: QWK={results['4b_ensemble_expv']['qwk']:.4f}")

    # ===============================================
    # 5. Regression head -> band
    # ===============================================
    print("\n=== 5. Regression head -> band mapping ===")
    reg = regression_pipeline(Xtr_f)
    t0 = time.time()
    reg.fit(Xtr_f, y_reg_tr)
    score = reg.predict(Xte_f)
    pred = pd.Series(regression_to_band(score), index=Xte.index)
    results["5_regression_to_band"] = full_metrics(yte, pred)
    # Also save the regression metrics
    from sklearn.metrics import mean_absolute_error, r2_score
    y_reg_te = y_reg_full.loc[te].reset_index(drop=True)
    results["5_regression_to_band"]["regression_mae"] = float(mean_absolute_error(y_reg_te, score))
    results["5_regression_to_band"]["regression_r2"] = float(r2_score(y_reg_te, score))
    print(f"  Regression->band:        QWK={results['5_regression_to_band']['qwk']:.4f}")
    print(f"  (Regression MAE={results['5_regression_to_band']['regression_mae']:.2f}, "
          f"R2={results['5_regression_to_band']['regression_r2']:.3f}) ({time.time()-t0:.0f}s)")

    # ===============================================
    # 6. REPORT_TYPE sample weighting
    # ===============================================
    print("\n=== 6. REPORT_TYPE sample-weighted HistGB ===")
    rt_tr = eng.loc[tr, "REPORT_TYPE"].reset_index(drop=True)
    # Weight = 1 / (frequency of own REPORT_TYPE) in training
    rt_counts = rt_tr.value_counts()
    sample_w = rt_tr.map(lambda v: 1.0 / rt_counts.get(v, 1) if pd.notna(v) else 1.0)
    sample_w = sample_w / sample_w.mean()  # normalise
    t0 = time.time()
    hgb_w = hgb_pipeline(Xtr_f)
    hgb_w.fit(Xtr_f, ytr, clf__sample_weight=sample_w.values)
    pred = pd.Series(hgb_w.predict(Xte_f), index=Xte.index)
    results["6_hgb_rt_weighted"] = full_metrics(yte, pred)
    print(f"  HGB + RT-weight:         QWK={results['6_hgb_rt_weighted']['qwk']:.4f}  ({time.time()-t0:.0f}s)")
    # Stratified eval
    rt_te = eng.loc[te, "REPORT_TYPE"].reset_index(drop=True)
    for rt_val in sorted(rt_te.dropna().unique()):
        mask = (rt_te == rt_val)
        if mask.sum() < 100: continue
        rep = evaluate_classifier(yte[mask], pred[mask])
        results["6_hgb_rt_weighted"][f"stratum_rt{int(rt_val)}_qwk"] = rep.qwk
        results["6_hgb_rt_weighted"][f"stratum_rt{int(rt_val)}_n"] = int(mask.sum())
        print(f"    stratum RT={int(rt_val)} (n={mask.sum()}): QWK={rep.qwk:.4f}")

    # Compare with un-weighted HGB on same stratification
    pred_unw = pd.Series(hgb.predict(Xte_f), index=Xte.index)
    for rt_val in sorted(rt_te.dropna().unique()):
        mask = (rt_te == rt_val)
        if mask.sum() < 100: continue
        rep = evaluate_classifier(yte[mask], pred_unw[mask])
        results["2b_hgb"][f"stratum_rt{int(rt_val)}_qwk"] = rep.qwk

    # ===============================================
    # 7. Calibration of HistGB
    # ===============================================
    print("\n=== 7. Calibrated HistGB (isotonic) ===")
    t0 = time.time()
    hgb_cal = CalibratedClassifierCV(
        estimator=hgb_pipeline(Xtr_f),
        method="isotonic", cv=3, n_jobs=1)
    hgb_cal.fit(Xtr_f, ytr)
    pred = pd.Series(hgb_cal.predict(Xte_f), index=Xte.index)
    results["7_hgb_calibrated"] = full_metrics(yte, pred)
    # Also expected value with calibration
    proba_cal = hgb_cal.predict_proba(Xte_f)
    cls_int_cal = np.array([RATING_TO_INT[c] for c in hgb_cal.classes_])
    expv = proba_cal @ cls_int_cal
    pred_expv = pd.Series([INT_TO_RATING[int(round(np.clip(v, 0, 6)))] for v in expv],
                          index=Xte.index)
    results["7_hgb_calibrated_expv"] = full_metrics(yte, pred_expv)
    print(f"  HGB calibrated:          QWK={results['7_hgb_calibrated']['qwk']:.4f}  ({time.time()-t0:.0f}s)")
    print(f"  HGB cal + expv:          QWK={results['7_hgb_calibrated_expv']['qwk']:.4f}")

    # ===============================================
    # 8. Bootstrap CIs on the champion's per-class F1
    # ===============================================
    # Pick the highest-QWK config
    qwk_ranked = sorted(results.items(), key=lambda kv: kv[1]["qwk"], reverse=True)
    champ_name = qwk_ranked[0][0]
    print(f"\n=== Champion: {champ_name} (QWK={qwk_ranked[0][1]['qwk']:.4f}) ===")
    print(f"Re-fitting champion to get predictions for bootstrap ...")
    # Lookup the champion's prediction; some are already stored above
    champ_predictions = {
        "2a_rf": predict_via_argmax(rf, Xte),
        "2b_hgb": predict_via_argmax(hgb, Xte_f),
        "2c_lgbm": predict_via_argmax(lgbm, Xte),
        "3a_rf_expv": predict_via_expected_value(rf, Xte),
        "3b_hgb_expv": predict_via_expected_value(hgb, Xte_f),
        "3c_lgbm_expv": predict_via_expected_value(lgbm, Xte),
        "4a_ensemble_argmax": pd.Series(np.array(rf.classes_)[mean_p.argmax(axis=1)],
                                        index=Xte.index),
        "4b_ensemble_expv": pred_expv,
        "5_regression_to_band": pd.Series(regression_to_band(reg.predict(Xte_f)),
                                          index=Xte.index),
        "6_hgb_rt_weighted": pd.Series(hgb_w.predict(Xte_f), index=Xte.index),
        "7_hgb_calibrated": pd.Series(hgb_cal.predict(Xte_f), index=Xte.index),
    }
    if champ_name in champ_predictions:
        cis = bootstrap_per_class_f1(yte, champ_predictions[champ_name])
        results["__champion_bootstrap_ci"] = {"name": champ_name, "per_class_f1_ci": cis}
        print(f"  Per-class F1 95% CIs:")
        for r in RATING_ORDER:
            print(f"    {r}: {cis[r]['mean']:.3f} [{cis[r]['ci_low']:.3f}, {cis[r]['ci_high']:.3f}]")

    # ===============================================
    # Summary table
    # ===============================================
    print("\n" + "=" * 80)
    print(f"{'Approach':35s}  Acc      BalAcc   F1       QWK")
    print("-" * 80)
    rows = []
    for k, v in results.items():
        if k.startswith("__"): continue
        rows.append((k, v["acc"], v["bal_acc"], v["macro_f1"], v["qwk"]))
    rows.sort(key=lambda r: r[4], reverse=True)
    for k, a, b, f, q in rows:
        print(f"{k:35s}  {a:.4f}   {b:.4f}   {f:.4f}   {q:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
