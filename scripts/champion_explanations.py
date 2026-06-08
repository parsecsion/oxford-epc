"""Permutation importance + SHAP for the current frozen champion.

Replaces the stale fig_perm_importance.png and fig_shap_summary.png that
were generated for the superseded LightGBM classifier. Uses the frozen
SapStratifiedRegressor (artefacts/champion.pkl) and the holdout slice of
reports/predictions_oxford.csv as input.

Outputs:
  reports/figures/fig_perm_importance.png  -- permutation importance plot
  reports/figures/fig_shap_summary.png     -- SHAP beeswarm on unified regressor
  reports/permutation_importance.csv       -- top-20 permutation deltas
  reports/champion_explanations.json       -- bookkeeping (timings, sample
                                              sizes, top-20 features) so the
                                              verifier can confirm the
                                              outputs are fresh.

Notes:
  * sklearn's permutation_importance expects a scoring callable that takes
    (estimator, X, y) -> float. SapStratifiedRegressor isn't an sklearn
    estimator (it has predict_band rather than predict), so we use a
    hand-rolled permutation routine that calls predict_band directly and
    scores on Quadratic Weighted Kappa. This is exactly the metric the
    project optimises against, so the resulting importance ranking is
    interpretable as "by how much would shuffling this column degrade the
    headline metric".
  * SHAP is computed on the UNIFIED regressor only -- the SAP-cohort
    specialist is fit on REPORT_TYPE==101 rows (~10% of training data) and
    its predictions are interpretable on the same feature scale, so a
    single SHAP plot on the unified arm is the cleanest representation of
    the global driver structure. The caption in the report should make
    this scope explicit.
"""
from __future__ import annotations
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import cohen_kappa_score

from src.data import (load_certificates, filter_oxford, coerce_numeric,
                      validate_consistency, cap_outliers, drop_fully_missing,
                      group_temporal_split, to_model_matrix)
from src.features import engineer
from src.models import sap_score_to_band
from src import RATING_TO_INT
from src.plots import fig_perm_importance

ROOT = Path(__file__).resolve().parents[1]
PICKLE_PATH = ROOT / "artefacts" / "champion.pkl"
FIG_DIR = ROOT / "reports" / "figures"
JSON_PATH = ROOT / "reports" / "champion_explanations.json"
PERM_CSV = ROOT / "reports" / "permutation_importance.csv"
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


def hybrid_qwk(champion, X, y_band_int, rt) -> float:
    """Score = Quadratic Weighted Kappa between predicted band and truth."""
    pred = champion.predict_band(X, rt)
    pred_int = np.array([RATING_TO_INT[b] for b in pred])
    return float(cohen_kappa_score(y_band_int, pred_int,
                                    labels=list(range(7)), weights="quadratic"))


def permutation_importance_hybrid(champion, X, y_band_int, rt,
                                    n_repeats: int = 5, seed: int = 42,
                                    sample_n: int = 4000) -> pd.DataFrame:
    """Custom permutation importance scoring on hybrid-routed band QWK.

    Shuffles each feature column ``n_repeats`` times and records the drop in
    QWK relative to the unpermuted baseline. Returns a tidy DataFrame:
    feature, importance_mean, importance_std.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    if sample_n < n:
        idx = rng.choice(n, size=sample_n, replace=False)
        Xs = X.iloc[idx].reset_index(drop=True)
        ys = y_band_int[idx]
        rts = rt.iloc[idx].reset_index(drop=True)
    else:
        Xs = X.reset_index(drop=True); ys = y_band_int
        rts = rt.reset_index(drop=True)

    baseline = hybrid_qwk(champion, Xs, ys, rts)
    print(f"  baseline QWK on sample of {len(Xs):,}: {baseline:.4f}")

    rows = []
    for col_i, col in enumerate(Xs.columns):
        drops = []
        for r in range(n_repeats):
            X_perm = Xs.copy()
            perm_idx = rng.permutation(len(X_perm))
            X_perm[col] = X_perm[col].values[perm_idx]
            qwk_perm = hybrid_qwk(champion, X_perm, ys, rts)
            drops.append(baseline - qwk_perm)
        rows.append({
            "feature": col,
            "importance_mean": float(np.mean(drops)),
            "importance_std": float(np.std(drops)),
        })
        if (col_i + 1) % 10 == 0 or col_i == len(Xs.columns) - 1:
            print(f"  permuted {col_i + 1}/{len(Xs.columns)} features")
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False)


def shap_summary_unified(champion, X, rt, out_path: Path,
                          sample_n: int = 2000):
    """SHAP TreeExplainer beeswarm on the unified regressor.

    Limited to REPORT_TYPE != 101 (RdSAP rows) because that's the path the
    unified regressor actually serves at inference time; the SAP-only
    specialist has its own (much smaller) explainer.
    """
    try:
        import shap
    except ImportError as e:
        print(f"  shap not installed ({e}); skipping SHAP summary")
        return {"skipped": True, "reason": "shap not installed"}

    rdsap_mask = (rt != 101).fillna(True).values
    X_rdsap = X.loc[rdsap_mask].reset_index(drop=True)
    rng = np.random.default_rng(42)
    if len(X_rdsap) > sample_n:
        idx = rng.choice(len(X_rdsap), size=sample_n, replace=False)
        X_shap = X_rdsap.iloc[idx]
    else:
        X_shap = X_rdsap

    pipe_u = champion.unified  # sklearn Pipeline: prep -> reg
    prep = pipe_u.named_steps["prep"]
    reg = pipe_u.named_steps["reg"]
    X_t = prep.transform(X_shap)
    feat_names = prep.get_feature_names_out()

    explainer = shap.TreeExplainer(reg)
    sv = explainer.shap_values(X_t)

    fig = plt.figure(figsize=(8, 5.2))
    shap.summary_plot(sv, X_t, feature_names=feat_names, show=False,
                       max_display=15)
    plt.title("SHAP summary - unified regressor (RdSAP path)\n"
              "SHAP units: SAP score points")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "sample_n": int(len(X_shap)),
        "n_features": int(X_t.shape[1]),
        "skipped": False,
    }


def main() -> int:
    if not PICKLE_PATH.exists():
        print(f"ERROR: {PICKLE_PATH} missing", file=sys.stderr)
        return 2

    print("Loading frozen champion ...")
    champion = pickle.load(open(PICKLE_PATH, "rb"))

    print("Loading & cleaning Oxford corpus ...")
    raw = load_certificates(ROOT / "certificates.csv")
    eng = engineer(build_nondedup(raw))
    tr, te = group_temporal_split(eng)
    X, y, _ = to_model_matrix(eng, missingness_ref=tr)  # match champion schema
    Xte = fill_cat(X.loc[te]).reset_index(drop=True)
    yte = y.loc[te].reset_index(drop=True)
    rt_te = eng.loc[te, "REPORT_TYPE"].astype("Int64").reset_index(drop=True)
    yte_int = np.array([RATING_TO_INT[b] for b in yte])
    print(f"  holdout: {Xte.shape}")

    # --- Permutation importance on the hybrid champion ---
    print("\n[1/3] Permutation importance (sample=4000, 5 repeats per feature)")
    print("       This drives 85 * 5 = 425 hybrid predictions; ~5-10 min ...")
    t0 = time.time()
    imp_df = permutation_importance_hybrid(
        champion, Xte, yte_int, rt_te,
        n_repeats=5, seed=42, sample_n=4000)
    perm_elapsed = time.time() - t0
    PERM_CSV.parent.mkdir(parents=True, exist_ok=True)
    imp_df.to_csv(PERM_CSV, index=False)
    print(f"  wrote {PERM_CSV.relative_to(ROOT)} "
          f"({len(imp_df)} features, {perm_elapsed:.0f}s)")

    # Plot top 20
    imp_series = imp_df.set_index("feature")["importance_mean"]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_perm_importance(imp_series.head(20),
                         FIG_DIR / "fig_perm_importance.png", top=20)
    print(f"  wrote {FIG_DIR/'fig_perm_importance.png'}")

    # --- SHAP on unified regressor ---
    print("\n[2/3] SHAP summary (unified regressor, RdSAP path, sample=2000)")
    t0 = time.time()
    shap_info = shap_summary_unified(
        champion, Xte, rt_te, FIG_DIR / "fig_shap_summary.png", sample_n=2000)
    shap_elapsed = time.time() - t0
    if not shap_info.get("skipped"):
        print(f"  wrote {FIG_DIR/'fig_shap_summary.png'} "
              f"({shap_info['sample_n']} samples, {shap_elapsed:.0f}s)")

    # --- Write JSON bookkeeping ---
    print("\n[3/3] Writing JSON ...")
    out = {
        "frozen_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "champion_pickle": str(PICKLE_PATH.relative_to(ROOT)),
        "permutation_importance": {
            "sample_n": 4000,
            "n_repeats": 5,
            "scoring": "Quadratic Weighted Kappa on hybrid-routed band predictions",
            "elapsed_seconds": float(perm_elapsed),
            "top_20": imp_df.head(20).to_dict(orient="records"),
        },
        "shap": {
            **shap_info,
            "elapsed_seconds": float(shap_elapsed) if not shap_info.get("skipped") else None,
            "scope": "unified regressor (RdSAP path), SHAP values in SAP-score units",
        },
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {JSON_PATH.relative_to(ROOT)}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
