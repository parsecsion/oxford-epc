"""Literature-backed robustness metrics for the frozen champion.

Consumes ``reports/predictions_oxford.csv`` (produced by predict_oxford.py)
and writes ``reports/champion_robustness.json`` containing every metric the
2026-05 literature review identified as missing or out-of-date:

* MAE in band-units (Frank & Hall 2001 recommended ordinal-distance loss).
* Linear-weighted kappa (Warrens 2012 robustness check on QWK).
* % within {5, 10, 15} SAP points -- the ONS Data Science Campus (2021) UK
  national benchmark reports 93% within 15.
* Per-class F1 with bootstrap 95% CIs on the CURRENT champion (the existing
  diagnostics.json CIs were computed on the superseded HGB classifier).
* QWK by confidence-proxy quartile -- validates that the proxy in
  predictions_oxford.csv has discriminative power.
* Per-segment (REPORT_TYPE) breakdown of all of the above.

Also regenerates the champion-specific figures whose old versions describe
the superseded LightGBM classifier:

* reports/figures/fig_confusion.png         (current champion's confusion matrix)
* reports/figures/fig_perm_importance.png   (deferred: requires a full fit)
* reports/figures/fig_classwise_f1.png      (per-band F1 with sample sizes)
* reports/figures/fig_score_reliability.png (predicted vs observed SAP
                                              score reliability curve --
                                              the analogue of a calibration
                                              plot for a regression head).

Pure-CSV in, JSON+PNG out. No model fit; runs in seconds.
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import (
    evaluate_classifier, evaluate_regressor,
    mae_band_units, linear_weighted_kappa,
    within_n_sap_points, qwk_by_confidence_quartile,
    bootstrap_per_class_f1_ci,
)
from src import RATING_ORDER, RATING_TO_INT
from src.plots import fig_confusion, fig_classwise_metric

ROOT = Path(__file__).resolve().parents[1]
PRED_CSV = ROOT / "reports" / "predictions_oxford.csv"
OUT_JSON = ROOT / "reports" / "champion_robustness.json"
FIG_DIR = ROOT / "reports" / "figures"


def score_reliability_curve(score_true: np.ndarray, score_pred: np.ndarray,
                             out_path: Path, n_bins: int = 10) -> dict:
    """Bin predictions, plot mean predicted vs mean observed SAP score.

    This is the natural calibration analogue for a regression head: in each
    decile of predicted score, do we observe the score we predicted? Slope
    near 1.0 + intercept near 0 means well-calibrated; systematic deviation
    means a known-correctable bias.
    """
    bins = np.quantile(score_pred, np.linspace(0, 1, n_bins + 1))
    bins[0] -= 1; bins[-1] += 1  # avoid edge exclusion
    idx = np.digitize(score_pred, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    mean_pred, mean_obs, n_per_bin = [], [], []
    for k in range(n_bins):
        mask = idx == k
        if mask.sum() < 5:
            continue
        mean_pred.append(float(score_pred[mask].mean()))
        mean_obs.append(float(score_true[mask].mean()))
        n_per_bin.append(int(mask.sum()))
    mean_pred = np.array(mean_pred); mean_obs = np.array(mean_obs)
    # Fit a 1-D line for the slope/intercept report
    slope, intercept = np.polyfit(mean_pred, mean_obs, 1)

    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    lo, hi = min(mean_pred.min(), mean_obs.min()), max(mean_pred.max(), mean_obs.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="Perfect calibration")
    ax.plot(mean_pred, mean_obs, marker="o", color="#3b6cb7",
            label=f"Champion (slope={slope:.3f}, intercept={intercept:+.2f})")
    for x, y, n in zip(mean_pred, mean_obs, n_per_bin):
        ax.text(x, y + 0.6, f"n={n:,}", ha="center", va="bottom", fontsize=7,
                color="#555")
    ax.set_xlabel("Mean predicted SAP score (per decile)")
    ax.set_ylabel("Mean observed SAP score (per decile)")
    ax.set_title("SAP-score reliability curve — hold-out test set")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "n_bins_with_data": int(len(mean_pred)),
    }


def main() -> int:
    if not PRED_CSV.exists():
        print(f"ERROR: {PRED_CSV} does not exist. "
              "Run scripts/predict_oxford.py first.", file=sys.stderr)
        return 2

    print(f"Loading {PRED_CSV.name} ...")
    df = pd.read_csv(PRED_CSV)
    print(f"  {len(df):,} predictions, "
          f"{len(df[df['split'] == 'test']):,} on holdout")

    # ------------------------------------------------------------------
    # Holdout slice -- the only one we report metrics on
    # ------------------------------------------------------------------
    te = df[df["split"] == "test"].reset_index(drop=True).copy()

    # Coerce dtypes -- evaluate_classifier needs object Series for labels
    y_true_band = te["actual_band"].astype(object)
    y_pred_band = te["predicted_band"].astype(object)
    score_true = te["actual_sap_score"].astype(float).to_numpy()
    score_pred = te["predicted_sap_score"].astype(float).to_numpy()
    confidence = te["confidence_proxy"].astype(float).to_numpy()
    rt = te["REPORT_TYPE"].astype("Int64")

    # ------------------------------------------------------------------
    # 1. Overall robustness metrics
    # ------------------------------------------------------------------
    print("\n[1/6] Overall robustness metrics ...")
    overall_cls = evaluate_classifier(y_true_band, y_pred_band)
    overall_reg = evaluate_regressor(score_true, score_pred)
    overall = {
        "n": int(len(te)),
        # Headline metrics (re-derived to confirm consistency with the
        # frozen-champion record):
        "qwk": overall_cls.qwk,
        "accuracy": overall_cls.accuracy,
        "balanced_accuracy": overall_cls.balanced_accuracy,
        "macro_f1": overall_cls.macro_f1,
        # Robustness additions:
        "mae_band_units": mae_band_units(y_true_band, y_pred_band),
        "linear_weighted_kappa": linear_weighted_kappa(y_true_band, y_pred_band),
        "score_mae": overall_reg["mae"],
        "score_rmse": overall_reg["rmse"],
        "score_r2": overall_reg["r2"],
    }
    overall.update(within_n_sap_points(score_true, score_pred,
                                        n_values=(5, 10, 15)))
    print(f"  QWK = {overall['qwk']:.4f}, "
          f"linear kappa = {overall['linear_weighted_kappa']:.4f}, "
          f"MAE-bands = {overall['mae_band_units']:.3f}")
    print(f"  ONS DSC comparator: "
          f"within  5 SAP pts = {overall['within_5']*100:.1f}%, "
          f"within 10 = {overall['within_10']*100:.1f}%, "
          f"within 15 = {overall['within_15']*100:.1f}%  "
          f"(ONS national benchmark: 93% within 15)")

    # ------------------------------------------------------------------
    # 2. Per-segment (REPORT_TYPE) breakdown of the same metrics
    # ------------------------------------------------------------------
    print("\n[2/6] Per-REPORT_TYPE breakdown ...")
    per_rt = {}
    for rt_val in sorted(rt.dropna().unique()):
        mask = (rt == rt_val).to_numpy()
        if mask.sum() < 100:
            continue
        sub_y_true = y_true_band[mask]; sub_y_pred = y_pred_band[mask]
        sub_s_true = score_true[mask]; sub_s_pred = score_pred[mask]
        cls = evaluate_classifier(sub_y_true, sub_y_pred)
        reg = evaluate_regressor(sub_s_true, sub_s_pred)
        entry = {
            "n": int(mask.sum()),
            "qwk": cls.qwk, "accuracy": cls.accuracy,
            "balanced_accuracy": cls.balanced_accuracy,
            "macro_f1": cls.macro_f1,
            "mae_band_units": mae_band_units(sub_y_true, sub_y_pred),
            "linear_weighted_kappa": linear_weighted_kappa(sub_y_true, sub_y_pred),
            "score_mae": reg["mae"], "score_rmse": reg["rmse"], "score_r2": reg["r2"],
        }
        entry.update(within_n_sap_points(sub_s_true, sub_s_pred))
        per_rt[f"rt_{int(rt_val)}"] = entry
        print(f"  rt_{int(rt_val)} (n={mask.sum()}): "
              f"QWK={cls.qwk:.4f}  MAE-bands={entry['mae_band_units']:.3f}  "
              f"within15={entry['within_15']*100:.1f}%")

    # ------------------------------------------------------------------
    # 3. Per-class F1 bootstrap CIs FOR THE CURRENT CHAMPION
    # ------------------------------------------------------------------
    print("\n[3/6] Per-class F1 bootstrap CIs (1000 resamples) ...")
    ci = bootstrap_per_class_f1_ci(y_true_band, y_pred_band,
                                    n_resamples=1000, seed=42)
    for b in RATING_ORDER:
        c = ci[b]
        print(f"  {b}: F1={c['mean']:.3f}  "
              f"95% CI [{c['ci_low_2.5']:.3f}, {c['ci_high_97.5']:.3f}]  "
              f"width={c['ci_width']:.3f}")

    # ------------------------------------------------------------------
    # 4. Confidence-proxy validation: does it monotone in QWK?
    # ------------------------------------------------------------------
    print("\n[4/6] Confidence-proxy validation ...")
    by_q = qwk_by_confidence_quartile(y_true_band, y_pred_band, confidence)
    for entry in by_q:
        print(f"  quartile {entry['quartile']} (n={entry['n']}, "
              f"conf in [{entry['confidence_range'][0]:.3f}, "
              f"{entry['confidence_range'][1]:.3f}]): "
              f"QWK={entry['qwk']:.4f}  acc={entry['accuracy']:.4f}")
    if len(by_q) >= 2:
        top_qwk = by_q[-1]["qwk"]
        bot_qwk = by_q[0]["qwk"]
        print(f"  Top - Bottom QWK delta = {top_qwk - bot_qwk:+.4f}  "
              f"({'monotone' if top_qwk > bot_qwk else 'NOT MONOTONE -- proxy is uninformative'})")

    # ------------------------------------------------------------------
    # 5. Regenerate the champion-specific figures
    # ------------------------------------------------------------------
    print("\n[5/6] Regenerating champion figures ...")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    yt_int = np.array([RATING_TO_INT[b] for b in y_true_band])
    yp_int = np.array([RATING_TO_INT[b] for b in y_pred_band])
    cm = confusion_matrix(yt_int, yp_int, labels=list(range(len(RATING_ORDER))))
    fig_confusion(cm, FIG_DIR / "fig_confusion.png")
    print(f"  wrote {FIG_DIR/'fig_confusion.png'}")

    fig_classwise_metric(overall_cls.per_class, FIG_DIR / "fig_classwise_f1.png",
                          metric="f1-score")
    print(f"  wrote {FIG_DIR/'fig_classwise_f1.png'}")

    reliability = score_reliability_curve(
        score_true, score_pred, FIG_DIR / "fig_score_reliability.png")
    print(f"  wrote {FIG_DIR/'fig_score_reliability.png'} "
          f"(slope={reliability['slope']:.3f}, "
          f"intercept={reliability['intercept']:+.2f})")

    # ------------------------------------------------------------------
    # 6. Write the JSON
    # ------------------------------------------------------------------
    print("\n[6/6] Writing JSON ...")
    out = {
        "source": str(PRED_CSV.relative_to(ROOT)),
        "ons_dsc_benchmark_note": (
            "ONS Data Science Campus (2021) 'Using Machine Learning to Predict "
            "Energy Efficiency' reports 57% within 5 SAP points and 93% within "
            "15 SAP points as the UK national benchmark."
        ),
        "overall": overall,
        "per_report_type": per_rt,
        "per_class_f1_bootstrap_ci": ci,
        "confidence_proxy_quartiles": by_q,
        "score_reliability": reliability,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {OUT_JSON.relative_to(ROOT)}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
