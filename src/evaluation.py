"""Evaluation: ordinal-aware metrics, calibration, fairness segments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score, brier_score_loss, classification_report,
    cohen_kappa_score, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

from . import RATING_ORDER, RATING_TO_INT


def encode_rating(y: pd.Series) -> np.ndarray:
    return y.map(RATING_TO_INT).to_numpy()


@dataclass
class ClassificationReport:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    qwk: float
    confusion: np.ndarray
    per_class: dict

    def to_dict(self) -> dict:
        return {
            "accuracy": float(self.accuracy),
            "balanced_accuracy": float(self.balanced_accuracy),
            "macro_f1": float(self.macro_f1),
            "qwk": float(self.qwk),
            "confusion": self.confusion.tolist(),
            "per_class": self.per_class,
        }


def evaluate_classifier(y_true: pd.Series, y_pred: pd.Series) -> ClassificationReport:
    yt = encode_rating(y_true)
    yp = y_pred.map(RATING_TO_INT).to_numpy() if y_pred.dtype == object else np.asarray(y_pred)
    labels = list(range(len(RATING_ORDER)))
    cm = confusion_matrix(yt, yp, labels=labels)
    per = classification_report(
        yt, yp, labels=labels, target_names=RATING_ORDER,
        output_dict=True, zero_division=0,
    )
    return ClassificationReport(
        accuracy=float((yt == yp).mean()),
        balanced_accuracy=float(balanced_accuracy_score(yt, yp)),
        macro_f1=float(f1_score(yt, yp, average="macro", labels=labels, zero_division=0)),
        qwk=float(cohen_kappa_score(yt, yp, labels=labels, weights="quadratic")),
        confusion=cm,
        per_class=per,
    )


def evaluate_regressor(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Robustness metrics for the ordinal classifier (added 2026-05 review pass)
#
# Rationale: QWK has known pathologies (Warrens 2012, Psychometrika) -- under
# some marginal distributions it ignores the central cell entirely, and it is
# sensitive to the number of categories. We therefore complement QWK with two
# robustness metrics requested by the literature review:
#
#   * MAE in band-units: the mean absolute distance between predicted and true
#     band positions on the ordinal scale 0..6. Directly interpretable as
#     "average number of bands off". The Frank & Hall (2001) ECML paper that
#     introduced ordinal-classification-via-regression recommends a band-units
#     loss for evaluation.
#   * Linear-weighted kappa: same Cohen's kappa but with linear (not quadratic)
#     weighting of off-diagonal cells. Less aggressive penalty for far errors,
#     which Warrens 2012 shows is more stable across marginals than QWK.
#
# For the regression head we also expose a percent-within-N-SAP-points metric,
# which is what the ONS Data Science Campus (Using Machine Learning to Predict
# Energy Efficiency, 2021) reports for the closest UK-national comparator
# (their headline: 93% of predictions within 15 SAP points). Without an
# external comparator our headline QWK sits in a vacuum; with it, we can
# state where we stand relative to the published UK benchmark.
# ---------------------------------------------------------------------------

def mae_band_units(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean absolute error on the ordinal 0..6 band scale.

    Both inputs are A..G labels (object dtype) or already-encoded 0..6 ints.
    Returns the mean |true_idx - pred_idx|, where the index is the position
    in ``RATING_ORDER``. A value of 0 means perfect prediction; 0.5 means
    "half a band off on average".
    """
    yt = encode_rating(y_true) if y_true.dtype == object else np.asarray(y_true)
    yp = y_pred.map(RATING_TO_INT).to_numpy() if (hasattr(y_pred, "dtype") and y_pred.dtype == object) else np.asarray(y_pred)
    return float(np.abs(yt - yp).mean())


def linear_weighted_kappa(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Cohen's kappa with linear (not quadratic) off-diagonal weights.

    Recommended by Warrens (2012) as a QWK robustness check.
    """
    yt = encode_rating(y_true) if y_true.dtype == object else np.asarray(y_true)
    yp = y_pred.map(RATING_TO_INT).to_numpy() if (hasattr(y_pred, "dtype") and y_pred.dtype == object) else np.asarray(y_pred)
    labels = list(range(len(RATING_ORDER)))
    return float(cohen_kappa_score(yt, yp, labels=labels, weights="linear"))


def within_n_sap_points(score_true: np.ndarray, score_pred: np.ndarray,
                         n_values: tuple[int, ...] = (5, 10, 15)) -> dict[str, float]:
    """Fraction of predictions within ``n`` SAP points of the true score.

    The ONS Data Science Campus benchmark (2021) reports 93% within 15 SAP
    points as the UK national comparator. Including this metric lets us
    position our result against the only public UK-wide benchmark.
    """
    err = np.abs(np.asarray(score_pred) - np.asarray(score_true))
    return {f"within_{n}": float((err <= n).mean()) for n in n_values}


def qwk_by_confidence_quartile(y_true: pd.Series, y_pred: pd.Series,
                                 confidence: np.ndarray) -> list[dict]:
    """Validate a confidence proxy: does higher confidence => higher QWK?

    Partitions the test set into quartiles of ``confidence`` and reports QWK
    in each. If the proxy carries any signal at all, the top-quartile QWK
    should exceed the bottom-quartile QWK by a comfortable margin. If the
    QWKs are flat across quartiles, the proxy is uninformative.
    """
    qs = np.quantile(confidence, [0.0, 0.25, 0.5, 0.75, 1.0])
    out = []
    for k, (lo, hi) in enumerate(zip(qs[:-1], qs[1:])):
        if k == 3:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        if mask.sum() < 50:
            continue
        rep = evaluate_classifier(y_true[mask], y_pred[mask])
        out.append({
            "quartile": k + 1,
            "n": int(mask.sum()),
            "confidence_range": [float(lo), float(hi)],
            "qwk": rep.qwk,
            "accuracy": rep.accuracy,
        })
    return out


def bootstrap_per_class_f1_ci(y_true: pd.Series, y_pred: pd.Series,
                               n_resamples: int = 1000, seed: int = 42
                               ) -> dict[str, dict[str, float]]:
    """Per-class F1 bootstrap 95% CIs by resampling test rows with replacement.

    For minority bands (A, F, G with only tens to hundreds of test samples),
    a point estimate is misleading; the CI width tells the reader how much
    of the headline F1 is signal vs sampling noise.
    """
    rng = np.random.default_rng(seed)
    yt_arr = encode_rating(y_true) if y_true.dtype == object else np.asarray(y_true)
    yp_arr = y_pred.map(RATING_TO_INT).to_numpy() if (hasattr(y_pred, "dtype") and y_pred.dtype == object) else np.asarray(y_pred)
    n = len(yt_arr)
    labels = list(range(len(RATING_ORDER)))
    f1_samples = {b: [] for b in RATING_ORDER}
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt_b = yt_arr[idx]; yp_b = yp_arr[idx]
        per = f1_score(yt_b, yp_b, average=None, labels=labels, zero_division=0)
        for b, v in zip(RATING_ORDER, per):
            f1_samples[b].append(float(v))
    out = {}
    for b in RATING_ORDER:
        arr = np.array(f1_samples[b])
        out[b] = {
            "mean": float(arr.mean()),
            "ci_low_2.5": float(np.quantile(arr, 0.025)),
            "ci_high_97.5": float(np.quantile(arr, 0.975)),
            "ci_width": float(np.quantile(arr, 0.975) - np.quantile(arr, 0.025)),
        }
    return out


def reliability_curve(y_true_binary: np.ndarray, y_prob: np.ndarray,
                      n_bins: int = 10) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (mean predicted prob per bin, observed frequency per bin, Brier)."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    mean_pred = np.array([y_prob[idx == k].mean() if (idx == k).any() else np.nan for k in range(n_bins)])
    obs_freq = np.array([y_true_binary[idx == k].mean() if (idx == k).any() else np.nan for k in range(n_bins)])
    brier = float(brier_score_loss(y_true_binary, y_prob))
    return mean_pred, obs_freq, brier


def fairness_breakdown(y_true: pd.Series, y_pred: pd.Series,
                       segment: pd.Series, name: str,
                       min_support: int = 30) -> pd.DataFrame:
    """Per-segment macro-F1, balanced accuracy, QWK and support.

    Segments with fewer than ``min_support`` rows are dropped to avoid
    reporting unstable metrics on tiny groups.
    """
    seg = segment.dropna()
    out = []
    for seg_value, sub in seg.groupby(seg):
        idx = sub.index
        if len(idx) < min_support:
            continue
        rep = evaluate_classifier(y_true.loc[idx], y_pred.loc[idx])
        out.append({
            "segment": name,
            "value": str(seg_value),
            "n": int(len(idx)),
            "balanced_accuracy": rep.balanced_accuracy,
            "macro_f1": rep.macro_f1,
            "qwk": rep.qwk,
        })
    if not out:
        return pd.DataFrame(columns=["segment", "value", "n",
                                     "balanced_accuracy", "macro_f1", "qwk"])
    return pd.DataFrame(out).sort_values(["segment", "value"]).reset_index(drop=True)
