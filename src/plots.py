"""Publication-quality plotting helpers (matplotlib / seaborn).

Every function takes data + an axis and writes a single figure.
The notebook calls each, then saves at 300 dpi to ``reports/figures/``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import RATING_ORDER

sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)


def _save(fig, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_rating_distribution(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    counts = df["CURRENT_ENERGY_RATING"].value_counts().reindex(RATING_ORDER).fillna(0)
    ax.bar(counts.index, counts.values, color=sns.color_palette("RdYlGn_r", len(RATING_ORDER)))
    ax.set_xlabel("EPC band")
    ax.set_ylabel("Number of certificates")
    ax.set_title("Distribution of current energy rating, Oxford")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=8)
    _save(fig, out)


def fig_rating_by_age(df: pd.DataFrame, out: Path) -> None:
    age_order = [
        "England and Wales: before 1900", "England and Wales: 1900-1929",
        "England and Wales: 1930-1949", "England and Wales: 1950-1966",
        "England and Wales: 1967-1975", "England and Wales: 1976-1982",
        "England and Wales: 1983-1990", "England and Wales: 1991-1995",
        "England and Wales: 1996-2002", "England and Wales: 2003-2006",
        "England and Wales: 2007-2011", "England and Wales: 2012 onwards",
    ]
    sub = df.dropna(subset=["CONSTRUCTION_AGE_BAND", "CURRENT_ENERGY_RATING"]).copy()
    sub["CONSTRUCTION_AGE_BAND"] = pd.Categorical(
        sub["CONSTRUCTION_AGE_BAND"], categories=age_order, ordered=True)
    ct = (
        pd.crosstab(sub["CONSTRUCTION_AGE_BAND"], sub["CURRENT_ENERGY_RATING"], normalize="index")
        .reindex(columns=RATING_ORDER, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bottom = np.zeros(len(ct))
    palette = sns.color_palette("RdYlGn_r", len(RATING_ORDER))
    for i, band in enumerate(RATING_ORDER):
        ax.bar(range(len(ct)), ct[band].values, bottom=bottom, label=band, color=palette[i])
        bottom += ct[band].values
    ax.set_xticks(range(len(ct)))
    ax.set_xticklabels([s.replace("England and Wales: ", "") for s in ct.index],
                       rotation=30, ha="right")
    ax.set_ylabel("Share within age band")
    ax.set_title("Rating composition by construction age band, Oxford")
    ax.legend(title="Band", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    _save(fig, out)


def fig_efficiency_box_by_walls(df: pd.DataFrame, out: Path,
                                top_n: int = 8) -> None:
    sub = df.dropna(subset=["CURRENT_ENERGY_EFFICIENCY", "WALLS_DESCRIPTION"]).copy()
    keep = sub["WALLS_DESCRIPTION"].value_counts().nlargest(top_n).index
    sub = sub[sub["WALLS_DESCRIPTION"].isin(keep)]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    sns.boxplot(data=sub, y="WALLS_DESCRIPTION", x="CURRENT_ENERGY_EFFICIENCY",
                ax=ax, orient="h",
                order=sub.groupby("WALLS_DESCRIPTION")["CURRENT_ENERGY_EFFICIENCY"].median()
                          .sort_values().index)
    ax.set_title("SAP score by wall construction (top wall types in Oxford)")
    ax.set_xlabel("SAP score (CURRENT_ENERGY_EFFICIENCY)")
    ax.set_ylabel("")
    _save(fig, out)


def fig_corr_spearman(df: pd.DataFrame, out: Path) -> None:
    cols = [c for c in [
        "CURRENT_ENERGY_EFFICIENCY", "TOTAL_FLOOR_AREA", "NUMBER_HABITABLE_ROOMS",
        "NUMBER_HEATED_ROOMS", "MULTI_GLAZE_PROPORTION", "LOW_ENERGY_LIGHTING",
        "EXTENSION_COUNT", "FLOOR_HEIGHT", "CONSTRUCTION_AGE_NUM",
    ] if c in df.columns]
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    corr = sub.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax,
                cbar_kws={"label": "Spearman ρ"})
    ax.set_title("Spearman correlation among numeric features")
    _save(fig, out)


def fig_confusion(cm: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=RATING_ORDER, yticklabels=RATING_ORDER,
                cbar_kws={"label": "n"})
    ax.set_xlabel("Predicted band")
    ax.set_ylabel("True band")
    ax.set_title("Confusion matrix, hold-out test set")
    _save(fig, out)


def fig_perm_importance(importances: pd.Series, out: Path, top: int = 20) -> None:
    s = importances.sort_values(ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.barh(s.index, s.values, color="#3b6cb7")
    ax.set_xlabel("Permutation importance Δ (mean over 10 repeats)")
    ax.set_title(f"Top {top} features by permutation importance")
    _save(fig, out)


def fig_calibration(mean_pred: np.ndarray, obs_freq: np.ndarray,
                    brier: float, out: Path, label: str = "EPC band C",
                    second: tuple | None = None) -> None:
    """Reliability curve for one class.

    Optionally overlay an isotonic-recalibrated curve via ``second =
    (mean_pred_iso, obs_freq_iso, brier_iso)``.
    """
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(mean_pred, obs_freq, marker="o", label=f"Champion (Brier={brier:.3f})")
    if second is not None:
        m2, o2, b2 = second
        ax.plot(m2, o2, marker="s", color="#2C5F2D",
                label=f"+ isotonic (Brier={b2:.3f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Reliability curve, {label}")
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, out)


def fig_fairness(fair: pd.DataFrame, out: Path,
                 metric_col: str = "qwk") -> None:
    """Horizontal bar chart of per-segment metrics, separate panel per segment."""
    if fair.empty:
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.text(0.5, 0.5, "No fairness segments above the support floor.",
                ha="center", va="center", fontsize=10)
        ax.axis("off")
        _save(fig, out)
        return
    segs = fair["segment"].unique().tolist()
    fig, axes = plt.subplots(len(segs), 1, figsize=(8, 1.0 + 0.4 * len(fair)),
                             gridspec_kw={"height_ratios": [
                                 max(1, fair[fair["segment"] == s].shape[0]) for s in segs]})
    if len(segs) == 1:
        axes = [axes]
    for ax, seg in zip(axes, segs):
        sub = fair[fair["segment"] == seg].sort_values(metric_col)
        colors = ["#2C5F2D" if v >= 0.6 else "#C2503A" if v < 0.4 else "#3b6cb7"
                  for v in sub[metric_col]]
        ax.barh(sub["value"].astype(str), sub[metric_col], color=colors)
        ax.set_xlim(min(-0.1, sub[metric_col].min() - 0.05), 1.0)
        ax.axvline(0, color="grey", lw=0.5)
        ax.set_title(f"{seg}: {metric_col.upper()} per segment", loc="left", fontsize=11)
        for i, (v, n) in enumerate(zip(sub[metric_col].values, sub["n"].values)):
            ax.text(v + 0.01, i, f"{v:.2f}  (n={n:,})", va="center", fontsize=8)
    _save(fig, out)


def fig_top_recommendations(rec_summary: pd.DataFrame, out: Path,
                            top: int = 12) -> None:
    """Horizontal bar chart of the most-frequent recommendations.

    ``rec_summary`` is expected to have columns ``measure``, ``count``,
    ``cost_mid_mean`` (median indicative cost across the rows that quote one).
    """
    s = rec_summary.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(s) + 1.2))
    bars = ax.barh(s["measure"].astype(str), s["count"], color="#3b6cb7")
    ax.set_xlabel("Number of Oxford EPCs recommending this measure")
    ax.set_title(f"Top {top} retrofit measures recommended by the SAP engine, Oxford")
    for bar, v, cost in zip(bars, s["count"], s["cost_mid_mean"]):
        cost_lbl = f"≈ £{int(cost):,}" if pd.notna(cost) and cost > 0 else "n/a"
        ax.text(v + max(s["count"]) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{int(v):,}   {cost_lbl}", va="center", fontsize=8)
    ax.set_xlim(right=max(s["count"]) * 1.18)
    _save(fig, out)


def fig_classwise_metric(per_class: dict, out: Path, metric: str = "f1-score") -> None:
    rows = []
    for k, v in per_class.items():
        if k in ("accuracy", "macro avg", "weighted avg"):
            continue
        rows.append({"band": k, "metric": v.get(metric, float("nan")),
                     "support": v.get("support", 0)})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    palette = sns.color_palette("RdYlGn_r", len(df))
    bars = ax.bar(df["band"], df["metric"], color=palette)
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric)
    ax.set_xlabel("EPC band")
    ax.set_title(f"Per-band {metric} on hold-out test set")
    for bar, v, s in zip(bars, df["metric"], df["support"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.2f}\nn={int(s):,}", ha="center", va="bottom", fontsize=8)
    _save(fig, out)
