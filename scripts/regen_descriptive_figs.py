"""Regenerate the §3.2 descriptive and §3.3 diagnostic figures + statistics
on the CURRENT cleaned data (post sentinel/FLOOR_LEVEL fix), so the figures
and the numbers quoted in the report are reproducible and consistent.

No model training — pure descriptive/diagnostic recomputation. Updates the
'diagnostic' and 'rating_distribution' sections of reports/metrics.json.
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_certificates, build_clean_frame
from src.features import engineer
from src import plots as P

FIG = ROOT / "reports" / "figures"
METRICS = ROOT / "reports" / "metrics.json"


def cramers_v(a, b):
    ct = pd.crosstab(a, b)
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.values.sum(); r, k = ct.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1)))) if min(r, k) > 1 else float("nan")


def main() -> int:
    print("Loading & cleaning ...")
    raw = load_certificates("certificates.csv")
    clean = build_clean_frame(raw)
    feat = engineer(clean)
    print(f"  clean {clean.shape}, engineered {feat.shape}")

    # --- §3.2 descriptive figures ---
    P.fig_rating_distribution(clean, FIG / "fig_rating_distribution.png")
    P.fig_rating_by_age(feat, FIG / "fig_rating_by_age.png")
    P.fig_efficiency_box_by_walls(feat, FIG / "fig_eff_by_walls.png")
    P.fig_corr_spearman(feat, FIG / "fig_corr_spearman.png")

    # property type x built form
    ct = pd.crosstab(clean["PROPERTY_TYPE"], clean["BUILT_FORM"])
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_title("Property type x built form - Oxford")
    fig.tight_layout(); fig.savefig(FIG / "fig_property_built_form.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    # missingness top-20
    miss = (clean.isna().mean().sort_values(ascending=False).head(20) * 100)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.barh(miss.index[::-1], miss.values[::-1], color="#c0732f")
    ax.set_xlabel("% missing in cleaned Oxford frame")
    ax.set_title("Top 20 columns by missingness")
    fig.tight_layout(); fig.savefig(FIG / "fig_missingness.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    # temporal volume (lodgements by inspection year)
    yr = pd.to_datetime(clean["INSPECTION_DATE"], errors="coerce").dt.year
    vc = yr.value_counts().sort_index()
    vc = vc[(vc.index >= 2008) & (vc.index <= 2025)]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.bar(vc.index.astype(int), vc.values, color="#3b6cb7")
    ax.set_xlabel("Inspection year"); ax.set_ylabel("EPC lodgements")
    ax.set_title("EPC lodgements by inspection year - Oxford")
    fig.tight_layout(); fig.savefig(FIG / "fig_temporal_volume.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print("  figures regenerated")

    # --- §3.3 diagnostic statistics ---
    sub = feat.dropna(subset=["CURRENT_ENERGY_EFFICIENCY", "CONSTRUCTION_AGE_NUM"])
    rho_age, p_age = stats.spearmanr(sub["CONSTRUCTION_AGE_NUM"], sub["CURRENT_ENERGY_EFFICIENCY"])
    rho_glaze, _ = stats.spearmanr(
        feat["MULTI_GLAZE_PROPORTION"].fillna(0),
        feat["CURRENT_ENERGY_EFFICIENCY"].fillna(feat["CURRENT_ENERGY_EFFICIENCY"].median()))
    cv = {
        "walls": cramers_v(feat["WALLS_DESCRIPTION"], feat["CURRENT_ENERGY_RATING"]),
        "age_band": cramers_v(feat["CONSTRUCTION_AGE_BAND"], feat["CURRENT_ENERGY_RATING"]),
        "main_fuel": cramers_v(feat["MAIN_FUEL"], feat["CURRENT_ENERGY_RATING"]),
        "built_form": cramers_v(feat["BUILT_FORM"], feat["CURRENT_ENERGY_RATING"]),
    }
    diag = {
        "spearman_age_sap": round(float(rho_age), 3),
        "spearman_age_sap_p": float(p_age),
        "spearman_glaze_sap": round(float(rho_glaze), 3),
        "cramers_v": {k: round(v, 3) for k, v in cv.items()},
    }
    rating_dist = clean["CURRENT_ENERGY_RATING"].value_counts().reindex(list("ABCDEFG")).fillna(0).astype(int).to_dict()
    print("\nDiagnostic statistics (current data):")
    print(f"  Spearman age-SAP   = {diag['spearman_age_sap']}  (p={p_age:.1e})")
    print(f"  Spearman glaze-SAP = {diag['spearman_glaze_sap']}")
    print(f"  Cramer's V         = {diag['cramers_v']}")
    print(f"  Rating distribution= {rating_dist}")

    # --- update metrics.json ---
    m = json.load(open(METRICS)) if METRICS.exists() else {}
    m["diagnostic"] = diag
    m["rating_distribution"] = rating_dist
    json.dump(m, open(METRICS, "w"), indent=2)
    print(f"\nUpdated {METRICS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
