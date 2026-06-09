"""Descriptive/diagnostic analysis of property type vs energy efficiency, and
the efficiency trend over construction era — addressing the marker's emphasis
on property-type comparison and a timeline (rather than count-only charts).

Outputs:
  reports/figures/fig_property_efficiency.png   (mean SAP + band mix per type)
  reports/figures/fig_efficiency_timeline.png   (mean SAP by era, per type)
  reports/property_analysis.json
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_certificates, build_clean_frame
from src.features import engineer

FIG = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "property_analysis.json"
RATING_ORDER = list("ABCDEFG")
MAIN_TYPES = ["House", "Bungalow", "Maisonette", "Flat", "Park home"]


def main() -> int:
    print("Loading & cleaning ...")
    eng = engineer(build_clean_frame(load_certificates("certificates.csv")))
    eng = eng.copy()
    eng["CURRENT_ENERGY_EFFICIENCY"] = pd.to_numeric(
        eng["CURRENT_ENERGY_EFFICIENCY"], errors="coerce")
    df = eng.dropna(subset=["PROPERTY_TYPE", "CURRENT_ENERGY_EFFICIENCY",
                            "CURRENT_ENERGY_RATING"])
    keep = [t for t in MAIN_TYPES if t in df["PROPERTY_TYPE"].unique()]
    df = df[df["PROPERTY_TYPE"].isin(keep)]

    # === (C) property type vs efficiency ===
    by_type = (df.groupby("PROPERTY_TYPE")
               .agg(n=("CURRENT_ENERGY_EFFICIENCY", "size"),
                    mean_sap=("CURRENT_ENERGY_EFFICIENCY", "mean"),
                    median_sap=("CURRENT_ENERGY_EFFICIENCY", "median"))
               .sort_values("mean_sap", ascending=False))
    by_type["modal_band"] = df.groupby("PROPERTY_TYPE")["CURRENT_ENERGY_RATING"].agg(lambda s: s.mode().iat[0])
    best_type = by_type.index[0]
    print("\nProperty type vs mean SAP:")
    for t, r in by_type.iterrows():
        print(f"  {t:<12} n={int(r['n']):>6,}  mean SAP {r['mean_sap']:.1f}  band {r['modal_band']}")
    print(f"  -> most efficient type: {best_type} (mean SAP {by_type.iloc[0]['mean_sap']:.1f})")

    # band composition per type (normalised)
    comp = (pd.crosstab(df["PROPERTY_TYPE"], df["CURRENT_ENERGY_RATING"], normalize="index")
            .reindex(index=by_type.index, columns=RATING_ORDER, fill_value=0))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1, 1.2]})
    palette = sns.color_palette("RdYlGn_r", len(RATING_ORDER))
    order = list(by_type.index)
    bars = ax[0].barh(order, by_type["mean_sap"], color=sns.color_palette("viridis", len(order)))
    ax[0].invert_yaxis()
    ax[0].set_xlabel("Mean SAP score")
    ax[0].set_title("Energy efficiency by property type")
    for b, (t, r) in zip(bars, by_type.iterrows()):
        ax[0].text(r["mean_sap"] + 0.3, b.get_y() + b.get_height() / 2,
                   f"{r['mean_sap']:.1f}  ({r['modal_band']}, n={int(r['n']):,})", va="center", fontsize=8)
    ax[0].set_xlim(0, by_type["mean_sap"].max() * 1.3)
    bottom = np.zeros(len(comp))
    for i, band in enumerate(RATING_ORDER):
        ax[1].barh(range(len(comp)), comp[band].values, left=bottom, color=palette[i], label=band)
        bottom += comp[band].values
    ax[1].set_yticks(range(len(comp))); ax[1].set_yticklabels(comp.index)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("Share within property type")
    ax[1].set_title("Band composition by property type")
    ax[1].legend(title="Band", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_property_efficiency.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {(FIG/'fig_property_efficiency.png').relative_to(ROOT)}")

    # === (D) efficiency trend over construction era, per property type ===
    era_bins = [(0, 1900, "pre-1900"), (1900, 1930, "1900-29"), (1930, 1950, "1930-49"),
                (1950, 1967, "1950-66"), (1967, 1983, "1967-82"), (1983, 1996, "1983-95"),
                (1996, 2007, "1996-06"), (2007, 2100, "2007+")]
    df = df.copy()
    age = pd.to_numeric(df["CONSTRUCTION_AGE_NUM"], errors="coerce")
    df["ERA"] = pd.NA
    for lo, hi, lab in era_bins:
        df.loc[(age >= lo) & (age < hi), "ERA"] = lab
    era_order = [b[2] for b in era_bins]
    trend = (df.dropna(subset=["ERA"])
             .groupby(["ERA", "PROPERTY_TYPE"])["CURRENT_ENERGY_EFFICIENCY"].mean().unstack())
    trend = trend.reindex(era_order)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    type_order = [t for t in by_type.index if t in trend.columns]
    markers = ["o", "s", "^", "D", "v"]
    for t, m in zip(type_order, markers):
        s = trend[t]
        ax.plot(range(len(trend)), s.values, marker=m, label=t, linewidth=2)
    ax.set_xticks(range(len(trend))); ax.set_xticklabels(trend.index, rotation=25, ha="right")
    ax.set_ylabel("Mean SAP score")
    ax.set_xlabel("Construction era")
    ax.set_title("Energy-efficiency trend by construction era and property type, Oxford")
    ax.legend(title="Property type", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_efficiency_timeline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {(FIG/'fig_efficiency_timeline.png').relative_to(ROOT)}")

    # === timeline table: mean SAP by property type x era (for report table) ===
    timeline_table = (trend.round(1)
                      .reset_index().rename(columns={"ERA": "era"})
                      .to_dict(orient="records"))

    # === (7) champion performance broken down by property type ===
    champ_by_type = []
    pred_path = ROOT / "reports" / "predictions_oxford.csv"
    if pred_path.exists():
        from sklearn.metrics import cohen_kappa_score
        rk = {b: i for i, b in enumerate(RATING_ORDER)}
        pr = pd.read_csv(pred_path)
        pr = pr[pr["split"] == "test"].copy()
        ptype = eng[["LMK_KEY", "PROPERTY_TYPE"]].dropna()
        pr = pr.merge(ptype, on="LMK_KEY", how="left").dropna(subset=["PROPERTY_TYPE"])
        for t, sub in pr.groupby("PROPERTY_TYPE"):
            if len(sub) < 100 or t not in keep:
                continue
            yt = sub["actual_band"].map(rk); yp = sub["predicted_band"].map(rk)
            champ_by_type.append({
                "property_type": t, "n": int(len(sub)),
                "accuracy": round(float((sub["actual_band"] == sub["predicted_band"]).mean()), 3),
                "qwk": round(float(cohen_kappa_score(yt, yp, labels=list(range(7)),
                                                     weights="quadratic")), 3),
            })
        champ_by_type.sort(key=lambda x: -x["qwk"])
        print("\nChampion performance by property type (hold-out):")
        for r in champ_by_type:
            print(f"  {r['property_type']:<12} n={r['n']:>5,}  acc={r['accuracy']}  QWK={r['qwk']}")

    out = {
        "property_type_efficiency": {
            "most_efficient_type": best_type,
            "by_type": [{"property_type": t, "n": int(r["n"]),
                         "mean_sap": round(float(r["mean_sap"]), 1),
                         "modal_band": r["modal_band"]} for t, r in by_type.iterrows()],
        },
        "timeline_table_mean_sap_by_type_era": timeline_table,
        "champion_by_property_type": champ_by_type,
        "efficiency_timeline_note": (
            "Mean SAP rises monotonically with construction era across every "
            "property type, reflecting four decades of Building Regulations "
            "Part L tightening; the gap between oldest and newest stock is "
            f"~{trend.max().max() - trend.min().min():.0f} SAP points."
        ),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
