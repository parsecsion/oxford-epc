"""Generate the additional chart types from the assessment's visualisation
taxonomy that the core pipeline did not yet include:

  fig_sap_histogram.png   — histogram + KDE with shaded A-G band regions
  fig_sap_violin.png      — violin: SAP by property type (width-scaled, n labelled)
  fig_measure_wordcloud.png — word cloud of retrofit measures (text relationship)
  fig_band_donut.png      — donut: EPC band share (part-to-whole)
  fig_treemap.png         — treemap: property type x built form, coloured by mean SAP
  fig_stacked_area.png    — stacked area: lodgements per year by band (trend breakdown)
  fig_bubble.png          — bubble: floor area x SAP x CO2 x property type

These are descriptive views (the SAP score / CO2 are the *outcomes* being
described, not model inputs), so no leakage concern arises.

Design notes (professional visualisation practice applied at the 2nd pass):
  * Histogram shades the seven statutory band regions in RdYlGn rather than
    drawing bare, unlabelled cut-off lines — the cut-offs now mean something.
  * Violin uses width-scaling (density_norm="width") so groups of very
    different size are shape-comparable, and prints n under each tick.
  * Donut moves percentages into a legend so the tiny A/F/G slices no longer
    collide; only slices >= 3% keep an on-wedge label.
  * Treemap colour encodes a *metric* (mean SAP per cell) on a sequential
    scale, not a meaningless rainbow; tiny tiles drop their text.
  * Stacked area uses integer year ticks and ends at the last *complete*
    year so the partial current year is not shown as a false cliff.
  * Bubble colours by property type (a 4th, non-redundant variable rather
    than re-encoding the y-axis) and carries an explicit CO2 size legend;
    the CO2 subscript is rendered via mathtext so it is not a tofu glyph.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import squarify
from wordcloud import WordCloud

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_certificates, load_recommendations, build_clean_frame
from src.features import engineer

FIG = ROOT / "reports" / "figures"
RATING_ORDER = list("ABCDEFG")
# statutory SAP -> band cut-offs (lower bound, upper bound, band), A..G
BAND_BOUNDS = [(92, 100, "A"), (81, 92, "B"), (69, 81, "C"), (55, 69, "D"),
               (39, 55, "E"), (21, 39, "F"), (1, 21, "G")]
sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
plt.rcParams["axes.titleweight"] = "bold"


def _save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {name}")


def main() -> int:
    print("Loading & cleaning ...")
    eng = engineer(build_clean_frame(load_certificates("certificates.csv")))
    eng["SAP"] = pd.to_numeric(eng["CURRENT_ENERGY_EFFICIENCY"], errors="coerce")
    df = eng.dropna(subset=["SAP"])
    bands = pd.Categorical(df["CURRENT_ENERGY_RATING"], categories=RATING_ORDER, ordered=True)
    band_palette = sns.color_palette("RdYlGn_r", 7)
    band_color = dict(zip(RATING_ORDER, band_palette))

    # 1) histogram + KDE with shaded, labelled band regions --------------------
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    sns.histplot(df["SAP"], bins=40, kde=True, color="#33415c",
                 edgecolor="white", alpha=0.85, ax=ax)
    ymax = ax.get_ylim()[1]
    for lo, hi, lab in BAND_BOUNDS:
        ax.axvspan(lo, hi, color=band_color[lab], alpha=0.16, lw=0)
        ax.text((lo + hi) / 2, ymax * 0.97, lab, ha="center", va="top",
                fontsize=9, fontweight="bold", color="#333")
    mean_sap = df["SAP"].mean()
    ax.axvline(mean_sap, color="black", ls="--", lw=1.1)
    ax.text(mean_sap + 1, ymax * 0.62, f"mean = {mean_sap:.0f}",
            fontsize=8.5, rotation=90, va="center")
    ax.set_xlim(0, 105)
    ax.set_title("Distribution of SAP score with statutory EPC bands, Oxford")
    ax.set_xlabel("SAP score (CURRENT_ENERGY_EFFICIENCY)"); ax.set_ylabel("Number of dwellings")
    _save(fig, "fig_sap_histogram.png")

    # 2) violin: SAP by property type (width-scaled, n labelled) ---------------
    main_types = ["Flat", "Maisonette", "House", "Bungalow"]
    sub = df[df["PROPERTY_TYPE"].isin(main_types)].copy()
    order = sub.groupby("PROPERTY_TYPE")["SAP"].median().sort_values(ascending=False).index.tolist()
    counts = sub["PROPERTY_TYPE"].value_counts()
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    sns.violinplot(data=sub, x="PROPERTY_TYPE", y="SAP", order=order,
                   hue="PROPERTY_TYPE", legend=False, palette="viridis",
                   inner="quartile", density_norm="width", cut=0, ax=ax)
    ax.set_xticklabels([f"{t}\n(n = {counts[t]:,})" for t in order])
    ax.set_title("SAP-score distribution by property type (width-scaled violin)")
    ax.set_xlabel(""); ax.set_ylabel("SAP score"); ax.set_ylim(0, 105)
    _save(fig, "fig_sap_violin.png")

    # 3) word cloud of retrofit measures (text relationship) -------------------
    rec = load_recommendations("recommendations.csv")
    ox = set(eng["LMK_KEY"].dropna().astype(str))
    rec = rec[rec["LMK_KEY"].astype(str).isin(ox)]
    meas = rec["IMPROVEMENT_SUMMARY_TEXT"].dropna().astype(str).str.split(",").str[0].str.strip()
    freq = meas.value_counts().head(45).to_dict()  # cap so nothing is unreadably faint
    wc = WordCloud(width=1000, height=480, background_color="white",
                   colormap="viridis", prefer_horizontal=0.95,
                   collocations=False, min_font_size=9,
                   relative_scaling=0.5).generate_from_frequencies(freq)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    ax.set_title("Most-recommended retrofit measures (font size = frequency), Oxford")
    _save(fig, "fig_measure_wordcloud.png")

    # 4) donut: EPC band share, percentages in a legend ------------------------
    counts_b = pd.Series(bands).value_counts().reindex(RATING_ORDER).fillna(0)
    total = int(counts_b.sum())
    pct = counts_b / total * 100
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    wedges, _ = ax.pie(
        counts_b.values, colors=band_palette, startangle=90, counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor="white"))
    # on-wedge % only for slices large enough to read
    for w, p in zip(wedges, pct.values):
        if p >= 3:
            ang = np.deg2rad((w.theta1 + w.theta2) / 2)
            r = 0.79
            ax.text(r * np.cos(ang), r * np.sin(ang), f"{p:.1f}%",
                    ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(0, 0, f"{total:,}\ndwellings", ha="center", va="center",
            fontsize=12, fontweight="bold")
    ax.legend(wedges, [f"{b} — {p:.1f}%" for b, p in zip(RATING_ORDER, pct.values)],
              title="Band", loc="center left", bbox_to_anchor=(1.0, 0.5),
              frameon=False, fontsize=9)
    ax.set_title("EPC band share, Oxford")
    _save(fig, "fig_band_donut.png")

    # 5) treemap: type x built form, colour = mean SAP (a metric, not rainbow) --
    grp = (df.groupby(["PROPERTY_TYPE", "BUILT_FORM"])
             .agg(n=("SAP", "size"), mean_sap=("SAP", "mean"))
             .sort_values("n", ascending=False).head(14).reset_index())
    norm = mcolors.Normalize(vmin=grp["mean_sap"].min(), vmax=grp["mean_sap"].max())
    cmap = plt.get_cmap("RdYlGn")
    colors = [cmap(norm(v)) for v in grp["mean_sap"]]
    # only label tiles large enough (share of total area) to avoid overflow
    share = grp["n"] / grp["n"].sum()
    labels = [f"{r.PROPERTY_TYPE}\n{r.BUILT_FORM}\n{int(r.n):,}  (SAP {r.mean_sap:.0f})"
              if s >= 0.04 else "" for r, s in zip(grp.itertuples(), share)]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    squarify.plot(sizes=grp["n"].values, label=labels, ax=ax, pad=True,
                  color=colors, text_kwargs={"fontsize": 8})
    ax.axis("off")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.01); cb.set_label("Mean SAP score")
    ax.set_title("Housing stock by property type and built form, shaded by mean SAP, Oxford")
    _save(fig, "fig_treemap.png")

    # 6) stacked area: lodgements per year by band (to last complete year) -----
    yr = pd.to_datetime(df["INSPECTION_DATE"], errors="coerce").dt.year
    last_full = int(yr.max()) - 1  # drop the partial current year -> no false cliff
    tab = pd.crosstab(yr, bands).reindex(columns=RATING_ORDER, fill_value=0)
    tab = tab[(tab.index >= 2008) & (tab.index <= last_full)]
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    ax.stackplot(tab.index.astype(int), [tab[b].values for b in RATING_ORDER],
                 labels=RATING_ORDER, colors=band_palette, alpha=0.9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    ax.set_xlim(2008, last_full)
    ax.set_title(f"EPC lodgements per year, by band (2008 to {last_full}), Oxford")
    ax.set_xlabel("Inspection year"); ax.set_ylabel("Lodgements")
    ax.legend(title="Band", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    _save(fig, "fig_stacked_area.png")

    # 7) bubble: floor area x SAP x CO2(size) x property type(colour) ----------
    b = df.dropna(subset=["TOTAL_FLOOR_AREA", "CO2_EMISSIONS_CURRENT"]).copy()
    b["CO2"] = pd.to_numeric(b["CO2_EMISSIONS_CURRENT"], errors="coerce")
    b = b[(b["CO2"] > 0) & b["PROPERTY_TYPE"].isin(main_types)].dropna(subset=["CO2"])
    samp = b.sample(min(1200, len(b)), random_state=42)
    type_colors = dict(zip(main_types, sns.color_palette("Set2", len(main_types))))
    co2_max = samp["CO2"].max()
    # bubble AREA proportional to CO2 (matplotlib s is area in pt^2)
    area = (samp["CO2"] / co2_max * 260).clip(8)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for t in main_types:
        m = (samp["PROPERTY_TYPE"] == t).values
        ax.scatter(samp.loc[m, "TOTAL_FLOOR_AREA"], samp.loc[m, "SAP"],
                   s=area[m], c=[type_colors[t]], alpha=0.55,
                   edgecolors="white", linewidths=0.3, label=t)
    ax.set_xlabel("Total floor area (m$^2$)"); ax.set_ylabel("SAP score")
    ax.set_title("Floor area vs SAP, bubble size = CO$_2$ emissions, colour = property type")
    type_leg = ax.legend(title="Property type", loc="lower right", fontsize=8, framealpha=0.9)
    ax.add_artist(type_leg)
    # explicit CO2 size legend (reference bubbles, area-scaled to match)
    refs = list(np.percentile(samp["CO2"], [25, 50, 90]))
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
                      markersize=np.sqrt(v / co2_max * 260),
                      label=f"{v:.0f} t") for v in refs]
    ax.legend(handles=handles, title="CO$_2$ (size)", loc="upper left",
              fontsize=8, labelspacing=1.6, borderpad=1.0, framealpha=0.9)
    _save(fig, "fig_bubble.png")

    print("\nAll 7 extra charts written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
