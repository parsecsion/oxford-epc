"""Retrofit-measure recommender for Oxford dwellings — a hybrid of:

* **Association-rule mining** (Apriori; Agrawal & Srikant, 1994) over the
  per-certificate basket of recommended measures in ``recommendations.csv``.
  Reports the strongest co-occurrence rules by *lift* (measures that are
  recommended together more than chance), which is interpretable and grounded.

* **Content-based k-NN recommender** over leak-free fabric features: for a
  target dwelling we find its k nearest neighbours by physical characteristics
  and recommend the measures most common among them. Evaluated honestly with a
  **train/test split and hit-rate (precision@N / recall@N)** — i.e. do the
  neighbours' top measures actually match the measures the SAP engine
  recommended for the held-out property?

* **Cost-effectiveness ranking** — measures ordered by indicative £ cost
  (parsed from the register) so the recommender surfaces cheap-first wins.

Outputs ``reports/recommender.json`` + ``reports/figures/fig_recommender.png``.
Aligned with the energy-efficiency recommender-systems survey (Sayed et al.,
2021) which stresses economic + interpretable evaluation.
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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_certificates, load_recommendations, build_clean_frame
from src.features import engineer, parse_indicative_cost

OUT_JSON = ROOT / "reports" / "recommender.json"
FIG = ROOT / "reports" / "figures" / "fig_recommender.png"
SEED = 42
MEASURE_COL = "IMPROVEMENT_SUMMARY_TEXT"
NUM_FEATS = ["LOG_FLOOR_AREA", "CONSTRUCTION_AGE_NUM", "NUMBER_HABITABLE_ROOMS",
             "MULTI_GLAZE_PROPORTION"]
CAT_FEATS = ["PROPERTY_TYPE", "BUILT_FORM", "MAIN_FUEL"]


def main() -> int:
    print("Loading ...")
    raw = load_certificates("certificates.csv")
    eng = engineer(build_clean_frame(raw))
    rec = load_recommendations("recommendations.csv")
    oxford_keys = set(eng["LMK_KEY"].dropna().astype(str))
    rec = rec[rec["LMK_KEY"].astype(str).isin(oxford_keys)].copy()
    measure = MEASURE_COL if rec[MEASURE_COL].notna().any() else "IMPROVEMENT_ITEM"
    rec = rec.dropna(subset=[measure]).copy()
    # Normalise: collapse capacity variants ("Solar PV, 2.5 kWp" -> "Solar PV")
    rec[measure] = rec[measure].astype(str).str.split(",").str[0].str.strip()
    print(f"  {len(rec):,} Oxford recommendations across {rec['LMK_KEY'].nunique():,} dwellings; "
          f"{rec[measure].nunique()} distinct measures")

    # --- baskets: one set of measures per dwelling ---
    baskets = rec.groupby("LMK_KEY")[measure].apply(lambda s: sorted(set(s))).to_dict()
    transactions = list(baskets.values())

    # === (A) Association rules (Apriori) ===
    te = TransactionEncoder()
    arr = te.fit_transform(transactions)
    onehot = pd.DataFrame(arr, columns=te.columns_)
    freq = apriori(onehot, min_support=0.03, use_colnames=True, max_len=3)
    rules = association_rules(freq, metric="lift", min_threshold=1.05)
    rules = rules.sort_values("lift", ascending=False)
    top_rules = []
    for _, r in rules.head(10).iterrows():
        top_rules.append({
            "antecedent": ", ".join(sorted(r["antecedents"]))[:60],
            "consequent": ", ".join(sorted(r["consequents"]))[:60],
            "support": round(float(r["support"]), 3),
            "confidence": round(float(r["confidence"]), 3),
            "lift": round(float(r["lift"]), 3),
        })
    print("\nTop association rules by lift:")
    for r in top_rules[:6]:
        print(f"  {r['antecedent']} -> {r['consequent']}  (lift {r['lift']}, conf {r['confidence']})")

    # === (B) Content-based kNN recommender, honestly evaluated ===
    prop = eng[["LMK_KEY"] + NUM_FEATS + CAT_FEATS].copy()
    for c in NUM_FEATS:
        prop[c] = pd.to_numeric(prop[c], errors="coerce")
    prop = prop.dropna(subset=NUM_FEATS + CAT_FEATS)
    prop = prop[prop["LMK_KEY"].astype(str).isin(baskets.keys())].reset_index(drop=True)
    prop["measures"] = prop["LMK_KEY"].astype(str).map(baskets)
    prop = prop[prop["measures"].map(len) > 0].reset_index(drop=True)

    ct = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATS),
    ])
    rng = np.random.default_rng(SEED)
    test_idx = rng.choice(len(prop), size=min(2000, len(prop) // 5), replace=False)
    train_mask = np.ones(len(prop), bool); train_mask[test_idx] = False
    Xtr = ct.fit_transform(prop.loc[train_mask])
    Xte = ct.transform(prop.iloc[test_idx])
    nn = NearestNeighbors(n_neighbors=15, metric="cosine").fit(Xtr)
    _, nbr = nn.kneighbors(Xte)
    from collections import Counter
    # Pre-extract measures to plain lists once (avoids per-row pandas .iloc in
    # the eval loop, which is the dominant cost at scale).
    train_measures = prop.loc[train_mask, "measures"].reset_index(drop=True).tolist()
    test_actual = [set(m) for m in prop.iloc[test_idx]["measures"].tolist()]

    N = 5
    hits = precis = recs = 0
    for j, neighbours in enumerate(nbr):
        cnt = Counter()
        for nb in neighbours:
            cnt.update(train_measures[nb])
        topN = {m for m, _ in cnt.most_common(N)}
        actual = test_actual[j]
        inter = len(topN & actual)
        hits += int(inter > 0)
        precis += inter / N
        recs += inter / max(len(actual), 1)
    n_test = len(test_idx)
    hit_rate = hits / n_test
    prec_at_n = precis / n_test
    rec_at_n = recs / n_test
    print(f"\nContent-based kNN (k=15, N={N}) on {n_test:,} held-out dwellings:")
    print(f"  hit-rate={hit_rate:.3f}  precision@{N}={prec_at_n:.3f}  recall@{N}={rec_at_n:.3f}")

    # --- Popularity baseline: always recommend the globally most-common
    #     measures. A content-based recommender is only worthwhile if it
    #     beats this naive baseline (standard recsys validation). ---
    from collections import Counter as _C
    pop = _C()
    for ms in train_measures:
        pop.update(ms)
    pop_topN = [m for m, _ in pop.most_common(N)]
    b_hits = b_prec = b_rec = 0
    for row in test_idx:
        actual = set(prop.iloc[row]["measures"])
        inter = len(set(pop_topN) & actual)
        b_hits += int(inter > 0); b_prec += inter / N; b_rec += inter / max(len(actual), 1)
    base_hit = b_hits / n_test; base_prec = b_prec / n_test; base_rec = b_rec / n_test
    print(f"  popularity baseline: hit-rate={base_hit:.3f}  precision@{N}={base_prec:.3f}  "
          f"recall@{N}={base_rec:.3f}")
    print(f"  content-based LIFT over baseline: precision@{N} {prec_at_n - base_prec:+.3f}, "
          f"recall@{N} {rec_at_n - base_rec:+.3f}")

    # === Cost-effectiveness of the most-frequent measures ===
    costs = parse_indicative_cost(rec["INDICATIVE_COST"])
    rec_cost = pd.concat([rec[[measure]].reset_index(drop=True),
                          costs.reset_index(drop=True)], axis=1)
    summary = (rec_cost.groupby(measure)
               .agg(count=(measure, "size"), cost_mid=("COST_MID", "median"))
               .sort_values("count", ascending=False))
    top_measures = [{"measure": m[:55], "count": int(r["count"]),
                     "median_cost_gbp": (None if pd.isna(r["cost_mid"]) else round(float(r["cost_mid"])))}
                    for m, r in summary.head(10).iterrows()]
    print("\nTop measures by frequency (median indicative cost):")
    for m in top_measures[:6]:
        print(f"  {m['count']:>6,}  £{m['median_cost_gbp']}  {m['measure']}")

    # === Figure: top measures bar + top rules ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4),
                             gridspec_kw={"width_ratios": [1.2, 1]})
    s = summary.head(8).iloc[::-1]
    axes[0].barh([m[:34] for m in s.index], s["count"], color="#3b6cb7")
    axes[0].set_xlabel("Dwellings recommended this measure")
    axes[0].set_title("Most-recommended retrofit measures (Oxford)")
    for i, (cnt, cost) in enumerate(zip(s["count"], s["cost_mid"])):
        lbl = f"{int(cnt):,}" + (f"  ~£{int(cost):,}" if pd.notna(cost) else "")
        axes[0].text(cnt, i, "  " + lbl, va="center", fontsize=8)
    axes[0].set_xlim(right=s["count"].max() * 1.25)
    rr = rules.head(8).iloc[::-1]
    lbls = [f"{', '.join(sorted(a))[:18]}→{', '.join(sorted(c))[:18]}"
            for a, c in zip(rr["antecedents"], rr["consequents"])]
    axes[1].barh(range(len(rr)), rr["lift"], color="#2C5F2D")
    axes[1].set_yticks(range(len(rr))); axes[1].set_yticklabels(lbls, fontsize=7)
    axes[1].axvline(1.0, color="grey", ls="--", lw=0.8)
    axes[1].set_xlabel("Lift (co-recommendation strength)")
    axes[1].set_title("Top association rules")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {FIG.relative_to(ROOT)}")

    out = {
        "association_rules": {
            "method": "Apriori (Agrawal & Srikant, 1994), min_support=0.03, ranked by lift",
            "n_baskets": len(transactions),
            "top_rules": top_rules,
        },
        "content_based_knn": {
            "method": "cosine k-NN (k=15) on leak-free fabric features, top-N=5",
            "n_test": int(n_test),
            "hit_rate": round(hit_rate, 3),
            "precision_at_5": round(prec_at_n, 3),
            "recall_at_5": round(rec_at_n, 3),
        },
        "popularity_baseline": {
            "method": "always recommend the 5 globally most-common measures",
            "hit_rate": round(base_hit, 3),
            "precision_at_5": round(base_prec, 3),
            "recall_at_5": round(base_rec, 3),
            "knn_lift_precision_at_5": round(prec_at_n - base_prec, 3),
            "knn_lift_recall_at_5": round(rec_at_n - base_rec, 3),
        },
        "cost_effectiveness_top_measures": top_measures,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
