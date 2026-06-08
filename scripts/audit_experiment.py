"""Phase-0 audit experiment: quantify the impact of the proposed rigor fixes
on the temporal holdout BEFORE committing to a full champion re-cascade.

Configs (all on the unified regression-to-band model, for clean isolation):
  A  baseline            current config (REPORT_TYPE kept, random early stop, no weight)
  B  group early-stop    early stopping decided on a UPRN-disjoint validation split
  C  drop REPORT_TYPE    methodology label removed
  D  sample-weight       weight = 1 / (certs per UPRN)  (down-weights repeat dwellings)
  E  tail-weighted loss  weight rises as SAP falls (prioritise low-efficiency tail)
  F  C+D combined        drop REPORT_TYPE + sample-weight

For each, report overall QWK, accuracy, and — crucially — RECALL on the
retrofit-relevant tail (bands E/F/G), which is what §4 actually needs.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import cohen_kappa_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import (load_certificates, filter_oxford, coerce_numeric,
                      validate_consistency, cap_outliers, drop_fully_missing,
                      group_temporal_split, to_model_matrix)
from src.features import engineer
from src.models import infer_column_groups, sap_score_to_band
from src import RATING_TO_INT, RATING_ORDER
SEED = 42


def build_nondedup(raw):
    return (raw.pipe(filter_oxford)
            .pipe(coerce_numeric, cols=["CURRENT_ENERGY_EFFICIENCY", "TOTAL_FLOOR_AREA",
                  "NUMBER_HABITABLE_ROOMS", "NUMBER_HEATED_ROOMS",
                  "CO2_EMISSIONS_CURRENT", "ENERGY_CONSUMPTION_CURRENT", "REPORT_TYPE"])
            .pipe(validate_consistency).pipe(cap_outliers)
            .pipe(drop_fully_missing, threshold=0.999)
            .dropna(subset=["CURRENT_ENERGY_RATING"])
            .loc[lambda d: d["CURRENT_ENERGY_RATING"].isin(list("ABCDEFG"))]
            .reset_index(drop=True))


def fill_cat(X):
    out = X.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(object).where(out[c].notna(), "__MISSING__")
    return out


def prep_for(Xtr):
    g = infer_column_groups(Xtr)
    return ColumnTransformer([
        ("num", "passthrough", g["numeric"]),
        ("lo", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), g["low_card"]),
        ("hi", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), g["high_card"]),
    ], remainder="drop", verbose_feature_names_out=False)


def reg(seed=SEED, early=True):
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.07, max_leaf_nodes=63, min_samples_leaf=20,
        l2_regularization=1.0, early_stopping=early, validation_fraction=0.15,
        n_iter_no_change=20, random_state=seed)


def evaluate(yte_band, score_pred):
    pred = pd.Series(sap_score_to_band(np.clip(score_pred, 1, 100)), index=yte_band.index)
    yt = yte_band.map(RATING_TO_INT).to_numpy(); yp = pred.map(RATING_TO_INT).to_numpy()
    qwk = cohen_kappa_score(yt, yp, labels=list(range(7)), weights="quadratic")
    acc = float((yte_band.values == pred.values).mean())
    # tail recall: among actual E/F/G, fraction predicted in {E,F,G}
    tail = yte_band.isin(["E", "F", "G"]).to_numpy()
    tail_pred = pred.isin(["E", "F", "G"]).to_numpy()
    tail_recall = float((tail & tail_pred).sum() / max(tail.sum(), 1))
    # per-band recall F and G
    rf = recall_score(yt, yp, labels=[RATING_TO_INT["F"]], average="macro", zero_division=0)
    rg = recall_score(yt, yp, labels=[RATING_TO_INT["G"]], average="macro", zero_division=0)
    return qwk, acc, tail_recall, rf, rg


def main():
    print("Loading ...")
    eng = engineer(build_nondedup(load_certificates("certificates.csv")))
    tr, te = group_temporal_split(eng)
    X, y, yreg = to_model_matrix(eng)
    uprn = eng["UPRN"].astype("string")
    Xtr_full = fill_cat(X.loc[tr]).reset_index(drop=True)
    Xte = fill_cat(X.loc[te]).reset_index(drop=True)
    yte_band = y.loc[te].reset_index(drop=True)
    yreg_tr = yreg.loc[tr].reset_index(drop=True).astype(float)
    uprn_tr = uprn.loc[tr].reset_index(drop=True)
    print(f"train {Xtr_full.shape}  test {Xte.shape}")

    def run(Xtr, sw=None, early=True, group_es=False):
        p = prep_for(Xtr)
        Xtr_t = p.fit_transform(Xtr); Xte_t = p.transform(Xte)
        if group_es:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
            core, val = next(gss.split(Xtr_t, yreg_tr, groups=uprn_tr.fillna("NA")))
            m = reg(early=False)
            m.fit(Xtr_t[core], yreg_tr.iloc[core],
                  sample_weight=None if sw is None else sw[core])
            best_i, best_e = 0, 1e9
            yv = yreg_tr.iloc[val].to_numpy()
            for i, yp in enumerate(m.staged_predict(Xte_t if False else Xtr_t[val])):
                e = np.sqrt(((yp - yv) ** 2).mean())
                if e < best_e: best_e, best_i = e, i + 1
            m2 = reg(early=False); m2.set_params(max_iter=max(best_i, 20))
            m2.fit(Xtr_t, yreg_tr, sample_weight=sw)
            return m2.predict(Xte_t)
        m = reg(early=early)
        m.fit(Xtr_t, yreg_tr, sample_weight=sw)
        return m.predict(Xte_t)

    # sample weights
    cnt = uprn_tr.map(uprn_tr.value_counts()).fillna(1).to_numpy()
    w_uprn = (1.0 / cnt).astype(float)
    # tail weight: inverse band frequency on training (rarer band -> higher weight)
    band_tr = y.loc[tr].reset_index(drop=True)
    freq = band_tr.map(band_tr.value_counts(normalize=True))
    w_tail = (1.0 / np.sqrt(freq.to_numpy())).astype(float)

    Xtr_noRT = Xtr_full.drop(columns=[c for c in ["REPORT_TYPE", "REPORT_TYPE_IS_MISSING"] if c in Xtr_full.columns])

    configs = [
        ("A baseline", lambda: run(Xtr_full)),
        ("B group-early-stop", lambda: run(Xtr_full, group_es=True)),
        ("C drop REPORT_TYPE", lambda: run(Xtr_noRT)),
        ("D sample-weight 1/uprn", lambda: run(Xtr_full, sw=w_uprn)),
        ("E tail-weighted loss", lambda: run(Xtr_full, sw=w_tail)),
        ("F C+D combined", lambda: run(Xtr_noRT, sw=w_uprn)),
    ]
    print(f"\n{'config':<24}{'QWK':>8}{'acc':>8}{'tailRec(EFG)':>14}{'recF':>7}{'recG':>7}")
    for name, fn in configs:
        qwk, acc, tr_, rf, rg = evaluate(yte_band, fn())
        print(f"{name:<24}{qwk:>8.4f}{acc:>8.4f}{tr_:>14.3f}{rf:>7.2f}{rg:>7.2f}")


if __name__ == "__main__":
    main()
