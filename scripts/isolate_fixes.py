"""Isolate which rigor fix (M1 weights / M2 drop-REPORT_TYPE / C3 train-fold
missingness) drives the -0.013 QWK regression, on the actual HYBRID champion.
Early-stopping is excluded (proven not the cause: all variants train ~400 it).

Prints overall QWK + band-F/G recall for each config so we adopt only the
net-positive subset.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import cohen_kappa_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import (load_certificates, filter_oxford, coerce_numeric,
                      validate_consistency, cap_outliers, drop_fully_missing,
                      group_temporal_split, to_model_matrix)
from src.features import engineer
from src.models import (infer_column_groups, regression_to_band_estimator,
                        SapStratifiedRegressor)
from src import RATING_TO_INT
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


def mk(Xtr, seed=SEED):
    g = infer_column_groups(Xtr)
    prep = ColumnTransformer([
        ("num", "passthrough", g["numeric"]),
        ("lo", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), g["low_card"]),
        ("hi", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), g["high_card"]),
    ], remainder="drop", verbose_feature_names_out=False)
    return Pipeline([("prep", prep), ("reg", regression_to_band_estimator(seed=seed))])


def main():
    raw = load_certificates("certificates.csv")
    eng = engineer(build_nondedup(raw))
    tr, te = group_temporal_split(eng)
    rt_tr = eng.loc[tr, "REPORT_TYPE"].astype("Int64").reset_index(drop=True)
    rt_te = eng.loc[te, "REPORT_TYPE"].astype("Int64").reset_index(drop=True)
    uprn_tr = eng.loc[tr, "UPRN"].reset_index(drop=True)
    w = (1.0 / uprn_tr.map(uprn_tr.value_counts()).fillna(1.0)).to_numpy()
    yte = eng.loc[te, "CURRENT_ENERGY_RATING"].reset_index(drop=True)

    def run(drop_meth, train_miss, use_w):
        X, _, yreg = to_model_matrix(eng, drop_methodology=drop_meth,
                                     missingness_ref=(tr if train_miss else None))
        Xtr = fill_cat(X.loc[tr]).reset_index(drop=True)
        Xte = fill_cat(X.loc[te]).reset_index(drop=True)
        yreg_tr = yreg.loc[tr].reset_index(drop=True)
        ch = SapStratifiedRegressor(seed=SEED).fit(
            mk(Xtr), mk(Xtr.loc[(rt_tr == 101).values]), Xtr, yreg_tr, rt_tr,
            sample_weight=(w if use_w else None))
        pred = pd.Series(ch.predict_band(Xte, rt_te), index=yte.index)
        yt = yte.map(RATING_TO_INT).to_numpy(); yp = pred.map(RATING_TO_INT).to_numpy()
        qwk = cohen_kappa_score(yt, yp, labels=list(range(7)), weights="quadratic")
        rf = recall_score(yt, yp, labels=[RATING_TO_INT["F"]], average="macro", zero_division=0)
        rg = recall_score(yt, yp, labels=[RATING_TO_INT["G"]], average="macro", zero_division=0)
        return qwk, rf, rg

    configs = [
        ("A old baseline (none)", dict(drop_meth=False, train_miss=False, use_w=False)),
        ("B +M2 drop REPORT_TYPE", dict(drop_meth=True, train_miss=False, use_w=False)),
        ("C +M1 weights", dict(drop_meth=False, train_miss=False, use_w=True)),
        ("D +C3 train-missingness", dict(drop_meth=False, train_miss=True, use_w=False)),
        ("E M1+M2", dict(drop_meth=True, train_miss=False, use_w=True)),
        ("F M1+M2+C3 (all)", dict(drop_meth=True, train_miss=True, use_w=True)),
    ]
    print(f"\n{'config':<28}{'QWK':>8}{'recF':>7}{'recG':>7}")
    for name, kw in configs:
        q, rf, rg = run(**kw)
        print(f"{name:<28}{q:>8.4f}{rf:>7.2f}{rg:>7.2f}")


if __name__ == "__main__":
    main()
