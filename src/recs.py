"""Analyse the ``recommendations.csv`` table.

The EPB Register publishes one row per recommended improvement measure
(hot-water cylinder insulation, low-energy lighting, internal/external
wall insulation, etc.) per certificate. We parse the indicative cost
ranges and cross-reference the most-frequent measures with the model's
SHAP findings.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .features import parse_indicative_cost


def filter_to_oxford(rec: pd.DataFrame, oxford_lmk_keys: Iterable[str]) -> pd.DataFrame:
    """Keep only recommendations attached to Oxford certificates."""
    s = pd.Series(list(oxford_lmk_keys), dtype="string")
    return rec[rec["LMK_KEY"].isin(s)].copy()


def summarise(rec: pd.DataFrame) -> pd.DataFrame:
    """Group by measure and return counts plus indicative-cost summaries."""
    if rec.empty:
        return pd.DataFrame(columns=[
            "measure", "count", "cost_low_median", "cost_high_median",
            "cost_mid_mean", "cost_mid_median",
        ])
    costs = parse_indicative_cost(rec["INDICATIVE_COST"])
    df = pd.concat([rec[["IMPROVEMENT_SUMMARY_TEXT"]].rename(
        columns={"IMPROVEMENT_SUMMARY_TEXT": "measure"}), costs], axis=1)
    g = df.groupby("measure", dropna=False)
    out = g.agg(
        count=("measure", "size"),
        cost_low_median=("COST_LOW", "median"),
        cost_high_median=("COST_HIGH", "median"),
        cost_mid_mean=("COST_MID", "mean"),
        cost_mid_median=("COST_MID", "median"),
    ).reset_index()
    out = out.sort_values("count", ascending=False).reset_index(drop=True)
    return out


def total_indicative_cost_per_property(rec: pd.DataFrame) -> pd.Series:
    """Sum of indicative-cost midpoints per ``LMK_KEY``."""
    if rec.empty:
        return pd.Series(dtype="float")
    costs = parse_indicative_cost(rec["INDICATIVE_COST"])
    df = pd.concat([rec[["LMK_KEY"]], costs], axis=1)
    return df.groupby("LMK_KEY")["COST_MID"].sum().rename("total_indicative_cost_mid")
