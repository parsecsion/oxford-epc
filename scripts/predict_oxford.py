"""Apply the frozen champion to the full Oxford EPC corpus.

Produces ``reports/predictions_oxford.csv`` with one row per certificate.
Each row carries:

* ``LMK_KEY``               — the certificate's lodgement key (unique).
* ``UPRN``                  — Unique Property Reference Number, where present.
* ``INSPECTION_DATE``       — date of the assessment.
* ``REPORT_TYPE``           — 100 (RdSAP) or 101 (SAP).
* ``actual_band``           — recorded ``CURRENT_ENERGY_RATING`` (the truth
                              we trained against; included so the file is
                              self-contained for QA without having to join
                              back to the raw data).
* ``predicted_sap_score``   — continuous model output in [0, 100].
* ``predicted_band``        — DLUHC-thresholded band (A..G).
* ``band_correct``          — predicted_band == actual_band (bool).
* ``abs_score_error``       — |predicted_sap_score - CURRENT_ENERGY_EFFICIENCY|.
* ``confidence_proxy``      — distance from the nearest band threshold,
                              normalised to [0, 1]. Low values mean the
                              prediction is on a band boundary (uncertain);
                              high values mean it sits comfortably inside
                              a band. Useful for downstream prioritisation
                              (e.g. "audit the bottom-decile-confidence Fs
                              first" for retrofit targeting).
* ``split``                 — train | test, so the file can be filtered
                              to genuinely out-of-time predictions.

Loads the pickle directly and validates its sha256 against the value
recorded in ``reports/champion_artefact.json`` — protects against silent
artefact replacement.
"""
from __future__ import annotations
import hashlib
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (load_certificates, filter_oxford, coerce_numeric,
                      validate_consistency, cap_outliers, drop_fully_missing,
                      group_temporal_split, to_model_matrix)
from src.features import engineer
from src.models import sap_score_to_band, SAP_BAND_THRESHOLDS  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
PICKLE_PATH = ROOT / "artefacts" / "champion.pkl"
ARTEFACT_JSON = ROOT / "reports" / "champion_artefact.json"
OUT_CSV = ROOT / "reports" / "predictions_oxford.csv"
RATING_ORDER = list("ABCDEFG")

# Band-boundary SAP scores (lower bound of each band). Used for the
# confidence proxy: distance to the nearest boundary, normalised by the
# half-width of the band that contains the prediction.
BAND_LOWER = {"A": 92, "B": 81, "C": 69, "D": 55, "E": 39, "F": 21, "G": 0}
BAND_UPPER = {"A": 100, "B": 91, "C": 80, "D": 68, "E": 54, "F": 38, "G": 20}


def build_nondedup(raw):
    return (raw.pipe(filter_oxford)
                .pipe(coerce_numeric, cols=[
                    "CURRENT_ENERGY_EFFICIENCY", "TOTAL_FLOOR_AREA",
                    "NUMBER_HABITABLE_ROOMS", "NUMBER_HEATED_ROOMS",
                    "CO2_EMISSIONS_CURRENT", "ENERGY_CONSUMPTION_CURRENT",
                    "REPORT_TYPE",
                ])
                .pipe(validate_consistency)
                .pipe(cap_outliers)
                .pipe(drop_fully_missing, threshold=0.999)
                .dropna(subset=["CURRENT_ENERGY_RATING"])
                .loc[lambda d: d["CURRENT_ENERGY_RATING"].isin(RATING_ORDER)]
                .reset_index(drop=True))


def fill_cat(X):
    out = X.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(object).where(out[c].notna(), "__MISSING__")
    return out


def sha256_file(p: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def confidence_proxy(score: np.ndarray) -> np.ndarray:
    """Distance from nearest band boundary, normalised to [0, 1].

    For each predicted SAP score we find the band it falls into, compute
    the distance to the nearest band boundary, and divide by the half-width
    of that band. Predictions deep inside a band score close to 1.0;
    predictions on a boundary score close to 0.0.
    """
    score = np.asarray(score, dtype=float)
    out = np.zeros_like(score)
    for band, lo in BAND_LOWER.items():
        up = BAND_UPPER[band]
        mask = (score >= lo) & (score <= up + 0.999)
        if not mask.any():
            continue
        mid = (lo + up) / 2.0
        half = max(1.0, (up - lo) / 2.0)
        out[mask] = 1.0 - np.abs(score[mask] - mid) / half
    return np.clip(out, 0.0, 1.0)


def main() -> int:
    # --- Load and verify the frozen champion ---
    if not PICKLE_PATH.exists():
        print(f"ERROR: {PICKLE_PATH} does not exist. "
              "Run scripts/freeze_champion.py first.", file=sys.stderr)
        return 2
    if not ARTEFACT_JSON.exists():
        print(f"ERROR: {ARTEFACT_JSON} does not exist.", file=sys.stderr)
        return 2
    recorded = json.load(open(ARTEFACT_JSON))
    actual_sha = sha256_file(PICKLE_PATH)
    if actual_sha != recorded["pickle_sha256"]:
        print(f"ERROR: champion.pkl sha256 mismatch.\n"
              f"  recorded: {recorded['pickle_sha256']}\n"
              f"  actual  : {actual_sha}\n"
              "Re-run scripts/freeze_champion.py.", file=sys.stderr)
        return 2
    print(f"Loaded champion ({len(recorded['feature_schema'])} features, "
          f"sha256 ok)")
    champion = pickle.load(open(PICKLE_PATH, "rb"))

    # --- Rebuild the same data frame the champion was trained on ---
    print("Loading & cleaning the full Oxford corpus ...")
    raw = load_certificates("certificates.csv")
    eng = engineer(build_nondedup(raw))
    tr, te = group_temporal_split(eng)
    # Same matrix construction as freeze_champion so the schema matches
    # (train-fold missingness set; REPORT_TYPE retained).
    X, y, y_reg = to_model_matrix(eng, missingness_ref=tr)
    Xfull = fill_cat(X).reset_index(drop=True)
    eng_idx = eng.reset_index(drop=True)
    rt_full = eng_idx["REPORT_TYPE"].astype("Int64")
    print(f"Full corpus: {Xfull.shape[0]} certificates, "
          f"{Xfull.shape[1]} features")

    # --- Schema validation: every recorded column must be present ---
    expected_cols = [s["name"] for s in recorded["feature_schema"]]
    missing = [c for c in expected_cols if c not in Xfull.columns]
    extra = [c for c in Xfull.columns if c not in expected_cols]
    if missing:
        print(f"ERROR: feature schema mismatch -- {len(missing)} expected "
              f"columns are missing. First 10: {missing[:10]}",
              file=sys.stderr)
        return 3
    if extra:
        # Extra columns are tolerable; we'll drop them so the pipeline sees
        # exactly the schema it was fit on.
        print(f"  (note: {len(extra)} extra columns in current frame -- "
              f"dropping for inference. First 5: {extra[:5]})")
        Xfull = Xfull[expected_cols]
    else:
        Xfull = Xfull[expected_cols]

    # --- Inference ---
    print("Predicting ...")
    pred_score = champion.predict_score(Xfull, rt_full)
    pred_band = sap_score_to_band(pred_score)

    # --- Build output frame ---
    split_label = pd.Series("train", index=eng_idx.index)
    split_label.loc[te] = "test"  # note: indices align because we reset_index above
    # Actually `te` is a position-based index into the original eng dataframe;
    # since we reset_index after `engineer(build_nondedup(raw))`, the indices
    # are the same 0..N-1 row positions, so this is correct.

    actual_band = y.reset_index(drop=True)
    actual_score = y_reg.reset_index(drop=True)

    out = pd.DataFrame({
        "LMK_KEY": eng_idx.get("LMK_KEY", pd.Series(["" for _ in range(len(eng_idx))])),
        "UPRN": eng_idx["UPRN"].astype("string"),
        "INSPECTION_DATE": eng_idx["INSPECTION_DATE"],
        "REPORT_TYPE": rt_full.astype("Int64"),
        "actual_band": actual_band,
        "actual_sap_score": actual_score.astype(float),
        "predicted_sap_score": np.round(pred_score, 2),
        "predicted_band": pred_band,
        "band_correct": (pred_band == actual_band.values),
        "abs_score_error": np.round(np.abs(pred_score - actual_score.values), 2),
        "confidence_proxy": np.round(confidence_proxy(pred_score), 3),
        "split": split_label.values,
    })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV.relative_to(ROOT)} "
          f"({len(out):,} rows, {out['UPRN'].notna().sum():,} with UPRN)")

    # --- Sanity report ---
    test_only = out[out["split"] == "test"]
    print(f"\nHoldout (split=='test') summary:")
    print(f"  n = {len(test_only):,}")
    print(f"  band_correct rate    = {test_only['band_correct'].mean():.4f}")
    print(f"  median |score error| = {test_only['abs_score_error'].median():.2f}")
    print(f"  mean confidence      = {test_only['confidence_proxy'].mean():.3f}")
    print(f"\nPredicted band distribution:")
    for band, n in out["predicted_band"].value_counts().sort_index().items():
        print(f"  {band}: {n:>6,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
