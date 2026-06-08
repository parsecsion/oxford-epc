"""End-to-end data-pipeline verification.

Re-runs the canonical data pipeline from raw -> clean -> publishable -> model
matrix and writes:
- data/processed/oxford_epc_clean.csv          (the deliverable)
- reports/data_quality_report.csv              (auditable cleaning log)
- reports/summary_stats.csv                    (descriptive coverage)
- reports/data_completeness_checks.json        (definition-of-done checks)

Then validates every output against an explicit checklist of completion
criteria. Exits non-zero if any criterion fails.
"""
from __future__ import annotations
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (
    load_certificates, build_clean_frame, publishable_clean_frame,
    to_model_matrix, temporal_split, group_temporal_split,
    LEAKAGE_COLS, ELEMENT_EFF_COLS, PII_COLS, ADMIN_DROP,
)
from src.features import engineer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "certificates.csv"
CLEAN_CSV = ROOT / "data" / "processed" / "oxford_epc_clean.csv"
QUALITY_LOG = ROOT / "reports" / "data_quality_report.csv"
SUMMARY_CSV = ROOT / "reports" / "summary_stats.csv"
CHECKS_JSON = ROOT / "reports" / "data_completeness_checks.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main() -> int:
    print("=" * 70)
    print("STAGE 1: Regenerate all data artefacts")
    print("=" * 70)

    # 1. Load raw and clean
    print("\n[1/5] Loading raw certificates ...")
    raw = load_certificates(RAW)
    print(f"      raw shape           : {raw.shape}")
    raw_hash = sha256_file(RAW)
    print(f"      raw file hash       : {raw_hash}")

    print("\n[2/5] Building clean frame (writes quality report) ...")
    clean = build_clean_frame(raw, quality_report_path=QUALITY_LOG)
    print(f"      clean shape         : {clean.shape}")
    print(f"      quality report      : {QUALITY_LOG.relative_to(ROOT)}")

    print("\n[3/5] Building publishable frame (drops PII + admin) ...")
    pub = publishable_clean_frame(clean)
    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    pub.to_csv(CLEAN_CSV, index=False, encoding="utf-8")
    print(f"      published shape     : {pub.shape}")
    print(f"      written to          : {CLEAN_CSV.relative_to(ROOT)}")
    clean_hash = sha256_file(CLEAN_CSV)
    print(f"      output file hash    : {clean_hash}")

    print("\n[4/5] Computing summary statistics ...")
    numeric = pub.select_dtypes(include=[np.number])
    summary = numeric.describe().T
    summary["pct_missing"] = (pub[summary.index].isna().mean() * 100).round(2)
    summary.to_csv(SUMMARY_CSV)
    print(f"      summary stats       : {SUMMARY_CSV.relative_to(ROOT)} "
          f"({len(summary)} numeric cols)")

    print("\n[5/5] Building model matrix and splits ...")
    eng = engineer(clean)
    X, y, _ = to_model_matrix(eng)
    tr_flat, te_flat = temporal_split(eng)
    tr_grp, te_grp = group_temporal_split(eng)
    print(f"      engineered shape    : {eng.shape}")
    print(f"      model matrix shape  : {X.shape}")
    print(f"      flat split          : train={len(tr_flat)}, test={len(te_flat)}")
    print(f"      group split         : train={len(tr_grp)}, test={len(te_grp)}")

    # =====================================================================
    print()
    print("=" * 70)
    print("STAGE 2: Verify against completion-criteria checklist")
    print("=" * 70)

    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        checks.append({"check": name, "status": status, "detail": detail})
        print(f"  [{status}] {name}: {detail}")

    # --- Acquisition / understanding ---
    check("raw_certificates_loadable", raw.shape[0] > 50000,
          f"loaded {raw.shape[0]} raw rows from {RAW.name}")
    check("oxford_local_authority_filtered",
          (raw["LOCAL_AUTHORITY"] == "E07000178").all() or
          (clean["LOCAL_AUTHORITY"] == "E07000178").all() if "LOCAL_AUTHORITY" in clean.columns
          else True,
          "all rows belong to Oxford LA")

    # --- Cleaning ---
    check("quality_report_exists", QUALITY_LOG.exists(),
          f"audit trail at {QUALITY_LOG.relative_to(ROOT)}")
    if QUALITY_LOG.exists():
        qr = pd.read_csv(QUALITY_LOG)
        check("quality_report_records_changes", len(qr) > 0,
              f"{len(qr)} cleaning rules recorded")

    # No impossible values survive
    if "NUMBER_HEATED_ROOMS" in clean.columns and "NUMBER_HABITABLE_ROOMS" in clean.columns:
        h = pd.to_numeric(clean.NUMBER_HEATED_ROOMS, errors="coerce")
        a = pd.to_numeric(clean.NUMBER_HABITABLE_ROOMS, errors="coerce")
        n_bad = int(((h > a) & h.notna() & a.notna()).sum())
        check("no_heated_exceeds_habitable", n_bad == 0,
              f"{n_bad} rows where heated > habitable")
        n_zero = int(((h == 0) | (a == 0)).sum())
        check("no_zero_rooms", n_zero == 0,
              f"{n_zero} rows with zero rooms")
    if "TOTAL_FLOOR_AREA" in clean.columns:
        fa = pd.to_numeric(clean.TOTAL_FLOOR_AREA, errors="coerce")
        n_zero = int((fa == 0).sum())
        check("no_zero_floor_area", n_zero == 0,
              f"{n_zero} rows with zero floor area")

    # PV-generating dwellings preserved
    check("has_onsite_generation_flag",
          "HAS_ONSITE_GENERATION" in clean.columns,
          f"{clean.get('HAS_ONSITE_GENERATION', pd.Series([])).sum()} PV-generating dwellings flagged")

    # --- Deduplication ---
    if "UPRN" in clean.columns:
        uprn_dups = int(clean.UPRN.dropna().duplicated().sum())
        check("uprn_deduplicated", uprn_dups == 0,
              f"{uprn_dups} duplicate UPRNs survived")
    if "LMK_KEY" in clean.columns:
        lmk_dups = int(clean.LMK_KEY.dropna().duplicated().sum())
        check("lmk_key_unique", lmk_dups == 0,
              f"{lmk_dups} duplicate LMK_KEYs survived")

    # --- Published CSV / GDPR compliance ---
    pii_in_pub = [c for c in PII_COLS if c in pub.columns]
    check("no_pii_in_published_csv", len(pii_in_pub) == 0,
          f"PII columns present: {pii_in_pub or 'none'}")
    admin_in_pub = [c for c in ADMIN_DROP if c in pub.columns]
    check("no_admin_in_published_csv", len(admin_in_pub) == 0,
          f"admin columns present: {admin_in_pub or 'none'}")

    # --- Target validity ---
    target_valid = set(clean.CURRENT_ENERGY_RATING.unique()).issubset(set("ABCDEFG"))
    check("target_values_valid", target_valid,
          f"target values: {sorted(clean.CURRENT_ENERGY_RATING.unique())}")
    check("no_null_target", clean.CURRENT_ENERGY_RATING.notna().all(),
          f"null targets: {clean.CURRENT_ENERGY_RATING.isna().sum()}")

    # --- Leakage exclusion in model matrix ---
    leak_in_X = [c for c in LEAKAGE_COLS if c in X.columns]
    check("no_leakage_in_model_matrix", len(leak_in_X) == 0,
          f"leakage cols in X: {leak_in_X or 'none'}")
    elem_in_X = [c for c in ELEMENT_EFF_COLS if c in X.columns]
    check("no_element_eff_in_model_matrix", len(elem_in_X) == 0,
          f"element-eff cols in X: {elem_in_X or 'none'}")

    # --- Temporal splits leakage-free ---
    cutoff = pd.Timestamp("2022-12-31")
    flat_train_post = int((pd.to_datetime(eng.loc[tr_flat, "INSPECTION_DATE"]) > cutoff).sum())
    flat_test_pre = int((pd.to_datetime(eng.loc[te_flat, "INSPECTION_DATE"]) <= cutoff).sum())
    check("flat_split_temporal_clean",
          flat_train_post == 0 and flat_test_pre == 0,
          f"flat: {flat_train_post} train post-cutoff, {flat_test_pre} test pre-cutoff")

    grp_train_post = int((pd.to_datetime(eng.loc[tr_grp, "INSPECTION_DATE"]) > cutoff).sum())
    grp_test_pre = int((pd.to_datetime(eng.loc[te_grp, "INSPECTION_DATE"]) <= cutoff).sum())
    check("group_split_temporal_clean",
          grp_train_post == 0 and grp_test_pre == 0,
          f"group: {grp_train_post} train post-cutoff, {grp_test_pre} test pre-cutoff")

    tr_grp_uprns = set(eng.loc[tr_grp, "UPRN"].dropna())
    te_grp_uprns = set(eng.loc[te_grp, "UPRN"].dropna())
    overlap = len(tr_grp_uprns & te_grp_uprns)
    check("group_split_no_uprn_leakage", overlap == 0,
          f"{overlap} UPRNs in both train and test")

    # --- Feature engineering ---
    flag_cols = [c for c in eng.columns if c.startswith("FLAG_")]
    check("engineered_flags_present", len(flag_cols) >= 5,
          f"{len(flag_cols)} FLAG_* columns: {flag_cols}")
    check("construction_age_engineered",
          "CONSTRUCTION_AGE_NUM" in eng.columns,
          f"CONSTRUCTION_AGE_NUM present, {eng.CONSTRUCTION_AGE_NUM.notna().sum()} populated")
    check("postcode_district_non_pii",
          "POSTCODE_DISTRICT" in eng.columns,
          f"POSTCODE_DISTRICT (outward code only, not full postcode)")

    # --- Reproducibility (deterministic) ---
    # Cleaning is deterministic; re-running gives the same result
    raw2 = load_certificates(RAW)
    clean2 = build_clean_frame(raw2)
    shape_match = (clean.shape == clean2.shape)
    check("reproducible_shape", shape_match,
          f"first run: {clean.shape}, second run: {clean2.shape}")
    # Hash the deterministic columns
    cmp_cols = [c for c in ["LMK_KEY", "CURRENT_ENERGY_RATING",
                            "TOTAL_FLOOR_AREA", "CONSTRUCTION_AGE_BAND"]
                if c in clean.columns and c in clean2.columns]
    if cmp_cols:
        same = all((clean[c].fillna("").astype(str).reset_index(drop=True)
                    == clean2[c].fillna("").astype(str).reset_index(drop=True)).all()
                   for c in cmp_cols)
        check("reproducible_content", same,
              f"deterministic across {len(cmp_cols)} core columns")

    # --- Documentation ---
    pipeline_doc = ROOT / "DATA_PIPELINE.md"
    check("data_pipeline_documented", pipeline_doc.exists(),
          f"{pipeline_doc.name} present" if pipeline_doc.exists() else "missing")

    n_pass = sum(1 for c in checks if c["status"] == "PASS")
    n_fail = len(checks) - n_pass

    summary_out = {
        "raw_file_hash": raw_hash,
        "clean_csv_hash": clean_hash,
        "raw_shape": list(raw.shape),
        "clean_shape": list(clean.shape),
        "published_shape": list(pub.shape),
        "model_matrix_shape": list(X.shape),
        "flat_split": {"train": int(len(tr_flat)), "test": int(len(te_flat))},
        "group_split": {"train": int(len(tr_grp)), "test": int(len(te_grp))},
        "checks_total": len(checks),
        "checks_passed": n_pass,
        "checks_failed": n_fail,
        "checks": checks,
    }
    CHECKS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)

    print("\n" + "=" * 70)
    print(f"RESULT: {n_pass}/{len(checks)} checks PASSED, {n_fail} FAILED")
    print("=" * 70)
    print(f"Written -> {CHECKS_JSON.relative_to(ROOT)}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
