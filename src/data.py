"""Typed loader, leakage filter, deduplicator, and temporal split.

The functions in this module are pure: they take a frame in and return a
frame out. They never mutate global state. This is what lets the pipeline
be reproducible.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Column groupings
# ----------------------------------------------------------------------

#: Columns whose values are *outputs* of the same SAP/RdSAP calculation that
#: produces the target. Including any of these as features for predicting
#: ``CURRENT_ENERGY_RATING`` is data leakage.
LEAKAGE_COLS: tuple[str, ...] = (
    "CURRENT_ENERGY_EFFICIENCY",
    "POTENTIAL_ENERGY_RATING",
    "POTENTIAL_ENERGY_EFFICIENCY",
    "ENVIRONMENT_IMPACT_CURRENT",
    "ENVIRONMENT_IMPACT_POTENTIAL",
    "ENVIRONMENTAL_IMPACT_CURRENT",     # raw schema spelling
    "ENVIRONMENTAL_IMPACT_POTENTIAL",
    "ENERGY_CONSUMPTION_CURRENT",
    "ENERGY_CONSUMPTION_POTENTIAL",
    "CO2_EMISSIONS_CURRENT",
    "CO2_EMISS_CURR_PER_FLOOR_AREA",
    "CO2_EMISSIONS_POTENTIAL",
    "LIGHTING_COST_CURRENT",
    "LIGHTING_COST_POTENTIAL",
    "HEATING_COST_CURRENT",
    "HEATING_COST_POTENTIAL",
    "HOT_WATER_COST_CURRENT",
    "HOT_WATER_COST_POTENTIAL",
)

#: Element-level efficiency star ratings — also SAP outputs. Kept in the
#: cleaned dataset for diagnostic plots, dropped from the model feature matrix.
ELEMENT_EFF_COLS: tuple[str, ...] = (
    "HOT_WATER_ENERGY_EFF", "HOT_WATER_ENV_EFF",
    "FLOOR_ENERGY_EFF", "FLOOR_ENV_EFF",
    "WINDOWS_ENERGY_EFF", "WINDOWS_ENV_EFF",
    "WALLS_ENERGY_EFF", "WALLS_ENV_EFF",
    "SHEATING_ENERGY_EFF", "SHEATING_ENV_EFF",
    "ROOF_ENERGY_EFF", "ROOF_ENV_EFF",
    "MAINHEAT_ENERGY_EFF", "MAINHEAT_ENV_EFF",
    "MAINHEATC_ENERGY_EFF", "MAINHEATC_ENV_EFF",
    "LIGHTING_ENERGY_EFF", "LIGHTING_ENV_EFF",
)

#: Personal data — excluded from the published cleaned CSV per GDPR/DPA 2018.
PII_COLS: tuple[str, ...] = (
    "ADDRESS1", "ADDRESS2", "ADDRESS3", "ADDRESS",
    "POSTCODE", "UPRN", "BUILDING_REFERENCE_NUMBER",
)

#: Constant or redundant administrative columns within a single LA.
ADMIN_DROP: tuple[str, ...] = (
    "LOCAL_AUTHORITY", "LOCAL_AUTHORITY_LABEL",
    "CONSTITUENCY", "CONSTITUENCY_LABEL", "COUNTY",
    "POSTTOWN", "LODGEMENT_DATETIME", "UPRN_SOURCE",
)

#: Type hints for the reader to suppress mixed-dtype warnings.
DTYPE_HINTS: dict[str, str] = {
    "LMK_KEY": "string",
    "ADDRESS1": "string", "ADDRESS2": "string", "ADDRESS3": "string",
    "POSTCODE": "string", "BUILDING_REFERENCE_NUMBER": "string",
    "CURRENT_ENERGY_RATING": "string", "POTENTIAL_ENERGY_RATING": "string",
    "PROPERTY_TYPE": "string", "BUILT_FORM": "string",
    "LOCAL_AUTHORITY": "string", "CONSTITUENCY": "string", "COUNTY": "string",
    "TRANSACTION_TYPE": "string",
    "ENERGY_TARIFF": "string", "MAINS_GAS_FLAG": "string",
    "FLOOR_LEVEL": "string", "FLAT_TOP_STOREY": "string",
    "MAIN_HEATING_CONTROLS": "string",
    "GLAZED_TYPE": "string", "GLAZED_AREA": "string",
    "HOTWATER_DESCRIPTION": "string",
    "FLOOR_DESCRIPTION": "string", "WINDOWS_DESCRIPTION": "string",
    "WALLS_DESCRIPTION": "string", "SECONDHEAT_DESCRIPTION": "string",
    "ROOF_DESCRIPTION": "string", "MAINHEAT_DESCRIPTION": "string",
    "MAINHEATCONT_DESCRIPTION": "string", "LIGHTING_DESCRIPTION": "string",
    "MAIN_FUEL": "string", "HEAT_LOSS_CORRIDOR": "string",
    "SOLAR_WATER_HEATING_FLAG": "string", "MECHANICAL_VENTILATION": "string",
    "TENURE": "string",
    "CONSTRUCTION_AGE_BAND": "string",
}

DATE_COLS: tuple[str, ...] = ("INSPECTION_DATE", "LODGEMENT_DATE")

#: Explicit "no data" sentinel strings used throughout the EPB register.
#: These are *not* genuine categories — they mean the assessor did not
#: record the field. Left as raw strings they (a) pollute the categorical
#: encoders with a fake level and (b) escape the missingness indicators,
#: which test for ``NaN`` and not for these tokens. We normalise them to
#: ``NaN`` at load time so every downstream consumer (clean frame, model
#: matrix, diagnostics) treats them consistently as missing. Matched
#: case-insensitively after whitespace stripping.
SENTINEL_STRINGS: frozenset[str] = frozenset({
    "NODATA!", "NO DATA!", "NODATA", "NO DATA", "INVALID!", "NOT DEFINED",
})


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def load_certificates(path: str | Path) -> pd.DataFrame:
    """Read certificates.csv with explicit dtypes and parsed dates.

    Parameters
    ----------
    path : str or Path
        Filesystem path to ``certificates.csv``.
    """
    df = pd.read_csv(
        path,
        dtype={k: v for k, v in DTYPE_HINTS.items()},
        parse_dates=list(DATE_COLS),
        low_memory=False,
        encoding="utf-8",
    )
    # Some EPC dumps spell the column ENVIRONMENT_IMPACT_*, others
    # ENVIRONMENTAL_IMPACT_*. Normalise to the canonical spelling used in
    # docs/columns.csv ("ENVIRONMENTAL_IMPACT_*").
    rename = {
        "ENVIRONMENT_IMPACT_CURRENT": "ENVIRONMENTAL_IMPACT_CURRENT",
        "ENVIRONMENT_IMPACT_POTENTIAL": "ENVIRONMENTAL_IMPACT_POTENTIAL",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df = normalise_sentinels(df)
    return df


def normalise_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Replace EPC "no data" sentinel strings with ``NaN`` in text columns.

    See :data:`SENTINEL_STRINGS`. Operates only on object / pandas-``string``
    columns; numeric and datetime columns are untouched. Comparison is
    whitespace-insensitive and case-insensitive.
    """
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if s.dtype == object or isinstance(s.dtype, pd.StringDtype):
            norm = s.astype("string").str.strip().str.upper()
            mask = norm.isin(SENTINEL_STRINGS).fillna(False)
            if mask.any():
                out.loc[mask, c] = pd.NA
    return out


def load_recommendations(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype="string", encoding="utf-8")


# ----------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------

def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def drop_pii(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in PII_COLS if c in df.columns])


def drop_admin(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in ADMIN_DROP if c in df.columns])


def drop_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Remove SAP-engine *output* columns. Keeps only physical features."""
    return df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])


#: Core fields that should be populated for a usable certificate. Used by
#: quality-aware dedup as the tie-breaker when chronology alone is insufficient
#: (e.g. when the latest cert is sparser than an earlier one for the same
#: dwelling).
_CORE_FIELDS: tuple[str, ...] = (
    "TOTAL_FLOOR_AREA", "NUMBER_HABITABLE_ROOMS", "WALLS_DESCRIPTION",
    "ROOF_DESCRIPTION", "WINDOWS_DESCRIPTION", "MAINHEAT_DESCRIPTION",
    "CONSTRUCTION_AGE_BAND", "PROPERTY_TYPE", "BUILT_FORM",
)


def deduplicate_latest(df: pd.DataFrame, key: str = "UPRN",
                       date_col: str = "INSPECTION_DATE",
                       quality_aware: bool = True) -> pd.DataFrame:
    """Keep one EPC per dwelling, preferring chronology then completeness.

    UPRN is the only durable identifier that ties multiple lodgements to the
    same physical building (``LMK_KEY`` is unique per certificate, not per
    property). With ``quality_aware=True`` (default), when a UPRN has multiple
    lodgements the row chosen is the one that maximises a combined score:

        score = (year - 2007) * 10 + (count of populated CORE_FIELDS)

    This biases toward the most recent record but switches to an earlier
    cert when the latest is materially sparser. Rows where ``key`` is
    missing are kept as-is (they cannot be de-duplicated against anything else).
    """
    if key not in df.columns or date_col not in df.columns:
        return df.reset_index(drop=True)

    has_key = df[key].notna()
    keyed = df[has_key].copy()
    no_key = df[~has_key]

    if not quality_aware:
        keyed = keyed.sort_values(date_col).drop_duplicates(subset=[key], keep="last")
        return pd.concat([keyed, no_key], ignore_index=True)

    # Composite score: chronology dominates, completeness breaks ties
    available_core = [c for c in _CORE_FIELDS if c in keyed.columns]
    completeness = keyed[available_core].notna().sum(axis=1) if available_core else 0
    year = pd.to_datetime(keyed[date_col], errors="coerce").dt.year.fillna(2007)
    keyed = keyed.assign(_score=(year - 2007) * 10 + completeness)
    keyed = (keyed.sort_values([key, "_score"])
                   .drop_duplicates(subset=[key], keep="last")
                   .drop(columns=["_score"]))
    return pd.concat([keyed, no_key], ignore_index=True)


# Note: an earlier version of this module exposed
# ``drop_unkeyed_likely_duplicates`` that attempted to fingerprint un-keyed
# rows against UPRN-keyed rows using the LMK_KEY prefix. Empirical testing
# (reports/data_quality_report.csv) showed it matched zero rows because
# LMK_KEY is unique per certificate, not per building, so the fingerprint
# could not link the two cohorts. The function was removed rather than left
# in place as dead code. Un-keyed rows (1,684 in the raw data, mostly
# new-builds before UPRN allocation) are retained as-is and treated as
# legitimate one-off certificates.


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Winsorise floor-area at the 99.5th percentile and drop sub-10 m² rows.

    The 99.5th-percentile cap (≈ 306 m² in Oxford) trims a long right tail of
    likely mis-classified non-domestic buildings (max raw value 2 605 m²).
    The 10 m² lower bound removes 105 raw rows where TOTAL_FLOOR_AREA is
    smaller than the smallest plausible studio bedsit and is therefore
    almost certainly a data-entry error rather than a real dwelling.
    """
    df = df.copy()
    if "TOTAL_FLOOR_AREA" in df.columns:
        cap = df["TOTAL_FLOOR_AREA"].quantile(0.995)
        df.loc[df["TOTAL_FLOOR_AREA"] > cap, "TOTAL_FLOOR_AREA"] = cap
        df = df[df["TOTAL_FLOOR_AREA"].fillna(99) >= 10]
    return df


# ----------------------------------------------------------------------
# Numeric / logical consistency validation
# ----------------------------------------------------------------------

def validate_consistency(df: pd.DataFrame, log_path: str | Path | None = None
                         ) -> pd.DataFrame:
    """Repair illegal values and surface implicit signal hidden in them.

    Five classes of issue are addressed:

    * ``NUMBER_HEATED_ROOMS > NUMBER_HABITABLE_ROOMS`` — physically impossible.
      Heated is clipped to habitable.
    * ``NUMBER_HEATED_ROOMS == 0`` or ``NUMBER_HABITABLE_ROOMS == 0`` — also
      impossible for a residential dwelling. Set to NaN so the imputer can fill.
    * ``TOTAL_FLOOR_AREA == 0`` — impossible. Set to NaN.
    * ``CO2_EMISSIONS_CURRENT < 0`` or ``ENERGY_CONSUMPTION_CURRENT < 0`` — only
      possible when on-site generation (PV, micro-wind) exceeds consumption.
      A new ``HAS_ONSITE_GENERATION`` flag is added so this signal survives the
      removal of the leakage columns themselves.
    * ``NUMBER_HABITABLE_ROOMS > 30`` — unflagged HMO or hotel; left in place
      but tracked in the quality report.

    A summary is written to ``log_path`` (default
    ``reports/data_quality_report.csv``) so the cleaning is auditable.
    """
    out = df.copy()
    log: list[dict] = []

    def _record(rule: str, n: int, action: str) -> None:
        log.append({"rule": rule, "n_rows_affected": int(n), "action": action})

    # 1) heated > habitable: clip heated to habitable
    if {"NUMBER_HEATED_ROOMS", "NUMBER_HABITABLE_ROOMS"}.issubset(out.columns):
        heat = pd.to_numeric(out["NUMBER_HEATED_ROOMS"], errors="coerce")
        hab = pd.to_numeric(out["NUMBER_HABITABLE_ROOMS"], errors="coerce")
        bad = (heat > hab) & heat.notna() & hab.notna()
        out.loc[bad, "NUMBER_HEATED_ROOMS"] = hab[bad]
        _record("heated > habitable", bad.sum(), "clip heated to habitable")

    # 2) zero rooms / zero floor area -> NaN
    for col in ("NUMBER_HEATED_ROOMS", "NUMBER_HABITABLE_ROOMS", "TOTAL_FLOOR_AREA"):
        if col in out.columns:
            s = pd.to_numeric(out[col], errors="coerce")
            mask = s == 0
            n = int(mask.sum())
            if n > 0:
                out.loc[mask, col] = np.nan
                _record(f"{col} == 0", n, "set NaN (impossible value)")

    # 3) negative consumption / emissions -> add HAS_ONSITE_GENERATION flag
    pv_mask = pd.Series(False, index=out.index)
    for col in ("CO2_EMISSIONS_CURRENT", "ENERGY_CONSUMPTION_CURRENT"):
        if col in out.columns:
            s = pd.to_numeric(out[col], errors="coerce")
            pv_mask = pv_mask | (s < 0)
    out["HAS_ONSITE_GENERATION"] = pv_mask.astype("Int8")
    _record("negative emissions/consumption", int(pv_mask.sum()),
            "added HAS_ONSITE_GENERATION=1")

    # 4) HMO outliers (kept, tracked only)
    if "NUMBER_HABITABLE_ROOMS" in out.columns:
        hab = pd.to_numeric(out["NUMBER_HABITABLE_ROOMS"], errors="coerce")
        n = int((hab > 30).sum())
        if n:
            _record("habitable_rooms > 30 (HMO)", n, "kept; tracked only")

    # 5) duplicate-rating-on-dedup tracker is reported elsewhere
    if log_path is not None:
        log_df = pd.DataFrame(log)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_df.to_csv(log_path, index=False)

    return out


def drop_fully_missing(df: pd.DataFrame, threshold: float = 0.999) -> pd.DataFrame:
    """Drop columns that are essentially entirely NaN.

    Default threshold of 0.999 (≥ 99.9% missing) catches columns like
    ``SHEATING_ENERGY_EFF`` which are 100% NaN in the Oxford subset because
    no dwelling there has a recorded second-heating system.
    """
    keep = [c for c in df.columns if df[c].isna().mean() < threshold]
    return df[keep]


def filter_oxford(df: pd.DataFrame) -> pd.DataFrame:
    """Defensive: ensure rows are Oxford even if upstream filtering changes."""
    if "LOCAL_AUTHORITY" in df.columns:
        df = df[df["LOCAL_AUTHORITY"] == "E07000178"]
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------
# High-level wranglers
# ----------------------------------------------------------------------

def build_clean_frame(raw: pd.DataFrame,
                      quality_report_path: str | Path | None = None
                      ) -> pd.DataFrame:
    """Apply the full cleaning chain and return an analysis-ready frame.

    Step order (and why):

    1. ``filter_oxford`` – defensive: ensures every row is Oxford.
    2. ``coerce_numeric`` – cast 30 fields whose raw dtype is mixed
       (whitespace and "NO DATA!" entries force pandas into ``object``).
    3. ``validate_consistency`` – repair impossible values (zero rooms,
       heated > habitable, zero area) and surface the negative-emission
       signal as ``HAS_ONSITE_GENERATION`` before the leakage columns
       are dropped further down the pipeline.
    4. ``drop_unkeyed_likely_duplicates`` – remove un-keyed rows that match
       a UPRN-keyed row on (LMK prefix, area, date).
    5. ``deduplicate_latest`` – one row per UPRN, preferring chronology then
       completeness of the core fields.
    6. ``cap_outliers`` – winsorise floor area, drop sub-10 m² dwellings.
    7. ``drop_fully_missing`` – column-level prune of ≥99.9% NaN fields.
    8. ``dropna(target)`` and ``A-G`` filter – drop rows with no/invalid
       target.

    The returned frame *retains* element ratings (``*_ENERGY_EFF``) and the
    secondary numeric target (``CURRENT_ENERGY_EFFICIENCY``) so diagnostic
    plots in §3.2-3.3 can use them. The model feature matrix used in §3.4
    is produced by :func:`to_model_matrix`, which strips every SAP-engine
    output (and, by default, raw description text and methodology labels)
    to prevent target leakage and methodology shortcuts.
    """
    out = (
        raw
        .pipe(filter_oxford)
        .pipe(coerce_numeric, cols=[
            "CURRENT_ENERGY_EFFICIENCY", "POTENTIAL_ENERGY_EFFICIENCY",
            "ENVIRONMENTAL_IMPACT_CURRENT", "ENVIRONMENTAL_IMPACT_POTENTIAL",
            "ENERGY_CONSUMPTION_CURRENT", "ENERGY_CONSUMPTION_POTENTIAL",
            "CO2_EMISSIONS_CURRENT", "CO2_EMISS_CURR_PER_FLOOR_AREA",
            "CO2_EMISSIONS_POTENTIAL",
            "LIGHTING_COST_CURRENT", "LIGHTING_COST_POTENTIAL",
            "HEATING_COST_CURRENT", "HEATING_COST_POTENTIAL",
            "HOT_WATER_COST_CURRENT", "HOT_WATER_COST_POTENTIAL",
            "TOTAL_FLOOR_AREA", "MULTI_GLAZE_PROPORTION",
            "EXTENSION_COUNT", "NUMBER_HABITABLE_ROOMS", "NUMBER_HEATED_ROOMS",
            "LOW_ENERGY_LIGHTING", "NUMBER_OPEN_FIREPLACES",
            "WIND_TURBINE_COUNT", "UNHEATED_CORRIDOR_LENGTH",
            "FLOOR_HEIGHT", "PHOTO_SUPPLY", "FLAT_STOREY_COUNT",
            "FIXED_LIGHTING_OUTLETS_COUNT", "LOW_ENERGY_FIXED_LIGHT_COUNT",
            "REPORT_TYPE",
        ])
        .pipe(validate_consistency, log_path=quality_report_path)
        .pipe(deduplicate_latest)
        .pipe(cap_outliers)
        .pipe(drop_fully_missing, threshold=0.999)
        # remove rows with no target
        .dropna(subset=["CURRENT_ENERGY_RATING"])
        # restrict to A-G valid bands
        .loc[lambda d: d["CURRENT_ENERGY_RATING"].isin(list("ABCDEFG"))]
        .reset_index(drop=True)
    )
    return out


def publishable_clean_frame(clean: pd.DataFrame) -> pd.DataFrame:
    """The frame written to ``data/processed/oxford_epc_clean.csv``.

    Drops PII and constant administrative columns. Keeps element ratings
    and the secondary numeric target so future analysts can reproduce
    diagnostic work without re-licensing the address data.
    """
    return clean.pipe(drop_pii).pipe(drop_admin)


def _sanitise_extension_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas extension-dtype columns (``string``, ``Int*``, ``Float*``)
    to plain numpy dtypes with ``np.nan`` for missingness.

    sklearn's SimpleImputer relies on ``X != X`` to detect missingness, which
    raises ``TypeError: boolean value of NA is ambiguous`` when it encounters
    ``pd.NA``. Casting at the model-matrix boundary keeps the cleaned CSV
    pandas-native while making the model frame sklearn-safe.
    """
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if isinstance(s.dtype, pd.StringDtype):
            out[c] = s.astype(object).where(s.notna(), np.nan)
        elif pd.api.types.is_extension_array_dtype(s):
            # Int8/Int16/Int32/Int64/Float32/Float64 extension dtypes
            out[c] = pd.to_numeric(s.astype(object), errors="coerce")
    return out


#: Free-text description columns. Useful for the engineered FLAG_* features
#: (see ``src.features.add_text_flags``) but harmful as raw inputs because
#: OrdinalEncoder assigns arbitrary integer codes that lose the physical
#: similarity structure (e.g. "Cavity wall, insulated" and "Cavity wall, filled
#: cavity" become unrelated codes). Dropped from the model matrix; the
#: extracted flags carry the predictive signal instead.
_RAW_DESCRIPTION_COLS: tuple[str, ...] = (
    "WALLS_DESCRIPTION", "ROOF_DESCRIPTION", "WINDOWS_DESCRIPTION",
    "MAINHEAT_DESCRIPTION", "MAINHEATCONT_DESCRIPTION", "HOTWATER_DESCRIPTION",
    "FLOOR_DESCRIPTION", "LIGHTING_DESCRIPTION", "SECONDHEAT_DESCRIPTION",
)

#: Methodology / administrative labels that correlate with rating only because
#: they encode *when* the certificate was lodged, not *what the building is
#: like*. Including them lets the model take a methodology shortcut that does
#: not generalise (e.g. SAP-10 certificates skew A/B regardless of fabric).
_METHODOLOGY_COLS: tuple[str, ...] = ("REPORT_TYPE", "LODGEMENT_DATE")

#: Raw columns that an engineered numeric feature fully supersedes. The raw
#: ``FLOOR_LEVEL`` string is recorded in incompatible formats ("Ground",
#: "00", "1st", "Basement") that the categorical encoder fragments into
#: unrelated levels; ``features.floor_level_num`` collapses them onto one
#: signed-integer storey scale (ground = 0, basement = -1, Nth = N) as
#: ``FLOOR_LEVEL_NUM``. We drop the raw column from the model matrix so the
#: two do not coexist; the clean published CSV keeps the raw column for
#: transparency.
_SUPERSEDED_COLS: tuple[str, ...] = ("FLOOR_LEVEL",)


def add_missingness_indicators(df: pd.DataFrame, threshold: float = 0.05,
                               reference: pd.Index | None = None
                               ) -> pd.DataFrame:
    """Add ``<COL>_IS_MISSING`` Int8 flags for columns above ``threshold``.

    Median/mode imputation assumes data are missing completely at random
    (MCAR). Many EPC fields are MNAR — ``FLAT_STOREY_COUNT`` is missing
    iff the dwelling is not a flat. A binary indicator lets the tree
    learn this structural meaning instead of pretending the imputed value
    is real.

    ``reference`` (optional): the row index on which to *decide* which
    columns exceed ``threshold`` — pass the training index so the feature
    *set* is not chosen using the test rows (a feature-selection leak). The
    indicator columns themselves are still computed for all rows.
    """
    out = df.copy()
    ref = df if reference is None else df.loc[reference]
    for c in df.columns:
        miss_rate = ref[c].isna().mean()
        if miss_rate >= threshold and c != "CURRENT_ENERGY_RATING":
            out[f"{c}_IS_MISSING"] = df[c].isna().astype("Int8")
    return out


def to_model_matrix(clean: pd.DataFrame, *,
                    drop_raw_descriptions: bool = False,
                    drop_methodology: bool = False,
                    add_missingness: bool = True,
                    missingness_threshold: float = 0.05,
                    missingness_ref: pd.Index | None = None,
                    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Produce ``(X, y_class, y_reg)`` with leakage features removed.

    Defaults are set by an ablation study (``reports/ablation_study.json``):
    of three candidate "improvements", only ``add_missingness`` measurably
    helps. Dropping the raw description columns costs −0.058 QWK because
    OrdinalEncoder's frequency-ordered codes preserve enough discrimination
    that trees can still split on them, even though the codes have no
    semantic meaning. Dropping ``REPORT_TYPE`` costs −0.005 QWK — a
    cleaner-design argument can be made for the principled choice, but
    the empirical evidence does not support it as a default.

    Parameters
    ----------
    drop_raw_descriptions : bool, default False
        Remove free-text ``*_DESCRIPTION`` columns. Available as an option
        for sensitivity analysis but harmful as a default (see ablation).
    drop_methodology : bool, default False
        Remove ``REPORT_TYPE`` and ``LODGEMENT_DATE``. Cleaner design but
        small empirical cost; off by default.
    add_missingness : bool, default True
        Append ``<COL>_IS_MISSING`` indicator columns for any feature with
        missingness ≥ ``missingness_threshold``. Improves macro-F1 by ~+0.014
        and per-class F1 for the minority bands A, F and G.
    """
    drop = (set(LEAKAGE_COLS) | set(ELEMENT_EFF_COLS) | set(PII_COLS)
            | set(ADMIN_DROP) | set(_SUPERSEDED_COLS))
    if drop_raw_descriptions:
        drop |= set(_RAW_DESCRIPTION_COLS)
    if drop_methodology:
        drop |= set(_METHODOLOGY_COLS)
    drop_present = [c for c in drop if c in clean.columns]

    feat = clean.drop(columns=drop_present + ["CURRENT_ENERGY_RATING"])
    if "LMK_KEY" in feat.columns:
        feat = feat.drop(columns=["LMK_KEY"])

    if add_missingness:
        feat = add_missingness_indicators(feat, threshold=missingness_threshold,
                                          reference=missingness_ref)

    feat = _sanitise_extension_dtypes(feat)
    y_class = clean["CURRENT_ENERGY_RATING"].astype(object)
    y_reg = clean.get("CURRENT_ENERGY_EFFICIENCY", pd.Series(index=clean.index, dtype="float"))
    return feat, y_class, y_reg


# ----------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------

def temporal_split(df: pd.DataFrame, cutoff: str = "2022-12-31",
                   date_col: str = "INSPECTION_DATE"
                   ) -> tuple[pd.Index, pd.Index]:
    """Return (train_idx, test_idx) using a strict temporal cutoff."""
    cutoff_ts = pd.Timestamp(cutoff)
    train = df.index[df[date_col] <= cutoff_ts]
    test = df.index[df[date_col] > cutoff_ts]
    return train, test


def group_temporal_split(df: pd.DataFrame, cutoff: str = "2022-12-31",
                         date_col: str = "INSPECTION_DATE",
                         key: str = "UPRN"
                         ) -> tuple[pd.Index, pd.Index]:
    """Group-aware temporal split that preserves multi-cert training signal.

    The flat ``temporal_split`` requires that the input frame be already
    deduplicated (one row per dwelling), discarding ~27 000 earlier
    certificates. ``group_temporal_split`` works on the *un*-deduplicated
    frame and recovers most of those rows for training:

    * For each UPRN, find the date of its latest certificate.
    * If that latest date ≤ ``cutoff`` → all of the UPRN's certificates
      go to **train** (no leakage: every cert of this dwelling is
      pre-cutoff).
    * If the latest date > ``cutoff`` → only the **latest** cert goes to
      **test**; earlier certs are discarded (they cannot enter train without
      leaking the dwelling into both folds).
    * Un-keyed rows are split by their own ``date_col`` value.

    The result: train gains ~14 000 longitudinal observations (multiple
    certs per dwelling showing pre-retrofit and post-retrofit states),
    while test is exactly one cert per dwelling so per-band metrics are
    not double-counted.

    Returns
    -------
    (train_idx, test_idx) : pd.Index, pd.Index
        Indices into ``df``. The intersection is empty by construction.
    """
    cutoff_ts = pd.Timestamp(cutoff)
    df = df.copy()
    df["_date_norm"] = pd.to_datetime(df[date_col], errors="coerce")

    has_key = df[key].notna() if key in df.columns else pd.Series(False, index=df.index)
    keyed = df[has_key]
    no_key = df[~has_key]

    train_parts: list[pd.Index] = []
    test_parts: list[pd.Index] = []

    if not keyed.empty:
        latest_per_key = keyed.groupby(key)["_date_norm"].transform("max")
        latest_in_train = latest_per_key <= cutoff_ts
        latest_in_test = latest_per_key > cutoff_ts

        # Train: every cert of UPRNs whose latest is pre-cutoff
        train_parts.append(keyed.index[latest_in_train])
        # Test: only the latest cert of UPRNs whose latest is post-cutoff
        post_subset = keyed[latest_in_test].copy()
        # idx of the latest cert per UPRN within post_subset
        latest_idx = post_subset.sort_values("_date_norm").drop_duplicates(
            subset=[key], keep="last").index
        test_parts.append(latest_idx)

    if not no_key.empty:
        no_key_pre = no_key.index[no_key["_date_norm"] <= cutoff_ts]
        no_key_post = no_key.index[no_key["_date_norm"] > cutoff_ts]
        train_parts.append(no_key_pre)
        test_parts.append(no_key_post)

    train_idx = pd.Index(np.concatenate([np.asarray(p) for p in train_parts]))
    test_idx = pd.Index(np.concatenate([np.asarray(p) for p in test_parts]))
    return train_idx, test_idx
