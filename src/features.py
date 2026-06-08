"""Feature engineering — pure functions on DataFrames."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Construction-age band -> integer year
# ----------------------------------------------------------------------

_AGE_BAND_MIDPOINTS = {
    "England and Wales: before 1900": 1880,
    "England and Wales: 1900-1929": 1914,
    "England and Wales: 1930-1949": 1939,
    "England and Wales: 1950-1966": 1958,
    "England and Wales: 1967-1975": 1971,
    "England and Wales: 1976-1982": 1979,
    "England and Wales: 1983-1990": 1986,
    "England and Wales: 1991-1995": 1993,
    "England and Wales: 1996-2002": 1999,
    "England and Wales: 2003-2006": 2004,
    "England and Wales: 2007 onwards": 2009,        # legacy banding (pre-2012)
    "England and Wales: 2007-2011": 2009,
    "England and Wales: 2012 onwards": 2015,
    "England and Wales: 2012-2021": 2016,           # Wales-only post-SAP10 banding
    "England and Wales: 2022 onwards": 2023,
    # Wales-only / unbranded variants
    "before 1900": 1880, "1900-1929": 1914, "1930-1949": 1939,
    "1950-1966": 1958, "1967-1975": 1971, "1976-1982": 1979,
    "1983-1990": 1986, "1991-1995": 1993, "1996-2002": 1999,
    "2003-2006": 2004, "2007-2011": 2009, "2007 onwards": 2009,
    "2012 onwards": 2015, "2012-2021": 2016, "2022 onwards": 2023,
}


_BARE_YEAR_RE = re.compile(r"^\s*(\d{4})\s*$")


def construction_age_num(s: pd.Series) -> pd.Series:
    """Map ``CONSTRUCTION_AGE_BAND`` to a single integer year.

    The EPB Register encodes age either as a band ("England and Wales:
    1930-1949"), a legacy band ("2007 onwards"), or — for SAP 10
    certificates issued from 2022 — as the bare construction year
    ("2019", "2024", …). We handle all three here so the field is fully
    populated for downstream modelling.
    """
    def _map(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return float("nan")
        v = str(v).strip()
        if v in _AGE_BAND_MIDPOINTS:
            return float(_AGE_BAND_MIDPOINTS[v])
        m = _BARE_YEAR_RE.match(v)
        if m:
            yr = int(m.group(1))
            if 1850 <= yr <= 2050:
                return float(yr)
        return float("nan")
    return s.map(_map).astype("Float32")


# ----------------------------------------------------------------------
# Floor level -> signed-integer storey
# ----------------------------------------------------------------------

_FLOOR_ORDINAL_RE = re.compile(r"^(\d{1,2})\s*(?:st|nd|rd|th)?(?:\s*floor)?$", re.I)


def floor_level_num(s: pd.Series) -> pd.Series:
    """Canonicalise the messy ``FLOOR_LEVEL`` field to a signed integer storey.

    The EPB register records floor level for flats and maisonettes in several
    mutually incompatible formats — numeric strings ("00", "01", "02"),
    ordinals ("1st", "2nd", "3rd"), and words ("Ground", "Basement") — which
    a categorical encoder would fragment into unrelated levels. We map them
    all onto one scale:

    * ``Basement`` / ``-1``                     -> -1
    * ``Ground`` / ``ground floor`` / ``00``    ->  0
    * ``Nth`` / ``0N`` / ``N``                  ->  N

    Vague descriptors ("mid floor", "top floor"), sentinels, and any
    unrecognised value -> ``NaN``, so the accompanying
    ``FLOOR_LEVEL_NUM_IS_MISSING`` indicator carries that signal rather than
    a fabricated number. Houses (where the field is not applicable) are
    already ``NaN`` and stay so.
    """
    def _map(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return float("nan")
        t = str(v).strip().lower()
        if t == "" or t == "<na>":
            return float("nan")
        if "basement" in t:
            return -1.0
        if t.startswith("ground") or t in ("g", "gf"):
            return 0.0
        m = _FLOOR_ORDINAL_RE.match(t)
        if m:
            return float(int(m.group(1)))
        try:
            return float(int(t))
        except ValueError:
            return float("nan")
    return s.map(_map).astype("Float32")


# ----------------------------------------------------------------------
# Postcode district (no PII — district is the public part of the postcode)
# ----------------------------------------------------------------------

def postcode_district(s: pd.Series) -> pd.Series:
    """Return the outward portion of a UK postcode (e.g. 'OX4 1AB' -> 'OX4').
    Using the district (not the full postcode) keeps the feature non-personal.
    """
    return (
        s.astype("string")
         .str.upper()
         .str.split(" ", n=1)
         .str[0]
         .fillna("UNK")
    )


# ----------------------------------------------------------------------
# Description-string flags  (regex extraction is robust to phrasing variants)
# ----------------------------------------------------------------------

_RE_CAVITY_INSULATED = re.compile(r"cavity wall.*(insulated|filled)", re.I)
_RE_SOLID_BRICK_BARE = re.compile(r"solid brick.*(no insulation|as built|partial insulation)", re.I)
_RE_HEAT_PUMP = re.compile(r"heat pump", re.I)
_RE_TRIPLE_GLAZED = re.compile(r"triple", re.I)
_RE_SINGLE_GLAZED = re.compile(r"^(?!.*double).*single", re.I)
_RE_NO_LOFT_INSULATION = re.compile(r"no insulation|loft insulation, no insulation", re.I)
# In the EPC ROOF_DESCRIPTION the depth comes BEFORE the phrase
# "loft insulation" — e.g. "Pitched, 250 mm loft insulation".
_RE_LOFT_MM = re.compile(r"(\d{2,3})\s*mm", re.I)


def _loft_depth_mm(s: str) -> int:
    """Extract loft-insulation depth in millimetres from a ROOF_DESCRIPTION
    string. Returns 0 if no depth is reported (e.g. flat roof, room-in-roof,
    or 'no insulation')."""
    if not s:
        return 0
    m = _RE_LOFT_MM.search(s)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return 0


def add_text_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean / numeric flags derived from the natural-language ``*_DESCRIPTION``
    fields. Each flag has a hypothesised sign on the EPC rating; their
    realised effects are reported in the SHAP analysis."""
    out = df.copy()
    walls = out.get("WALLS_DESCRIPTION", pd.Series("", index=out.index)).fillna("")
    roof = out.get("ROOF_DESCRIPTION", pd.Series("", index=out.index)).fillna("")
    mainheat = out.get("MAINHEAT_DESCRIPTION", pd.Series("", index=out.index)).fillna("")
    windows = out.get("WINDOWS_DESCRIPTION", pd.Series("", index=out.index)).fillna("")

    out["FLAG_CAVITY_INSULATED"] = walls.str.contains(_RE_CAVITY_INSULATED).astype("Int8")
    out["FLAG_SOLID_BRICK_BARE"] = walls.str.contains(_RE_SOLID_BRICK_BARE).astype("Int8")
    out["FLAG_HEAT_PUMP"] = mainheat.str.contains(_RE_HEAT_PUMP).astype("Int8")
    out["FLAG_TRIPLE_GLAZED"] = windows.str.contains(_RE_TRIPLE_GLAZED).astype("Int8")
    out["FLAG_SINGLE_GLAZED"] = windows.str.contains(_RE_SINGLE_GLAZED).astype("Int8")

    # Loft depth: numeric (mm) and a 270 mm threshold flag (current Building
    # Regulations Part L recommended depth for a new loft).
    depth = roof.apply(_loft_depth_mm).astype("Int16")
    no_loft = roof.str.contains(_RE_NO_LOFT_INSULATION, regex=True, na=False)
    depth = depth.where(~no_loft, 0)
    out["LOFT_INSULATION_MM"] = depth
    out["FLAG_LOFT_270MM"] = (depth >= 270).astype("Int8")
    out["FLAG_NO_LOFT_INSULATION"] = no_loft.astype("Int8")

    # Mains-heating fuel + system flags (more reliably populated than
    # 'condensing' which is rarely present in MAINHEAT_DESCRIPTION strings).
    out["FLAG_GAS_BOILER"] = mainheat.str.contains(r"mains gas", regex=True, case=False, na=False).astype("Int8")
    out["FLAG_ELECTRIC_HEATING"] = mainheat.str.contains(
        r"electric|storage heater", regex=True, case=False, na=False).astype("Int8")
    return out


# ----------------------------------------------------------------------
# Numeric ratios and date parts
# ----------------------------------------------------------------------

def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    fla = out.get("FIXED_LIGHTING_OUTLETS_COUNT")
    lel = out.get("LOW_ENERGY_FIXED_LIGHT_COUNT")
    if fla is not None and lel is not None:
        denom = fla.replace(0, np.nan)
        out["LOW_ENERGY_LIGHTING_RATIO"] = (lel / denom).clip(0, 1).astype("Float32")
    nhr = out.get("NUMBER_HEATED_ROOMS")
    nhab = out.get("NUMBER_HABITABLE_ROOMS")
    if nhr is not None and nhab is not None:
        denom = nhab.replace(0, np.nan)
        out["HEATED_ROOM_RATIO"] = (nhr / denom).clip(0, 1).astype("Float32")
    if "TOTAL_FLOOR_AREA" in out.columns and "NUMBER_HABITABLE_ROOMS" in out.columns:
        denom = out["NUMBER_HABITABLE_ROOMS"].replace(0, np.nan)
        out["AREA_PER_ROOM"] = (out["TOTAL_FLOOR_AREA"] / denom).astype("Float32")
    if "TOTAL_FLOOR_AREA" in out.columns:
        out["LOG_FLOOR_AREA"] = np.log1p(out["TOTAL_FLOOR_AREA"]).astype("Float32")
    return out


def add_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "INSPECTION_DATE" in out.columns:
        d = pd.to_datetime(out["INSPECTION_DATE"], errors="coerce")
        out["INSPECTION_YEAR"] = d.dt.year.astype("Int16")
        out["INSPECTION_QUARTER"] = d.dt.quarter.astype("Int8")
    if "CONSTRUCTION_AGE_BAND" in out.columns:
        out["CONSTRUCTION_AGE_NUM"] = construction_age_num(out["CONSTRUCTION_AGE_BAND"])
    if "FLOOR_LEVEL" in out.columns:
        out["FLOOR_LEVEL_NUM"] = floor_level_num(out["FLOOR_LEVEL"])
    if "POSTCODE" in out.columns:
        out["POSTCODE_DISTRICT"] = postcode_district(out["POSTCODE"])
    return out


# ----------------------------------------------------------------------
# Top-level orchestrator
# ----------------------------------------------------------------------

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    return df.pipe(add_date_parts).pipe(add_text_flags).pipe(add_ratios)


# ----------------------------------------------------------------------
# Recommendations parsing
# ----------------------------------------------------------------------

_COST_RE = re.compile(r"£?([\d,]+)\s*-\s*£?([\d,]+)")
_SINGLE_RE = re.compile(r"£?([\d,]+)")


def parse_indicative_cost(s: pd.Series) -> pd.DataFrame:
    """Parse strings like '£4,000 - £14,000' or '£25' into low/high/mid floats."""
    def _one(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return (np.nan, np.nan, np.nan)
        v = str(v).replace("\xa0", " ").strip()
        m = _COST_RE.search(v)
        if m:
            lo = float(m.group(1).replace(",", ""))
            hi = float(m.group(2).replace(",", ""))
            return (lo, hi, (lo + hi) / 2.0)
        m = _SINGLE_RE.search(v)
        if m:
            x = float(m.group(1).replace(",", ""))
            return (x, x, x)
        return (np.nan, np.nan, np.nan)
    arr = np.array([_one(v) for v in s.tolist()], dtype="float64")
    return pd.DataFrame(arr, columns=["COST_LOW", "COST_HIGH", "COST_MID"], index=s.index)
