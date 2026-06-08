"""Unsupervised property segmentation for Oxford's housing stock.

Design choices (best-practice, evidence-based):

* **Mixed-type data → K-Prototypes** (Huang, 1998). EPC fabric features are a
  mix of numeric (floor area, construction-age year, rooms) and categorical
  (property type, built form, wall construction, main fuel). One-hot + K-Means
  distorts the geometry of categorical levels; K-Prototypes uses a combined
  dissimilarity (squared Euclidean for numerics, simple-matching for
  categoricals), which the EPC-clustering literature favours for this data.
* **Leak-free features only** — exactly the physical fabric the predictive
  model is allowed to see. No SAP score, no element-efficiency ratings, no
  costs. The cluster SAP/band profile is computed *after* clustering, purely
  for interpretation, so the segments are not circularly defined by the
  outcome.
* **k chosen by multiple internal indices** — not the elbow alone. We report
  the K-Prototypes cost (elbow), the silhouette coefficient (on a Gower
  distance matrix, the correct metric for mixed data), and pick k by the
  silhouette peak tempered by interpretability.

Outputs ``reports/clustering.json`` and ``reports/figures/fig_cluster_profile.png``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Leak-free clustering features (numeric + categorical).
NUMERIC_FEATURES = ["LOG_FLOOR_AREA", "CONSTRUCTION_AGE_NUM",
                    "NUMBER_HABITABLE_ROOMS", "MULTI_GLAZE_PROPORTION"]
CATEGORICAL_FEATURES = ["PROPERTY_TYPE", "BUILT_FORM", "MAIN_FUEL", "WALL_TYPE"]


def wall_type(df: pd.DataFrame) -> pd.Series:
    """Collapse WALLS_DESCRIPTION into a compact, policy-relevant category."""
    w = df.get("WALLS_DESCRIPTION", pd.Series("", index=df.index)).fillna("").str.lower()
    out = pd.Series("other", index=df.index, dtype="object")
    out[w.str.contains("cavity")] = "cavity"
    out[w.str.contains("solid brick")] = "solid_brick"
    out[w.str.contains("granite|sandstone|stone")] = "stone"
    out[w.str.contains("timber|system|cob")] = "timber_system"
    return out


def build_cluster_frame(eng: pd.DataFrame) -> pd.DataFrame:
    """Select the leak-free mixed-type feature frame, drop incomplete rows."""
    out = eng.copy()
    out["WALL_TYPE"] = wall_type(out)
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    out = out[cols + ["CURRENT_ENERGY_EFFICIENCY", "CURRENT_ENERGY_RATING"]].copy()
    for c in NUMERIC_FEATURES:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    out["CURRENT_ENERGY_EFFICIENCY"] = pd.to_numeric(
        out["CURRENT_ENERGY_EFFICIENCY"], errors="coerce").astype("float64")
    out = out.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES).reset_index(drop=True)
    for c in CATEGORICAL_FEATURES:
        out[c] = out[c].astype(str)
    return out


def gower_matrix(num: np.ndarray, cat: np.ndarray) -> np.ndarray:
    """Gower distance for a (small) mixed-type sample.

    Numerics: |a-b| normalised by feature range. Categoricals: 0/1 mismatch.
    Returns the mean across all features (equal weighting).
    """
    n = num.shape[0]
    if n > 5000:
        raise ValueError(
            f"gower_matrix is O(n^2) in memory ({n}x{n} = "
            f"{n*n*8/1e9:.1f} GB); pass a sample <= 5000 rows, not {n}.")
    ranges = np.ptp(num, axis=0)
    ranges[ranges == 0] = 1.0
    p = num.shape[1] + cat.shape[1]
    D = np.zeros((n, n), dtype="float64")
    for j in range(num.shape[1]):
        col = num[:, j][:, None]
        D += np.abs(col - col.T) / ranges[j]
    for j in range(cat.shape[1]):
        col = cat[:, j][:, None]
        D += (col != col.T).astype("float64")
    return D / p
