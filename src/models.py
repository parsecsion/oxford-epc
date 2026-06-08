"""Estimator and ColumnTransformer factories.

The factories return *unfit* objects so they can be embedded inside an
``sklearn.Pipeline`` and fitted only on the training fold. This is what
prevents preprocessing leakage between train and test.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

from . import SEED


# ----------------------------------------------------------------------
# Column type inference
# ----------------------------------------------------------------------

#: Categorical columns we explicitly know are ordinal (preserve order).
ORDINAL_FEATURE_LEVELS: dict[str, list[str]] = {
    "GLAZED_TYPE": ["single", "double", "triple", "secondary glazing", "INVALID!", "not defined"],
}


def infer_column_groups(X: pd.DataFrame) -> dict[str, list[str]]:
    numeric, low_card, high_card = [], [], []
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s):
            numeric.append(c)
        else:
            n = s.astype("string").nunique(dropna=True)
            (low_card if n <= 25 else high_card).append(c)
    return {"numeric": numeric, "low_card": low_card, "high_card": high_card}


# ----------------------------------------------------------------------
# Preprocessor
# ----------------------------------------------------------------------

def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    groups = infer_column_groups(X)

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
    ])
    low_card_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False)),
    ])
    high_card_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # Hash-style: ordinal-encode for tree models; trees handle this fine,
        # and it keeps the matrix dense and small.
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, groups["numeric"]),
            ("lo", low_card_pipe, groups["low_card"]),
            ("hi", high_card_pipe, groups["high_card"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ----------------------------------------------------------------------
# Estimator factories
# ----------------------------------------------------------------------

def logistic_baseline(class_weight: str | None = "balanced") -> LogisticRegression:
    return LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight=class_weight,
        random_state=SEED,
    )


def rf_classifier(n_estimators: int = 400) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=SEED,
        min_samples_leaf=2,
    )


def hist_gradient_boosting(max_iter: int = 400, **kw):
    """sklearn HistGradientBoostingClassifier with native NaN handling.

    Unlike RandomForest, this model handles missing values without explicit
    imputation: at each split, NaN samples are routed to whichever child
    minimises loss. With the ``add_missingness`` indicators already in the
    feature matrix, the model receives both the binary "is_missing" signal
    *and* an unimputed numeric column — strictly more information than
    median-imputed RF.

    Class imbalance is handled via ``class_weight='balanced'``, which
    reweights the per-sample loss by inverse class frequency.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    defaults = dict(
        max_iter=max_iter,
        learning_rate=0.07,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
    )
    defaults.update(kw)
    return HistGradientBoostingClassifier(**defaults)


def xgb_classifier(num_class: int = 7, **kw) -> "object":
    """Returns a configured XGBClassifier (lazy import)."""
    from xgboost import XGBClassifier
    defaults = dict(
        n_estimators=600, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        objective="multi:softprob", num_class=num_class,
        tree_method="hist", random_state=SEED, n_jobs=-1,
        eval_metric="mlogloss",
    )
    defaults.update(kw)
    return XGBClassifier(**defaults)


def lgbm_classifier(num_class: int = 7, class_weight: str | None = None,
                    **kw) -> "object":
    """LightGBM multiclass classifier.

    ``class_weight='balanced'`` re-weights the loss so minority bands (A, F, G)
    receive proportionally higher gradient — improves macro-F1 at the cost of
    a small QWK reduction. Without it, the model collapses toward the modal
    bands C and D.
    """
    from lightgbm import LGBMClassifier
    defaults = dict(
        n_estimators=800, learning_rate=0.05, max_depth=-1, num_leaves=63,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        objective="multiclass", num_class=num_class,
        random_state=SEED, n_jobs=-1, verbose=-1,
        class_weight=class_weight,
    )
    defaults.update(kw)
    return LGBMClassifier(**defaults)


def xgb_regressor(**kw) -> "object":
    from xgboost import XGBRegressor
    defaults = dict(
        n_estimators=800, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        objective="reg:squarederror", tree_method="hist",
        random_state=SEED, n_jobs=-1,
    )
    defaults.update(kw)
    return XGBRegressor(**defaults)


# ----------------------------------------------------------------------
# Pipeline assembly
# ----------------------------------------------------------------------

def build_pipeline(estimator, X: pd.DataFrame) -> Pipeline:
    return Pipeline([
        ("prep", make_preprocessor(X)),
        ("clf", estimator),
    ])


# ----------------------------------------------------------------------
# Champion: regression-to-band + SAP-stratified hybrid routing
# ----------------------------------------------------------------------

#: DLUHC official SAP-score → EPC band boundaries.
SAP_BAND_THRESHOLDS = (
    (92, "A"), (81, "B"), (69, "C"), (55, "D"),
    (39, "E"), (21, "F"), (0, "G"),
)


def sap_score_to_band(score: np.ndarray) -> np.ndarray:
    """Map SAP score → EPC band using fixed DLUHC thresholds.

    A: 92+, B: 81-91, C: 69-80, D: 55-68, E: 39-54, F: 21-38, G: 1-20.

    Validated against the cleaned dataset: 100% exact match between
    derived band and recorded ``CURRENT_ENERGY_RATING``.
    """
    raw = np.asarray(score, dtype=float)
    # SAP is defined on [1, 100]; a regressor can extrapolate beyond it, which
    # both produces an impossible score and inflates RMSE/MAE. Clip before
    # thresholding. NaN cannot occur from a fitted HGB but is guarded so a
    # stray NaN never yields an uninitialised (undefined) band.
    s = np.clip(raw, 1.0, 100.0)
    out = np.full(len(s), "C", dtype=object)   # neutral fallback for NaN cells
    finite = ~np.isnan(s)
    out[finite & (s >= 92)] = "A"
    out[finite & (s < 92) & (s >= 81)] = "B"
    out[finite & (s < 81) & (s >= 69)] = "C"
    out[finite & (s < 69) & (s >= 55)] = "D"
    out[finite & (s < 55) & (s >= 39)] = "E"
    out[finite & (s < 39) & (s >= 21)] = "F"
    out[finite & (s < 21)] = "G"
    return out


def regression_to_band_estimator(seed: int = 42):
    """HistGradientBoostingRegressor configured for the SAP-score target.

    Fit with ``CURRENT_ENERGY_EFFICIENCY`` as the target; predictions are
    continuous SAP scores in [0, 100+]. Use ``sap_score_to_band`` on the
    predictions to get an EPC band classification.

    CV-validated headline (``reports/cv_champions.json``): QWK = 0.8640 ±
    0.0006 (5-fold GroupKFold on the group-split training set). On the
    temporal holdout, wrapped in the hybrid ``SapStratifiedRegressor``,
    QWK = 0.7714 ± 0.0019 across seeds 42, 123, 2026
    (``reports/sap_stratified.json`` section ``4_seed_stability``).
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.07, max_leaf_nodes=63,
        min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=20, random_state=seed,
    )


def _fit_pipe_es(pipe, X, y, es_val_mask, sample_weight):
    """Fit a ``(prep -> reg)`` pipeline with TEMPORAL early stopping.

    HistGradientBoosting's built-in early stopping uses a *random* validation
    split, which (a) leaks UPRN groups across the split and (b) — more
    importantly for a temporally-split problem — validates on the *same*
    pre-cutoff distribution as training, so the model trains to the iteration
    limit and overfits that era, hurting the post-cutoff holdout. Here the
    early-stopping validation is the **latest slice of the training data by
    inspection date** (passed as ``es_val_mask``): the stopping criterion
    mirrors the real deployment task (predict the near future from the past),
    so the chosen iteration count generalises across the temporal shift.
    """
    prep = pipe.named_steps["prep"]
    reg = pipe.named_steps["reg"]
    Xt = prep.fit_transform(X)
    y_arr = np.asarray(y, dtype=float)
    vm = np.asarray(es_val_mask, dtype=bool)
    if vm.sum() < 50 or (~vm).sum() < 50:           # degenerate split -> no ES
        reg.set_params(early_stopping=False)
        reg.fit(Xt, y_arr, sample_weight=sample_weight)
        return pipe
    core = np.where(~vm)[0]; val = np.where(vm)[0]
    reg.set_params(early_stopping=False)
    sw_core = None if sample_weight is None else sample_weight[core]
    reg.fit(Xt[core], y_arr[core], sample_weight=sw_core)
    yv = y_arr[val]
    best_i, best_e = 20, np.inf
    for i, yp in enumerate(reg.staged_predict(Xt[val])):
        e = float(np.mean((yp - yv) ** 2))
        if e < best_e:
            best_e, best_i = e, i + 1
    # Refit on the FULL training set for the temporally-validated iteration count.
    reg.set_params(max_iter=max(int(best_i), 20))
    reg.fit(Xt, y_arr, sample_weight=sample_weight)
    return pipe


class SapStratifiedRegressor:
    """Hybrid regression-to-band model with SAP-cohort routing.

    Two underlying regressors:

    * ``unified``  – trained on the full training set; used for every
      row whose ``REPORT_TYPE`` is not 101 (and as the fallback when
      ``REPORT_TYPE`` is missing).
    * ``sap_only`` – trained on the ``REPORT_TYPE == 101`` (SAP) subset;
      used only for SAP-assessed test rows.

    Empirically (``reports/sap_stratified.json`` section ``3_hybrid``,
    produced by ``scripts/sap_stratified.py``) this is a *strict* Pareto
    improvement over the unified model on the temporal holdout:

    * Overall QWK: 0.7560 → 0.7696 (+0.0136)
    * RdSAP cohort QWK: 0.7059 (unchanged — uses the unified arm)
    * SAP cohort QWK: 0.5173 → 0.5782 (+0.0609)

    Multi-seed stability (seeds 42, 123, 2026, same JSON section
    ``4_seed_stability``): holdout QWK = 0.7714 ± 0.0019, range
    0.7696–0.7735. Std is comfortably below the 0.01 verifier threshold.

    The full stratification (``2_stratified``: separate model for every
    REPORT_TYPE) lifts overall QWK to 0.7677 but *regresses* the dominant
    RdSAP cohort (0.7059 → 0.6988, -0.0071) because the RdSAP-only model
    loses the multi-task benefit of the small SAP minority. Hybrid routing
    preserves the multi-task signal for the dominant cohort and adds a
    specialist only where it pays off.

    Caller is responsible for tracking ``REPORT_TYPE`` per row (it lives
    in the engineered feature matrix and must be passed alongside X to
    ``predict_band``).
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.unified = None  # sklearn Pipeline
        self.sap_only = None  # sklearn Pipeline

    def fit(self, pipe_unified, pipe_sap, X, y_sap_score, report_type,
            es_val_mask=None, sample_weight=None):
        """Fit both regressors.

        Parameters
        ----------
        pipe_unified, pipe_sap : sklearn.Pipeline
            Two pre-built pipelines (preprocessor + ``regression_to_band_estimator``).
        X : DataFrame
            Full training feature matrix.
        y_sap_score : Series
            Continuous SAP score target (``CURRENT_ENERGY_EFFICIENCY``).
        report_type : Series
            ``REPORT_TYPE`` aligned to X (100 = RdSAP, 101 = SAP).
        es_val_mask : array-like of bool, optional
            Per-row mask marking the **temporal early-stopping validation**
            (the latest training certificates by date). When given, both arms
            stop on this leak-free, deployment-aligned hold-out instead of the
            built-in random split. ``None`` preserves the legacy behaviour.
        sample_weight : array-like, optional
            Per-row weight. Pass ``1 / certs-per-UPRN`` to stop frequently
            re-assessed dwellings from dominating the loss.
        """
        sap_mask = (report_type == 101).values
        if sap_mask.sum() < 500:
            raise ValueError(
                f"SAP training subset too small ({int(sap_mask.sum())} rows); "
                "minimum 500 recommended for the SAP-only model."
            )
        X = X.reset_index(drop=True)
        y = y_sap_score.reset_index(drop=True)
        sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        idx = np.where(sap_mask)[0]
        if es_val_mask is None:
            pipe_unified.fit(X, y, **({} if sw is None else {"reg__sample_weight": sw}))
            pipe_sap.fit(X.iloc[idx], y.iloc[idx])
        else:
            vm = np.asarray(es_val_mask, dtype=bool)
            _fit_pipe_es(pipe_unified, X, y, vm, sw)
            _fit_pipe_es(
                pipe_sap, X.iloc[idx].reset_index(drop=True),
                y.iloc[idx].reset_index(drop=True), vm[idx],
                None if sw is None else sw[idx])
        self.unified = pipe_unified
        self.sap_only = pipe_sap
        return self

    def predict_score(self, X, report_type) -> np.ndarray:
        """Return continuous SAP-score predictions via hybrid routing."""
        if self.unified is None or self.sap_only is None:
            raise RuntimeError("Model not fitted; call fit() first.")
        score = self.unified.predict(X)
        sap_mask = (report_type == 101).values
        if sap_mask.sum() > 0:
            score[sap_mask] = self.sap_only.predict(X.loc[sap_mask])
        # SAP is bounded [1, 100]; clip extrapolated predictions so the score
        # is physically valid and RMSE/MAE are not inflated by impossible values.
        return np.clip(score, 1.0, 100.0)

    def predict_band(self, X, report_type) -> np.ndarray:
        """Return EPC band predictions via hybrid routing + DLUHC thresholds."""
        return sap_score_to_band(self.predict_score(X, report_type))


# ----------------------------------------------------------------------
# CORN ordinal NN (TensorFlow). Imports lazily so the rest of the project
# does not require TF to be installed.
# ----------------------------------------------------------------------

def corn_ordinal_nn(input_dim: int, num_classes: int = 7,
                    hidden: Sequence[int] = (128, 64),
                    dropout: float = 0.2):
    """Cao, Mirjalili & Raschka (2020) CORN-style ordinal classifier.

    Output: (num_classes - 1) sigmoid heads. Each head models
    P(Y > k | x) for k = 0..K-2. Predictions are obtained by chaining the
    sigmoids; this preserves rank consistency across thresholds.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inp = layers.Input(shape=(input_dim,))
    h = inp
    for n in hidden:
        h = layers.Dense(n, activation="relu")(h)
        h = layers.Dropout(dropout)(h)
    out = layers.Dense(num_classes - 1, activation="sigmoid", name="thresholds")(h)
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy")  # CORN target is binarised externally
    return model


def corn_to_class(probs: np.ndarray) -> np.ndarray:
    """Convert chained sigmoid outputs to a single class label (0..K-1)."""
    # Number of thresholds exceeded
    return (probs >= 0.5).sum(axis=1)
