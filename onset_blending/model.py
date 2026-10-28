"""The fitted multinomial blend model.

`fit_final` is a port of 3_fit_final_model.py (the final fit on every training
year) and `FittedModel.predict` of apply_blend_model.py's prediction path,
with identical numerics (phase 2 of docs/pipeline-redesign.md). A model
converts to and from the coefficient bundle those scripts pickle, and to
plain arrays and JSON-able metadata.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import patsy
from sklearn.preprocessing import StandardScaler

from ._legacy.blend_evaluation_utils import (
    _make_multinom_clf,
    apply_formula_sample_support,
    build_formulas_from_spec,
    restrict_to_allowed,
)
from ._legacy.misc import interval_bins
from .evaluation import RunSettings, log_sample_support, prepare_inputs

__all__ = ["FittedModel", "design_matrix", "fit", "fit_final"]

log = logging.getLogger(__name__)

_SCALER_ARRAYS = ("scaler_mean", "scaler_scale", "scaler_var")


@dataclass(frozen=True, eq=False)
class FittedModel:
    """A multinomial logistic model on standardised formula features.

    Without a scaler (coefficient pickles from before the bundle format),
    the coefficients apply to the unscaled features.
    """

    formula: str  # the expanded formula that was fitted
    features: tuple[str, ...]  # design-matrix columns, in coefficient order
    classes: tuple[str, ...]  # outcome classes, in coefficient order
    coef: np.ndarray  # (n_classes, n_features)
    intercept: np.ndarray  # (n_classes,)
    scaler_mean: np.ndarray | None = None  # (n_features,)
    scaler_scale: np.ndarray | None = None
    scaler_var: np.ndarray | None = None  # kept so the legacy scaler is rebuilt exactly
    n_samples: int | None = None  # rows the scaler was fitted on

    def __post_init__(self):
        set_ = object.__setattr__
        set_(self, "features", tuple(self.features))
        set_(self, "classes", tuple(self.classes))
        n_classes, n_features = len(self.classes), len(self.features)
        shapes = {"coef": (n_classes, n_features), "intercept": (n_classes,)}
        has_scaler = self.scaler_mean is not None
        if has_scaler:
            shapes.update({name: (n_features,) for name in _SCALER_ARRAYS})
        elif any(getattr(self, name) is not None for name in _SCALER_ARRAYS):
            raise ValueError("scaler_mean, scaler_scale and scaler_var go together.")
        for name, shape in shapes.items():
            value = getattr(self, name)
            if value is None or np.shape(value) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {np.shape(value)}.")
            # C order, like apply_blend_model.py's coefficient matrix, so that
            # predict's matrix product runs the same BLAS kernel bit for bit.
            array = np.array(value, dtype=float, order="C")
            array.flags.writeable = False
            set_(self, name, array)
        if has_scaler != (self.n_samples is not None):
            raise ValueError("n_samples is required with, and only with, a scaler.")
        if has_scaler:
            set_(self, "n_samples", int(self.n_samples))

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Class probabilities `cv_<class>` for each row of `frame`, on its index.

        Rows with a non-finite feature get NaN. As apply_blend_model.py: the
        design is built from the saved formula and must have exactly the
        saved features, at least one of them finite.
        """
        features = list(self.features)
        design = design_matrix(self.formula, frame)
        missing = [f for f in features if f not in design]
        unexpected = [f for f in design if f not in features]
        if missing or unexpected:
            raise ValueError(
                "Applied design matrix does not match the saved training schema. "
                f"Missing: {missing}; unexpected: {unexpected}."
            )
        design = design.loc[:, features]
        all_nonfinite = [
            column
            for column in features
            if not np.isfinite(
                pd.to_numeric(design[column], errors="coerce").to_numpy(dtype=float)
            ).any()
        ]
        if all_nonfinite:
            raise ValueError(
                "Saved model features have no finite values in the application "
                f"data: {all_nonfinite}."
            )
        if self.scaler_mean is not None:
            design = pd.DataFrame(
                self._scaler().transform(design.values.astype(float)),
                columns=features,
                index=design.index,
            )

        X = design[features].values.astype(float)
        scores = X @ self.coef.T + self.intercept
        bad = ~np.isfinite(X).all(axis=1)
        probs = np.full((len(X), len(self.classes)), np.nan)
        if (~bad).any():
            s = scores[~bad]
            s -= s.max(axis=1, keepdims=True)
            exp_s = np.exp(s)
            probs[~bad] = exp_s / exp_s.sum(axis=1, keepdims=True)
        return pd.DataFrame(probs, columns=[f"cv_{c}" for c in self.classes], index=design.index)

    def to_legacy_bundle(self) -> dict:
        """The coefficient bundle 3_fit_final_model.py pickles: coefs, scaler,
        features, formula."""
        rows = [
            {
                "class": cls,
                "feature": feature,
                "coefficient": float(self.coef[i, j]),
                "intercept": float(self.intercept[i]),
            }
            for i, cls in enumerate(self.classes)
            for j, feature in enumerate(self.features)
        ]
        return {
            "coefs": pd.DataFrame(rows),
            "scaler": self._scaler() if self.scaler_mean is not None else None,
            "features": list(self.features),
            "formula": self.formula,
        }

    @classmethod
    def from_legacy_bundle(cls, bundle: dict, formula: str | None = None) -> FittedModel:
        """A model from a final or per-year coefficient bundle.

        `formula` is used only when the bundle has none (older bundles); as
        in apply_blend_model.py, classes are the sorted bundle classes and a
        feature missing from a class's coefficients counts as 0.
        """
        formula = bundle.get("formula") or formula
        if not formula:
            raise ValueError(
                "The coefficient bundle has no saved formula; pass the formula it was fitted with."
            )
        coef_df = bundle["coefs"]
        features = list(bundle["features"])
        classes = sorted(coef_df["class"].unique())
        coef = np.zeros((len(classes), len(features)))
        intercept = np.zeros(len(classes))
        for i, cls_ in enumerate(classes):
            cls_rows = coef_df[coef_df["class"] == cls_]
            intercept[i] = cls_rows["intercept"].iloc[0]
            for j, feature in enumerate(features):
                feature_row = cls_rows[cls_rows["feature"] == feature]
                if feature_row.empty:
                    warnings.warn(
                        f"Feature '{feature}' not found in coefs for class '{cls_}' — using 0"
                    )
                else:
                    coef[i, j] = float(feature_row["coefficient"].iloc[0])

        scaler = bundle.get("scaler")
        stats = {}
        if scaler is not None:
            if not (scaler.with_mean and scaler.with_std):
                raise ValueError("Only a centring and scaling StandardScaler is supported.")
            stats = dict(
                scaler_mean=scaler.mean_,
                scaler_scale=scaler.scale_,
                scaler_var=scaler.var_,
                n_samples=int(scaler.n_samples_seen_),
            )
        return cls(formula, features, classes, coef, intercept, **stats)

    def to_arrays(self) -> tuple[dict, dict[str, np.ndarray]]:
        """(JSON-able metadata, named float arrays): the model without pickles."""
        meta = {
            "formula": self.formula,
            "features": list(self.features),
            "classes": list(self.classes),
            "n_samples": self.n_samples,
        }
        names = ("coef", "intercept") + (_SCALER_ARRAYS if self.scaler_mean is not None else ())
        return meta, {name: getattr(self, name) for name in names}

    @classmethod
    def from_arrays(cls, meta: dict, arrays: dict[str, np.ndarray]) -> FittedModel:
        """The model `to_arrays` describes."""
        return cls(
            formula=meta["formula"],
            features=meta["features"],
            classes=meta["classes"],
            n_samples=meta["n_samples"],
            **{name: arrays.get(name) for name in ("coef", "intercept", *_SCALER_ARRAYS)},
        )

    def _scaler(self) -> StandardScaler:
        """The fitted StandardScaler these statistics came from."""
        # Fitting a placeholder gives sklearn's own attribute set, order and
        # types; the statistics are then swapped in.
        scaler = StandardScaler().fit(np.zeros((1, len(self.features))))
        scaler.n_samples_seen_ = type(scaler.n_samples_seen_)(self.n_samples)
        scaler.mean_ = self.scaler_mean.copy()
        scaler.var_ = self.scaler_var.copy()
        scaler.scale_ = self.scaler_scale.copy()
        return scaler


def design_matrix(formula: str, data: pd.DataFrame) -> pd.DataFrame:
    """The formula's right-hand side as a Patsy design matrix, without the
    intercept column and keeping rows with missing values."""
    rhs = formula.split("~", 1)[1].strip() if "~" in formula else formula
    return patsy.dmatrix(
        rhs, data, return_type="dataframe", NA_action=patsy.NAAction(NA_types=[])
    ).drop(columns=["Intercept"], errors="ignore")


def fit(formula: str, train: pd.DataFrame) -> FittedModel:
    """Fit the formula on the rows with a known outcome and finite features."""
    design = design_matrix(formula, train)
    features = list(design.columns)
    y = train["outcome"].values
    valid = np.array([isinstance(v, str) for v in y])
    X = design.values.astype(float)
    mask = valid & np.isfinite(X).all(axis=1)
    X_fit, y_fit = X[mask], y[mask]
    if len(np.unique(y_fit)) < 2:
        raise ValueError("Fewer than 2 outcome classes in training data — cannot fit model.")
    log.info("  Training rows used : %d / %d", mask.sum(), len(train))
    log.info("  Outcome classes    : %s", sorted(np.unique(y_fit)))
    log.info("  Feature columns    : %d", len(features))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_fit)
    clf = _make_multinom_clf()
    clf.fit(X_scaled, y_fit)
    return FittedModel(
        formula=formula,
        features=features,
        classes=clf.classes_,
        coef=clf.coef_,
        intercept=clf.intercept_,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        scaler_var=scaler.var_,
        n_samples=int(scaler.n_samples_seen_),
    )


def fit_final(
    weekly: pd.DataFrame,
    cv_spec: dict,
    dissemination: pd.DataFrame,
    model_name: str = "blended_model",
) -> FittedModel:
    """Fit `model_name`'s formula on every training year that is not a true
    holdout, on the dissemination cells, after the evaluation's sample support.

    `dissemination` is the table as read from its CSV (see `prepare_inputs`).
    """
    run = RunSettings.from_spec(cv_spec)
    weekly, cells, n_bins = prepare_inputs(weekly, cv_spec, dissemination)
    formulas = build_formulas_from_spec(cv_spec, run.cutoff_mode, data=weekly, n_bins=n_bins)
    if model_name not in formulas:
        raise ValueError(
            f"Model '{model_name}' not found in spec formulas. Available: {list(formulas)}"
        )
    weekly, support = apply_formula_sample_support(weekly, formulas)
    log_sample_support(support)

    allowed = restrict_to_allowed(weekly, cells)
    fit_years = run.fit_years
    train = allowed[allowed["year"].isin(fit_years)].copy()
    if train.empty:
        raise ValueError(
            "Final fit has no rows in the configured training years and dissemination cells."
        )
    present = {value for value in train["outcome"].unique() if isinstance(value, str)}
    missing = [value for value in interval_bins(n_bins) if value not in present]
    if missing:
        raise ValueError(
            "Final-fit training rows are missing outcome classes: " + ", ".join(missing)
        )
    log.info("  Training years        : %s", sorted(fit_years))
    log.info("  Training rows (allowed cells) : %d", len(train))
    log.info("\nFormula : %s", formulas[model_name])
    log.info("\nFitting %s on all %d training years...", model_name, len(fit_years))
    return fit(formulas[model_name], train)
