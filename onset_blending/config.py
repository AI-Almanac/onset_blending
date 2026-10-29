"""BlendConfig: the one authoritative configuration of a blend.

It replaces the six stage specs (regrid, raw_data ground truth and forecasts,
combine, connect, cv_models). Each fact is stated once, organised by concept:

    domain        where: grid cells or polygon units, unit ids, dissemination
                  units, area of interest
    observations  the ground-truth rainfall files
    models        each forecast model once
    onset         the onset definition, threshold, season start, reference onset
    climatology   the KDE climatology
    features      weekly bins, probability days, joins, rain transform
    evaluation    years, CV, formulas, baselines, calibration, MME, externals
    outputs       what to write

Everything derivable (column names and prefixes, artifact names and tags,
rain-predictor windows, climatology year windows) is computed by the
functions at the bottom of this module rather than configured, so the
config cannot contradict itself. Unknown keys are rejected, and all
cross-section rules are checked up front by `BlendConfig.validate`.
"""

from __future__ import annotations

import dataclasses
import difflib
import re
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Union

import yaml

__all__ = [
    "BlendConfig",
    "ConfigError",
    "load_config",
]


class ConfigError(ValueError):
    """A configuration that cannot be used; lists every problem found."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__(
            "Invalid blend configuration:\n" + "\n".join(f"  - {p}" for p in self.problems)
        )


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------

_MONTH_DAY = re.compile(r"^(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$")


def _month_day(value: Any, path: str, problems: list[str]) -> str | None:
    text = str(value).strip()
    if not _MONTH_DAY.match(text):
        problems.append(f"{path}: expected a month-day like '05-01', got {value!r}")
        return None
    return text


def _years(value: Any, path: str, problems: list[str]) -> tuple[int, ...] | None:
    """Years from a list, an int, or a spec like '2000:2024' or '2000:2010,2015'."""
    if value is None:
        return ()
    if isinstance(value, bool):
        problems.append(f"{path}: expected years, got {value!r}")
        return None
    if isinstance(value, int):
        return (value,)
    if isinstance(value, (list, tuple)):
        years = []
        for item in value:
            parsed = _years(item, path, problems)
            if parsed is None:
                return None
            years.extend(parsed)
        return tuple(sorted(dict.fromkeys(years)))
    text = str(value).strip()
    years = []
    for part in filter(None, (p.strip() for p in text.split(","))):
        match = re.fullmatch(r"(\d{4})\s*(?::|\.\.|-|to)\s*(\d{4})", part)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            if end < start:
                problems.append(f"{path}: range {part!r} ends before it starts")
                return None
            years.extend(range(start, end + 1))
        elif re.fullmatch(r"\d{4}", part):
            years.append(int(part))
        else:
            problems.append(f"{path}: cannot read years from {part!r} (use 2000:2024 or a list)")
            return None
    return tuple(sorted(dict.fromkeys(years)))


def _year_window(value: Any, path: str, problems: list[str]) -> tuple[int, int] | None:
    """A (start, end) pair of years, from [2000, 2024] or '2000:2024'."""
    years = _years(value, path, problems)
    if years is None:
        return None
    if len(years) < 1:
        problems.append(f"{path}: expected a start and end year")
        return None
    return (years[0], years[-1])


def _year_windows(value: Any, path: str, problems: list[str]) -> tuple[tuple[int, int], ...] | None:
    """(start, end) year windows from [[2000, 2024], '1990:2020', ...]."""
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        value = [value]
    windows = []
    for index, item in enumerate(value):
        window = _year_window(item, f"{path}[{index}]", problems)
        if window is None:
            return None
        windows.append(window)
    return tuple(windows)


# Field metadata keys.
_PARSE = "parse"  # custom parser(value, path, problems)
_DOC = "doc"


def _field(default=dataclasses.MISSING, *, parse=None, doc: str = "", factory=None):
    metadata = {_DOC: doc}
    if parse is not None:
        metadata[_PARSE] = parse
    if factory is not None:
        return field(default_factory=factory, metadata=metadata)
    return field(default=default, metadata=metadata)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class FileSet:
    """Input NetCDF files: a folder and a file-name pattern, or explicit paths."""

    folder: str | None = _field(None, doc="Directory to search.")
    pattern: str = _field(r"\.nc$", doc="Regular expression searched in file names.")
    paths: tuple[str, ...] = _field(factory=tuple, doc="Explicit files instead of folder + pattern.")
    years: tuple[int, ...] = _field(
        factory=tuple, parse=_years, doc="Keep only files whose name holds one of these years."
    )

    def _check(self, path: str, problems: list[str]) -> None:
        if bool(self.folder) == bool(self.paths):
            problems.append(f"{path}: give either 'folder' (with 'pattern') or 'paths', not both")
        if self.folder:
            try:
                re.compile(self.pattern)
            except re.error as exc:
                problems.append(f"{path}.pattern: not a regular expression ({exc})")


@dataclass(frozen=True, kw_only=True)
class GridIds:
    """How lat/lon grid cells are named ("<lat>_<lon>")."""

    decimal_digits: int | None = _field(None, doc="Digits per coordinate; inferred when absent.")
    format: Literal["trimmed", "fixed"] = _field("trimmed", doc="Trim trailing zeros, or fixed digits.")


@dataclass(frozen=True, kw_only=True)
class UnitTable:
    """A CSV of units (dissemination units, centroids), keyed by an id column."""

    file: str = _field(doc="CSV path.")
    id_col: str | None = _field(None, doc="Unit-id column; inferred (id, adm3_name, lat/lon) when absent.")


@dataclass(frozen=True, kw_only=True)
class Regrid:
    """Aggregate gridded inputs onto the blend's units by area weights: polygon
    units (e.g. ADM3 districts) from a shapefile, or, without one, the
    observation grid's cells (forecasts on other grids are regridded to it)."""

    shapefile: str | None = _field(None, doc="Unit polygons; absent means the observation grid cells.")
    unit_id_col: str = _field("adm3_name", doc="Stable unit id column in the shapefile.")
    unit_name_col: str | None = _field(None, doc="Display-name column, kept in the centroids.")
    parent_id_col: str | None = _field(None, doc="Parent-unit column (checked to exist).")
    crs: str = _field("EPSG:4326", doc="CRS for overlays and areas.")
    grid_lat_var: str | None = _field(None, doc="Grid latitude coordinate name; inferred when absent.")
    grid_lon_var: str | None = _field(None, doc="Grid longitude coordinate name; inferred when absent.")
    clip_to_coverage: bool = _field(True, doc="Restrict and renormalise weights to observed cells.")
    weights: str | None = _field(None, doc="Supplied observation weights (lat, lon, unit, weight).")


@dataclass(frozen=True, kw_only=True)
class BBox:
    lat_min: float | None = None
    lat_max: float | None = None
    lon_min: float | None = None
    lon_max: float | None = None


@dataclass(frozen=True, kw_only=True)
class AreaOfInterest:
    """Restrict the blend to a box, or to the units inside GeoJSON outlines."""

    bbox: BBox | None = None
    geometry: str | None = _field(None, doc="GeoJSON file of outlines.")

    def _check(self, path: str, problems: list[str]) -> None:
        if (self.bbox is None) == (self.geometry is None):
            problems.append(f"{path}: give exactly one of 'bbox' or 'geometry'")


@dataclass(frozen=True, kw_only=True)
class Domain:
    grid_ids: GridIds = _field(factory=GridIds)
    regrid: Regrid | None = _field(None, doc="Polygon units; absent means grid cells are the units.")
    dissemination: UnitTable | None = _field(None, doc="Units the blend is for.")
    dissemination_scope: Literal["inputs", "training"] = _field(
        "inputs",
        doc="inputs: only these units are read and predicted; training: every unit is "
        "predicted, but the blend trains and is scored on these.",
    )
    centroids: UnitTable | None = _field(None, doc="Unit lat/lon (written by the regrid when absent).")
    area_of_interest: AreaOfInterest | None = None


@dataclass(frozen=True, kw_only=True)
class Observations:
    files: FileSet
    variable: str = _field(doc="Rainfall variable in the files.")
    dims: dict[str, str] = _field(factory=dict, doc="Rename file dimensions to time/lat/lon.")
    missing_rain: Literal["keep", "drop", "zero"] = _field(
        "keep", doc="keep: NaN stays; drop: drop any series with a NaN; zero: treat as 0 mm."
    )
    cell_transform: str | None = _field(None, doc="Source-to-unit weights CSV (source_id, target_id, weight).")
    date_marks: Literal["end", "start"] = _field(
        "end",
        doc="Which end of its 24-hour window an observation's date marks. end: the window "
        "ends on the date (IMD: the 24 h to 08:30 IST); start: the date is the day the "
        "rain fell (calendar-day products such as CHIRPS). Forecast lead day d marks the "
        "end of its window (the UTC day ending at issue + d), so start dates move a day "
        "later to align with it.",
    )


@dataclass(frozen=True, kw_only=True)
class LeadDays:
    min: int = 1
    max: int = 45


@dataclass(frozen=True, kw_only=True)
class RainHorizon:
    """How far forecast rainfall is trusted for rain predictors."""

    max_day: int | None = _field(None, doc="Last rainfall lead day used; absent means legacy (probability days + 10).")
    policy: Literal["legacy", "exact", "truncate"] | None = _field(
        None, doc="exact (default with max_day): reject later days; truncate: ignore them."
    )

    @property
    def resolved_policy(self) -> str:
        if self.policy is not None:
            return self.policy
        return "legacy" if self.max_day is None else "exact"


_WINDOW_TOKENS = ("trigger", "window", "dry_spell", "sum_window", "follow", "min_dry")


@dataclass(frozen=True, kw_only=True)
class RainPredictor:
    """A weekly rain feature: the max (or min, or excess over the onset
    threshold: diff) of rolling `window`-day rain sums."""

    agg: Literal["diff", "max", "min"]
    window: int | str = _field(
        "trigger", doc="Days, or a token from the onset definition: " + ", ".join(_WINDOW_TOKENS)
    )


@dataclass(frozen=True, kw_only=True)
class ForecastModel:
    name: str = _field(doc="Model name; prefixes every column derived from it.")
    files: FileSet = _field(doc="The model's forecast files.")
    variable: str = _field("tp", doc="Rainfall variable.")
    dims: dict[str, str] = _field(
        factory=dict,
        doc="Rename dims to time/day/lat/lon/number, for names a file's format does not "
        "already tell (core/io.py: aliases such as prediction_timedelta_daily, member).",
    )
    unit_scale: float = _field(
        1.0,
        doc="Multiply rainfall by this to get mm, for files that declare no units "
        "(declared units, e.g. 'm', decide the conversion).",
    )
    max_member: int | None = _field(None, doc="Keep members whose index is <= this.")
    lead_days: LeadDays = _field(factory=LeadDays, doc="Lead days read from the files.")
    rain_horizon: RainHorizon = _field(factory=RainHorizon)
    rain_predictors: tuple[RainPredictor, ...] = _field(factory=tuple)
    rain_transform: str | float | None = _field(None, doc="Overrides features.rain_transform.")
    cell_transform: str | None = _field(None, doc="Source-to-unit weights CSV for this model.")
    regrid_weights: str | None = _field(None, doc="Supplied regrid weights for this model's grid.")


@dataclass(frozen=True, kw_only=True)
class DrySpell:
    mode: Literal["consecutive_dry", "window_sum"] = "consecutive_dry"
    min_dry_days: int = _field(7, doc="consecutive_dry: dry days in a row that veto an onset.")
    dry_day_min_mm: float | None = _field(None, doc="Below this a day is dry; default wet_day_min_mm.")
    sum_window: int = _field(10, doc="window_sum: window length.")
    sum_min_mm: float = _field(5.0, doc="window_sum: a window below this vetoes an onset.")


@dataclass(frozen=True, kw_only=True)
class OnsetDefinition:
    window: int = _field(5, doc="Trigger window, days.")
    trigger_rule: Literal["all_days_wet", "first_day_wet"] = "all_days_wet"
    wet_day_min_mm: float = 1.0
    wet_day_comparison: Literal["gte", "gt"] = "gte"
    follow_days: int = _field(21, doc="Follow-up period checked for a dry spell.")
    followup_anchor: Literal["after_trigger", "onset_day"] = "after_trigger"
    dry_spell: DrySpell = _field(factory=DrySpell)


@dataclass(frozen=True, kw_only=True)
class ThresholdScale:
    factor: float = 1.0
    offset: float = 0.0
    min: float | None = None
    max: float | None = None


@dataclass(frozen=True, kw_only=True)
class Threshold:
    """Rain over the trigger window that counts as an onset: one value in mm,
    or a per-unit table (CSV, NetCDF or .mat)."""

    mm: float | None = None
    file: str | None = None
    id_col: str | None = _field(None, doc="CSV unit-id column (then adm3_name, id, lat/lon).")
    value_col: str = _field("onset_thresh", doc="CSV threshold column.")
    lat_col: str = "lat"
    lon_col: str = "lon"
    lat_var: str = _field("lat", doc="NetCDF latitude variable.")
    lon_var: str = _field("lon", doc="NetCDF longitude variable.")
    value_var: str = _field("MWmean", doc="NetCDF threshold variable.")
    mat_lat_name: str = "lat"
    mat_lon_name: str = "lon"
    mat_value_name: str = "onset.day.thres"
    scale: ThresholdScale | None = _field(None, doc="clip(t * factor + offset, min, max) for a table.")

    def _check(self, path: str, problems: list[str]) -> None:
        if (self.mm is None) == (self.file is None):
            problems.append(f"{path}: give exactly one of 'mm' or 'file'")
        if self.mm is not None and self.scale is not None:
            problems.append(f"{path}.scale: only applies to a threshold table")


@dataclass(frozen=True, kw_only=True)
class ReferenceOnset:
    """The reference (climatological) onset: one month-day every year, or a
    table of dates per year and/or unit."""

    month_day: str | None = _field(None, parse=_month_day)
    file: str | None = None
    date_col: str | None = _field(None, doc="Date column; otherwise year_col + day_col + base_date.")
    year_col: str | None = None
    day_col: str | None = None
    base_date: str | None = _field(None, parse=_month_day)
    unit_col: str | None = _field(None, doc="Per-unit dates: the unit-id column.")

    def _check(self, path: str, problems: list[str]) -> None:
        if (self.month_day is None) == (self.file is None):
            problems.append(f"{path}: give exactly one of 'month_day' or 'file'")
        if self.file is not None and self.date_col is None and not (
            self.year_col and self.day_col and self.base_date
        ):
            problems.append(f"{path}: a table needs 'date_col', or 'year_col', 'day_col' and 'base_date'")


@dataclass(frozen=True, kw_only=True)
class Onset:
    definition: OnsetDefinition = _field(factory=OnsetDefinition)
    threshold: Threshold = _field(doc="Required: without one no onset can occur.")
    season_start: str = _field(
        "05-01", parse=_month_day, doc="Onsets are searched from this date; onset days count from it."
    )
    fixed_cutoff: str = _field(
        "06-02", parse=_month_day, doc="Start of the fixed_cutoff forecast probabilities."
    )
    reference_onset: ReferenceOnset | None = None


@dataclass(frozen=True, kw_only=True)
class ClimatologyHorizon:
    start: str = _field(parse=_month_day)
    end: str = _field(parse=_month_day)
    forecast_window: int = 0


@dataclass(frozen=True, kw_only=True)
class Climatology:
    issue_end: str = _field("07-31", parse=_month_day, doc="Last issue date each season.")
    forecast_window: int | None = _field(None, doc="Lead days; default features.probability_days.")
    horizons: tuple[ClimatologyHorizon, ...] = _field(factory=tuple, doc="Lead days by issue date.")
    cv_by_year: bool = _field(True, doc="Leave-one-year-out KDE.")
    min_onset_years: int = _field(10, doc="Cells with fewer observed onsets are left out.")
    train_years: tuple[int, int] | None = _field(
        None, parse=_year_window, doc="Default: first observed year to the year before the first forecast year."
    )
    test_years: tuple[int, int] | None = _field(
        None, parse=_year_window, doc="Default: the forecast years."
    )
    windows: tuple[tuple[int, int], ...] = _field(
        factory=tuple, parse=_year_windows,
        doc="Extra climatologies trained on these (start, end) year windows.",
    )
    workers: int | None = _field(None, doc="Processes; default: the run's.")


@dataclass(frozen=True, kw_only=True)
class Bins:
    count: int = _field(4, doc="Week bins (outcomes week1..weekN, later).")
    days: int = _field(7, doc="Days per bin.")


@dataclass(frozen=True, kw_only=True)
class Features:
    bins: Bins = _field(factory=Bins)
    probability_days: int = _field(28, doc="Lead days of onset probability summed into the bins.")
    join: Literal["inner", "full"] = _field(
        "inner", doc="inner: rows every source has; full: all rows."
    )
    trim_after_true_onset: bool = _field(True, doc="Drop issue dates after the observed onset.")
    rain_transform: str | float = _field("identity", doc="identity, sqrt, fourth_root, log1p, power:<p>.")


@dataclass(frozen=True, kw_only=True)
class Years:
    training: tuple[int, ...] = _field(parse=_years)
    cv_holdout: tuple[int, ...] = _field(parse=_years)
    true_holdout: tuple[int, ...] = _field(factory=tuple, parse=_years)

    @property
    def holdout(self) -> tuple[int, ...]:
        """Every year scored: true holdouts first, as the evaluation orders them."""
        return tuple(dict.fromkeys([*self.true_holdout, *self.cv_holdout]))


@dataclass(frozen=True, kw_only=True)
class Formula:
    text: str
    enabled: bool = True


@dataclass(frozen=True, kw_only=True)
class WindowVariant:
    """Formula variants that swap a feature for its windowed-climatology version."""

    base: str = _field(doc="Formula to vary.")
    windows: tuple[tuple[int, int], ...] = _field(
        factory=tuple, parse=_year_windows, doc="(start, end) climatology windows."
    )
    replace: str = _field("prob_clim", doc="Feature replaced by its windowed version.")
    only_if_cutoff_mode: str | None = None


@dataclass(frozen=True, kw_only=True)
class ClimatologyBaselines:
    unconditional: bool = _field(True, doc="Score unc_clim_raw (the skill baseline).")
    conditional: bool = _field(True, doc="Score clim_raw.")
    windows: tuple[tuple[int, int], ...] = _field(
        factory=tuple, parse=_year_windows, doc="Also score windowed clim_raw."
    )


@dataclass(frozen=True, kw_only=True)
class ForecastEvaluation:
    model: str
    variant: Literal["base", "fixed_cutoff", "ref_onset"] | None = _field(
        None, doc="Probability series scored; default the cutoff mode's."
    )
    raw: bool = True
    calibrated: bool = True
    platt_weights: bool = _field(False, doc="Also export Platt weights fitted on the training years.")


@dataclass(frozen=True, kw_only=True)
class MmeSource:
    name: str = _field(doc="A climatology baseline (clim_raw...) or a model.")
    source: Literal["clim", "forecast"] = "clim"
    cal_variant: Literal["base", "fixed_cutoff", "ref_onset"] = "base"


@dataclass(frozen=True, kw_only=True)
class RandomForest:
    """The CRPS random forest blend (blend.random_forest): daily onset
    probabilities from each model's daily rainfall and the climatology."""

    name: str = _field("blended_forest", doc="Its name in the scores.")
    trees: int = _field(100, doc="Trees in the forest.")
    min_samples_leaf: int = _field(20, doc="Fewest training samples a leaf may hold.")
    max_depth: int | None = _field(None, doc="Deepest a tree may grow; unset for no limit.")
    max_samples: float = _field(1.0, doc="Share of the training samples each tree's bootstrap draws.")
    max_features: Literal["sqrt", "all"] = _field("sqrt", doc="Features tried at each split: the square root of their number, or all.")
    random_state: int = _field(123, doc="Seed; CV fold i uses random_state + i + 1.")
    max_day: int = _field(28, doc="Lead days with their own probability (the week bins' days); later ones pool.")
    missing_onset: Literal["after_week4", "drop"] = _field(
        "after_week4", doc="Seasons without an onset: a lead day past max_day, or left out."
    )
    years_per_fold: int = _field(1, doc="CV holdout years held out together per fold, in contiguous blocks.")


@dataclass(frozen=True, kw_only=True)
class Mme:
    """A multi-model ensemble with RPS-optimised weights."""

    enabled: bool = False
    variants: tuple[str, ...] = _field(("fixed_cutoff",), doc="Labels the MME runs carry.")
    sources: tuple[MmeSource, ...] = _field(factory=tuple)
    optimize: Literal["auto", "always", "never"] = _field(
        "auto", doc="auto: optimise only when the holdout years are 2000-2024; else read saved weights."
    )
    weights_year_tag: str | None = _field(None, doc="Tag of the saved weights to read.")


@dataclass(frozen=True, kw_only=True)
class ExternalPrediction:
    name: str
    file: str = _field(doc="Pickle of cv_* predictions (relative to the outputs dir).")
    method: str = "global"


@dataclass(frozen=True, kw_only=True)
class Evaluation:
    years: Years
    cutoff_mode: Literal["fixed_cutoff", "ref_onset", "no_ref_filter"] = "fixed_cutoff"
    cv_methods: tuple[Literal["global", "local", "clusters"], ...] = ("global",)
    cluster_var: str = _field("cluster", doc="Grouping column for clusters CV.")
    formulas: dict[str, Formula] = _field(factory=dict)
    final_formula: str | None = _field(
        None, doc="The formula fitted on every training year for operations; default the only enabled one."
    )
    window_variants: tuple[WindowVariant, ...] = _field(factory=tuple)
    climatology_baselines: ClimatologyBaselines = _field(factory=ClimatologyBaselines)
    forecasts: tuple[ForecastEvaluation, ...] | None = _field(
        None, doc="Raw/calibrated scoring of each model; default every model, both."
    )
    mme: Mme = _field(factory=Mme)
    external_predictions: tuple[ExternalPrediction, ...] = _field(factory=tuple)
    random_forest: RandomForest | None = _field(None, doc="Also blend with the CRPS random forest.")


@dataclass(frozen=True, kw_only=True)
class Outputs:
    dir: str | None = None
    intermediates: tuple[
        Literal["observations", "forecasts", "climatology", "combined", "features"], ...
    ] = _field(factory=tuple, doc="Stage results to save for debugging.")
    pickles: bool = _field(False, doc="Also write .pkl twins of the CSV tables.")


@dataclass(frozen=True, kw_only=True)
class BlendConfig:
    # Fields in the order a config file reads best (kw_only allows required
    # fields after defaulted ones).
    name: str
    data_root: str | None = _field(
        None,
        doc="Relative paths resolve against this directory; from a file, it is itself "
        "relative to the file, which is the default root. From a dict: the working directory.",
    )
    domain: Domain = _field(factory=Domain)
    observations: Observations
    models: tuple[ForecastModel, ...]
    onset: Onset
    climatology: Climatology = _field(factory=Climatology)
    features: Features = _field(factory=Features)
    evaluation: Evaluation
    outputs: Outputs = _field(factory=Outputs)

    # -- loading -----------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict) -> "BlendConfig":
        problems: list[str] = []
        config = _build(cls, data, "", problems)
        if problems:
            raise ConfigError(problems)
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BlendConfig":
        path = Path(path)
        with open(path) as handle:
            data = yaml.safe_load(handle) or {}
        if isinstance(data, dict):
            root = path.parent.resolve() / (data.get("data_root") or ".")
            data = {**data, "data_root": str(root.resolve())}
        return cls.from_dict(data)

    def path(self, value: str | Path) -> Path:
        """A configured path, resolved against data_root."""
        value = Path(value)
        if value.is_absolute() or self.data_root is None:
            return value
        return Path(self.data_root) / value

    def to_dict(self) -> dict:
        return _dump(self)

    def to_yaml(self, path: str | Path | None = None) -> str:
        text = yaml.safe_dump(self.to_dict(), sort_keys=False)
        if path is not None:
            Path(path).write_text(text)
        return text

    # -- lookups -----------------------------------------------------------

    def model(self, name: str) -> ForecastModel:
        for model in self.models:
            if model.name == name:
                return model
        raise KeyError(name)

    @property
    def model_names(self) -> tuple[str, ...]:
        return tuple(model.name for model in self.models)

    def forecast_evaluations(self) -> tuple[ForecastEvaluation, ...]:
        """The configured raw/calibrated scoring, or every model's by default."""
        if self.evaluation.forecasts is not None:
            return self.evaluation.forecasts
        return tuple(ForecastEvaluation(model=name) for name in self.model_names)

    def final_formula(self) -> str:
        """The formula the operational model is fitted with."""
        if self.evaluation.final_formula is not None:
            return self.evaluation.final_formula
        return next(name for name, f in self.evaluation.formulas.items() if f.enabled)

    def climatology_forecast_window(self) -> int:
        if self.climatology.forecast_window is not None:
            return self.climatology.forecast_window
        return self.features.probability_days

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        """Every rule that ties sections together; raises ConfigError."""
        problems: list[str] = []
        _check_tree(self, "", problems)

        names = self.model_names
        if not names:
            problems.append("models: at least one forecast model is required")
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            problems.append(f"models: names must be unique (repeated: {', '.join(duplicates)})")
        for name in names:
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(_[A-Za-z0-9]+)*", name):
                problems.append(
                    f"models: name {name!r} must be letters, digits and single underscores"
                )

        features = self.features
        bins = features.bins
        if bins.count < 1 or bins.days < 1:
            problems.append("features.bins: count and days must be positive")
        elif bins.count * bins.days > features.probability_days:
            problems.append(
                f"features.bins: {bins.count} bins of {bins.days} days need "
                f"{bins.count * bins.days} probability days, but features.probability_days "
                f"is {features.probability_days}"
            )
        _check_rain_transform(features.rain_transform, "features.rain_transform", problems)

        for index, model in enumerate(self.models):
            where = f"models[{index}] ({model.name})"
            lead = model.lead_days
            if lead.min < 0 or lead.max < lead.min:
                problems.append(f"{where}.lead_days: need 0 <= min <= max")
            if lead.max < features.probability_days:
                problems.append(
                    f"{where}.lead_days.max is {lead.max}, below features.probability_days "
                    f"({features.probability_days})"
                )
            horizon = model.rain_horizon
            if horizon.max_day is not None:
                if horizon.max_day <= 0:
                    problems.append(f"{where}.rain_horizon.max_day must be positive")
                if horizon.policy == "legacy":
                    problems.append(f"{where}.rain_horizon: 'legacy' cannot have a max_day")
                if lead.min != 1:
                    problems.append(f"{where}: a rain_horizon.max_day needs lead_days.min of 1")
                if lead.max < horizon.max_day:
                    problems.append(
                        f"{where}.lead_days.max ({lead.max}) is below rain_horizon.max_day "
                        f"({horizon.max_day})"
                    )
            elif horizon.policy not in (None, "legacy"):
                problems.append(f"{where}.rain_horizon: policy {horizon.policy!r} needs a max_day")
            elif model.rain_predictors and lead.max < features.probability_days + 10:
                problems.append(
                    f"{where}: rain predictors without a rain_horizon.max_day read "
                    f"{features.probability_days + 10} lead days (probability_days + 10), "
                    f"but lead_days.max is {lead.max}"
                )
            diff_windows = [p.window for p in model.rain_predictors if p.agg == "diff"]
            if len(diff_windows) > 1:
                problems.append(
                    f"{where}.rain_predictors: one 'diff' predictor per model "
                    "(their columns share one name)"
                )
            for p_index, predictor in enumerate(model.rain_predictors):
                window = predictor.window
                if isinstance(window, str) and window.lower() not in _WINDOW_TOKENS:
                    problems.append(
                        f"{where}.rain_predictors[{p_index}].window: {window!r} is not a number "
                        f"or one of {', '.join(_WINDOW_TOKENS)}"
                    )
                elif isinstance(window, int) and window < 1:
                    problems.append(f"{where}.rain_predictors[{p_index}].window must be >= 1")
            if model.rain_transform is not None:
                _check_rain_transform(model.rain_transform, f"{where}.rain_transform", problems)

        climatology = self.climatology
        if climatology.horizons and climatology.forecast_window is not None:
            problems.append("climatology: give 'forecast_window' or 'horizons', not both")
        if not climatology.horizons and self.climatology_forecast_window() < features.probability_days:
            problems.append(
                f"climatology.forecast_window ({self.climatology_forecast_window()}) is below "
                f"features.probability_days ({features.probability_days})"
            )
        for window in climatology.windows:
            if window[1] < window[0]:
                problems.append(f"climatology.windows: {window} ends before it starts")

        evaluation = self.evaluation
        years = evaluation.years
        if not years.training:
            problems.append("evaluation.years.training is empty")
        if not years.holdout:
            problems.append("evaluation.years: no cv_holdout or true_holdout years to score")
        if evaluation.cutoff_mode == "ref_onset" and self.onset.reference_onset is None:
            problems.append("evaluation.cutoff_mode 'ref_onset' needs onset.reference_onset")
        forest = evaluation.random_forest
        if forest is not None:
            bins = self.features.bins
            if forest.max_day != bins.count * bins.days:
                problems.append(
                    f"evaluation.random_forest.max_day must span the week bins ({bins.count} x {bins.days} days)"
                )
            if forest.trees < 1 or forest.min_samples_leaf < 1 or forest.years_per_fold < 1:
                problems.append("evaluation.random_forest: trees, min_samples_leaf and years_per_fold must be at least 1")
            if not 0.0 < forest.max_samples <= 1.0:
                problems.append("evaluation.random_forest.max_samples must be in (0, 1]")
        enabled = {name: f for name, f in evaluation.formulas.items() if f.enabled}
        if not enabled:
            problems.append("evaluation.formulas: at least one enabled formula is required")
        if evaluation.final_formula is not None and evaluation.final_formula not in enabled:
            problems.append(f"evaluation.final_formula: {evaluation.final_formula!r} is not an enabled formula")
        elif evaluation.final_formula is None and len(enabled) > 1:
            problems.append(
                "evaluation.final_formula: name the formula operations uses "
                f"(one of {', '.join(sorted(enabled))})"
            )
        available = feature_bases(self)
        for name, formula in enabled.items():
            for token in formula_terms(formula.text):
                if token not in available:
                    close = difflib.get_close_matches(token, sorted(available), n=2)
                    hint = f" (did you mean {' or '.join(close)}?)" if close else ""
                    problems.append(
                        f"evaluation.formulas.{name}: {token!r} is not a feature this "
                        f"config builds{hint}"
                    )
        for v_index, variant in enumerate(evaluation.window_variants):
            if variant.base not in enabled:
                problems.append(
                    f"evaluation.window_variants[{v_index}].base: {variant.base!r} is not an "
                    "enabled formula"
                )
            for window in variant.windows:
                if tuple(window) not in climatology.windows:
                    problems.append(
                        f"evaluation.window_variants[{v_index}]: window {window} is not in "
                        "climatology.windows"
                    )
        for window in evaluation.climatology_baselines.windows:
            if tuple(window) not in climatology.windows:
                problems.append(
                    f"evaluation.climatology_baselines.windows: {window} is not in climatology.windows"
                )
        evaluated = set()
        for f_index, scoring in enumerate(self.forecast_evaluations()):
            if scoring.model not in names:
                problems.append(f"evaluation.forecasts[{f_index}].model: {scoring.model!r} is not a model")
            evaluated.add((scoring.model, scoring.variant or evaluation.cutoff_mode, scoring.calibrated))
        mme = evaluation.mme
        if mme.enabled:
            if len(mme.sources) < 2:
                problems.append("evaluation.mme: an ensemble needs at least two sources")
            baselines = set(climatology_baseline_names(self))
            for s_index, source in enumerate(mme.sources):
                if source.source == "clim" and source.name not in baselines:
                    problems.append(
                        f"evaluation.mme.sources[{s_index}]: {source.name!r} is not a scored "
                        f"climatology baseline ({', '.join(sorted(baselines))})"
                    )
                if source.source == "forecast" and (source.name, source.cal_variant, True) not in evaluated:
                    problems.append(
                        f"evaluation.mme.sources[{s_index}]: needs evaluation.forecasts to calibrate "
                        f"{source.name!r} ({source.cal_variant})"
                    )
        if not evaluation.climatology_baselines.unconditional:
            problems.append(
                "evaluation.climatology_baselines.unconditional: unc_clim_raw is the baseline "
                "every skill score is measured against and cannot be turned off"
            )

        if problems:
            raise ConfigError(problems)


# ---------------------------------------------------------------------------
# Derived names: the single place columns, prefixes and tags come from
# ---------------------------------------------------------------------------

CLIMATOLOGY_FEATURE = "prob_clim"
VARIANT_SUFFIX = {"base": "", "fixed_cutoff": "_fixed_cutoff", "ref_onset": "_ref", "no_ref_filter": ""}


def window_tag(window: tuple[int, int]) -> str:
    return f"{window[0]}_{window[1]}"


def resolve_window(predictor: RainPredictor, onset: OnsetDefinition) -> int:
    """A rain predictor's window in days (tokens read from the onset definition)."""
    window = predictor.window
    if isinstance(window, int):
        return window
    token = window.lower()
    if token in ("trigger", "window"):
        return onset.window
    if token in ("dry_spell", "sum_window"):
        return onset.dry_spell.sum_window
    if token == "follow":
        return onset.follow_days
    if token == "min_dry":
        return onset.dry_spell.min_dry_days
    raise ValueError(f"unknown rain window token {window!r}")


def rain_feature_base(model: str, predictor: RainPredictor, onset: OnsetDefinition) -> str:
    if predictor.agg == "diff":
        return f"diff_{model}"
    return f"{predictor.agg}_{model}_{resolve_window(predictor, onset)}day"


def probability_feature_base(model: str, variant: str) -> str:
    return f"{model}_p_onset{VARIANT_SUFFIX[variant]}"


def feature_bases(config: BlendConfig) -> set[str]:
    """Every feature a formula may use: `<base>_qx` (one per week) or `<base>_week<n>`."""
    bases = {CLIMATOLOGY_FEATURE, f"{CLIMATOLOGY_FEATURE}_unc"}
    bases |= {f"{CLIMATOLOGY_FEATURE}_{window_tag(w)}" for w in config.climatology.windows}
    for model in config.models:
        bases |= {probability_feature_base(model.name, v) for v in ("base", "fixed_cutoff", "ref_onset")}
        bases |= {
            rain_feature_base(model.name, p, config.onset.definition) for p in model.rain_predictors
        }
    expanded = {f"{base}_qx" for base in bases}
    expanded |= {f"{base}_week{w}" for base in bases for w in range(1, config.features.bins.count + 1)}
    expanded |= {f"{CLIMATOLOGY_FEATURE}_unc_earlier", "onset_threshold", "lat", "lon"}
    return expanded


# Patsy functions a formula may call; their names are not features.
_FORMULA_FUNCTIONS = re.compile(r"^(?:I|C|Q|center|standardize|scale|np|numpy)(?:\..+)?$")


def formula_terms(text: str) -> list[str]:
    """The feature names a formula's right-hand side refers to."""
    rhs = text.split("~", 1)[1] if "~" in text else text
    tokens = [t.strip() for t in re.split(r"[+\-*:(),]", rhs)]
    return [
        t
        for t in tokens
        if t and not re.fullmatch(r"\d+(\.\d+)?", t) and not _FORMULA_FUNCTIONS.match(t)
    ]


def climatology_baseline_names(config: BlendConfig) -> list[str]:
    baselines = config.evaluation.climatology_baselines
    names = ["unc_clim_raw"] if baselines.unconditional else []
    if baselines.conditional:
        names.append("clim_raw")
    names += [f"clim_raw_{window_tag(w)}" for w in baselines.windows]
    return names


def outcome_labels(config: BlendConfig) -> list[str]:
    return [f"week{w}" for w in range(1, config.features.bins.count + 1)] + ["later"]


def cutoff_tag(config: BlendConfig) -> str:
    return {"fixed_cutoff": "_fixed_cutoff", "no_ref_filter": "_no_ref_filter"}.get(
        config.evaluation.cutoff_mode, ""
    )


def year_tag(config: BlendConfig) -> str:
    years = config.evaluation.years.holdout
    if not years:
        return ""
    return f"_{years[0]}" if len(years) == 1 else f"_{min(years)}_{max(years)}"


def output_tag(config: BlendConfig) -> str:
    """The `{cutoff}{years}` suffix every evaluation artifact carries."""
    return f"{cutoff_tag(config)}{year_tag(config)}"


def load_config(path: str | Path) -> BlendConfig:
    return BlendConfig.from_yaml(path)


# ---------------------------------------------------------------------------
# A small strict loader for frozen dataclasses
# ---------------------------------------------------------------------------


def _type_hints(cls) -> dict[str, Any]:
    return typing.get_type_hints(cls, globalns=globals())


def _build(cls, data: Any, path: str, problems: list[str]):
    if data is None:
        data = {}
    if not isinstance(data, dict):
        problems.append(f"{path or 'config'}: expected a mapping, got {type(data).__name__}")
        return None
    hints = _type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    for key in data:
        if key not in fields:
            close = difflib.get_close_matches(str(key), list(fields), n=1)
            hint = f" (did you mean {close[0]!r}?)" if close else ""
            problems.append(f"{_join(path, key)}: unknown key{hint}")
    values = {}
    for name, f in fields.items():
        where = _join(path, name)
        if name in data:
            parse = f.metadata.get(_PARSE)
            raw = data[name]
            values[name] = parse(raw, where, problems) if parse and raw is not None else _convert(
                hints[name], raw, where, problems
            )
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            problems.append(f"{where}: required")
            values[name] = None
    try:
        return cls(**values)
    except TypeError as exc:  # a required field missing; already reported
        problems.append(f"{path or 'config'}: {exc}")
        return None


def _convert(hint, value, path: str, problems: list[str]):
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        options = [a for a in args if a is not type(None)]
        # str | float style unions: take the value's own type when it is one.
        for option in options:
            if option in (int, float, str) and isinstance(value, option) and not isinstance(value, bool):
                return value
        if len(options) == 1:
            return _convert(options[0], value, path, problems)
        for option in options:
            trial: list[str] = []
            converted = _convert(option, value, path, trial)
            if not trial:
                return converted
        problems.append(f"{path}: {value!r} does not fit {hint}")
        return None
    if origin is Literal:
        if value not in args:
            problems.append(f"{path}: {value!r} is not one of {', '.join(map(repr, args))}")
            return None
        return value
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            value = [value]
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(args[0], item, f"{path}[{i}]", problems) for i, item in enumerate(value))
        if len(value) != len(args):
            problems.append(f"{path}: expected {len(args)} values, got {len(value)}")
            return None
        return tuple(_convert(a, item, f"{path}[{i}]", problems) for i, (a, item) in enumerate(zip(args, value)))
    if origin is dict:
        if not isinstance(value, dict):
            problems.append(f"{path}: expected a mapping")
            return {}
        key_type, value_type = args
        if value_type is Formula:
            return {
                str(k): (Formula(text=v) if isinstance(v, str) else _build(Formula, v, f"{path}.{k}", problems))
                for k, v in value.items()
            }
        return {str(k): _convert(value_type, v, f"{path}.{k}", problems) for k, v in value.items()}
    if dataclasses.is_dataclass(hint):
        return _build(hint, value, path, problems)
    if hint is bool:
        if not isinstance(value, bool):
            problems.append(f"{path}: expected true or false, got {value!r}")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            problems.append(f"{path}: expected a whole number, got {value!r}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"{path}: expected a number, got {value!r}")
            return value
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            problems.append(f"{path}: expected text, got {value!r}")
        return value
    return value


def _check_tree(node, path: str, problems: list[str]) -> None:
    """Run each section's own `_check`."""
    if dataclasses.is_dataclass(node):
        check = getattr(node, "_check", None)
        if check is not None:
            check(path or "config", problems)
        for f in dataclasses.fields(node):
            _check_tree(getattr(node, f.name), _join(path, f.name), problems)
    elif isinstance(node, tuple):
        for i, item in enumerate(node):
            _check_tree(item, f"{path}[{i}]", problems)
    elif isinstance(node, dict):
        for k, item in node.items():
            _check_tree(item, f"{path}.{k}", problems)


def _check_rain_transform(value, path: str, problems: list[str]) -> None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return
    text = str(value).strip().lower()
    if text in ("identity", "none", "", "sqrt", "fourth_root", "log1p"):
        return
    if re.fullmatch(r"power:\s*-?\d+(\.\d+)?", text):
        return
    problems.append(f"{path}: {value!r} is not identity, sqrt, fourth_root, log1p or power:<p>")


def _years_text(years: tuple[int, ...]) -> str | int:
    """Years as compact text: 2000:2020 for a run, comma-separated runs otherwise."""
    if len(years) == 1:
        return years[0]
    runs, start = [], None
    for prev, year in zip((None, *years), years):
        if start is None:
            start = year
        elif year != prev + 1:
            runs.append((start, prev))
            start = year
    runs.append((start, years[-1]))
    return ",".join(f"{a}:{b}" if a != b else f"{a}" for a, b in runs)


def _window_text(window: tuple[int, int]) -> str | int:
    return window[0] if window[0] == window[1] else f"{window[0]}:{window[1]}"


def _dump(node):
    if dataclasses.is_dataclass(node):
        out = {}
        for f in dataclasses.fields(node):
            value = getattr(node, f.name)
            default = (
                f.default
                if f.default is not dataclasses.MISSING
                else (f.default_factory() if f.default_factory is not dataclasses.MISSING else dataclasses.MISSING)
            )
            if value == default:
                continue
            parse = f.metadata.get(_PARSE)
            if parse is _years:
                out[f.name] = _years_text(value)
            elif parse is _year_window:
                out[f.name] = _window_text(value)
            elif parse is _year_windows:
                out[f.name] = [_window_text(w) for w in value]
            else:
                out[f.name] = _dump(value)
        return out
    if isinstance(node, tuple):
        return [_dump(item) for item in node]
    if isinstance(node, dict):
        return {k: _dump(v) for k, v in node.items()}
    return node


def _join(path: str, key) -> str:
    return f"{path}.{key}" if path else str(key)
