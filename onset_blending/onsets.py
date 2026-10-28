"""Onsets: observed onset dates per unit and season, and forecast onset
probabilities per unit and issue date."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import _parallel, _specs, domain, inputs
from ._legacy import nc_utils, onset_utils
from .config import BlendConfig, ForecastModel

# Peak bytes per forecast value while one file is processed, and the cost
# beyond its values (per-row columns, reader tiles); measured on India 0.25°.
_FORECAST_BYTES_PER_VALUE = 18
_FORECAST_FIXED_BYTES = 512 * 2**20


def _context(config: BlendConfig, spec: dict) -> dict:
    return {
        "spec": spec,
        "convention": domain.id_convention(config),
        "area": domain.AreaFilter(config),
        "thresholds": onset_utils.read_thresholds(spec),
        "reference_onset": onset_utils.read_ref_onset_dates(spec),
        "cell_transform": nc_utils.read_cell_transform(spec),
    }


def _observed_file(path: Path, context: dict) -> pd.DataFrame:
    """One observation file's onsets: (id, year) rows with onset_day and onset_date."""
    spec = context["spec"]
    variable = spec["input"]["value_col"]
    df = nc_utils.nc_read_groundtruth_long(
        nc_path=str(path),
        var_name=variable,
        dim_rename_map=spec["dimensions"]["rename"],
        missing_rain_policy=spec["options"]["missing_rain_policy"],
    )
    if df is None:
        import netCDF4

        with netCDF4.Dataset(path) as ds:
            found = ", ".join(ds.variables)
        raise ValueError(f"{path.name} has no {variable!r} variable (it has: {found})")
    df = domain.with_ids(df, context["convention"])
    if context["cell_transform"] is not None:
        df = nc_utils.transform_groundtruth_long(df, context["cell_transform"], variable.lower(), spec=spec)
    df = context["area"](df)
    return nc_utils.process_ground_truth_rainfall_id(
        df,
        spec,
        ref_onset_dt=context["reference_onset"],
        thr_dt=context["thresholds"],
        value_col=variable.lower(),
    )["wide"]


def observed(config: BlendConfig, *, workers: int = 1, budget_bytes: int | None = None) -> pd.DataFrame:
    """Observed onsets for every observation file: one row per unit and season
    (id, year, onset_idx, onset_date, onset_day, cutoff_date)."""
    paths = inputs.list_files(config.observations.files, config, "observations")
    weights = _regrid_weights(config, None)
    context = _context(config, _specs.observation_spec(config))
    with domain.on_units(paths, weights, workers=workers, budget_bytes=budget_bytes) as unit_paths:
        parts = _parallel.map_ordered(_observed_file, unit_paths, context, workers)
    return pd.concat(parts, ignore_index=True)


def blendable_ids(observed_wide: pd.DataFrame, min_onset_years: int) -> frozenset[str]:
    """Units with at least min_onset_years observed onsets over every season.

    The climatology keeps only units with that many onsets in its training
    years, and the inner join drops every forecast unit the climatology
    lacks, so no forecast unit outside this set reaches the features.
    """
    has_onset = pd.to_numeric(observed_wide["onset_day"], errors="coerce").notna()
    counts = observed_wide.loc[has_onset, "id"].astype(str).value_counts()
    return frozenset(counts[counts >= int(min_onset_years)].index)


@dataclass
class ForecastOnsets:
    """A model's forecast onsets, one table per file (a season each)."""

    model: str
    parts: list[pd.DataFrame]
    has_spread: bool = False
    member_counts: list[int] = field(default_factory=list)

    years: list[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.years:
            self.years = sorted({int(y) for part in self.parts for y in part["year"].dropna().unique()})

    def release(self) -> None:
        """Drop the tables, keeping the summary."""
        self.parts = []


def _has_spread(df: pd.DataFrame) -> bool:
    """Whether the members' rainfall spread (std, ddof=1) is non-null anywhere:
    some unit and issue date has two or more non-missing members."""
    if "number" not in df.columns:
        return False
    keys = [df["id"], df["time"]]
    for col in (c for c in df.columns if c.startswith("rain_day_")):
        if (df[col].notna().groupby(keys).sum() >= 2).any():
            return True
    return False


def _kept_points(path: Path, context: dict):
    """The forecast rows for only the units that reach the features, and
    whether the members spread anywhere in the area of interest; None when
    the file's layout needs a full read."""
    spec, variable, rename = context["spec"], context["variable"], context["spec"]["dimensions"]["rename"]
    points = nc_utils.forecast_points(str(path), variable, rename, day_dim="day")
    if points is None:
        return None
    points = domain.with_ids(points, context["convention"])
    marked = points.assign(_point=np.arange(len(points)))
    inside = context["area"].area(marked)
    in_area = np.zeros(len(points), dtype=bool)
    in_area[inside["_point"].to_numpy()] = True
    if points["id"][in_area].duplicated().any():
        return None
    keep = in_area.copy()
    if context["keep_ids"] is not None:
        keep &= points["id"].isin(context["keep_ids"]).to_numpy()
    df, has_spread = nc_utils.read_forecast_points(
        str(path), variable, rename, spec, keep,
        spread_over=in_area, scale=context["unit_scale"], day_dim="day", prefix="rain",
    )
    return domain.with_ids(df, context["convention"]), has_spread


def _in_millimetres(df: pd.DataFrame, scale: float) -> pd.DataFrame:
    """The lead-day rainfall columns times scale, kept side by side as one block."""
    if scale == 1.0:
        return df
    rain = [c for c in df.columns if c.startswith("rain_day_")]
    columns = list(df.columns)
    first = columns.index(rain[0]) if rain else 0
    if not rain or columns[first : first + len(rain)] != rain:
        return df.assign(**{c: df[c] * scale for c in rain})
    scaled = pd.DataFrame(df[rain].to_numpy() * scale, columns=rain, index=df.index, copy=False)
    return pd.concat([df.iloc[:, :first], scaled, df.iloc[:, first + len(rain) :]], axis=1)


def _forecast_file(path: Path, context: dict) -> dict:
    """One forecast file's onset probabilities, member counts and spread."""
    spec = context["spec"]
    kept = None if context["cell_transform"] is not None or context["area"].dissemination else _kept_points(path, context)
    if kept is not None:
        df, has_spread = kept
    else:
        df = nc_utils.nc_read_forecast_wide(
            nc_path=str(path),
            var_name=context["variable"],
            dim_rename_map=spec["dimensions"]["rename"],
            spec=spec,
            day_dim="day",
            prefix="rain",
        )
        if df is None:
            raise ValueError(f"{path.name} has no {context['variable']!r} variable")
        df = _in_millimetres(df, context["unit_scale"])
        df = domain.with_ids(df, context["convention"])
        if context["cell_transform"] is not None:
            df = nc_utils.transform_forecast_wide(df, context["cell_transform"], spec=spec)
        df = context["area"](df)
        # Before keeping only the blended units: whether a model has a rain
        # spread depends on the whole area of interest.
        has_spread = _has_spread(df)
        if context["keep_ids"] is not None:
            df = df[df["id"].isin(context["keep_ids"])]
    member_counts = df.groupby(["id", "time"]).size().tolist() if "number" in df.columns else []
    processed = nc_utils.process_rainfall_forecast_id(
        df, spec, ref_onset_dt=context["reference_onset"], thr_dt=context["thresholds"]
    )
    return {"wide": processed["wide"], "member_counts": member_counts, "has_spread": has_spread}


def _forecast_bytes(path: Path, context: dict) -> int:
    """Peak memory to process one forecast file."""
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        variable = ds.variables.get(context["variable"])
        values = math.prod(variable.shape) if variable is not None else 0
    kept = 1.0
    if context["keep_ids"] is not None:
        points = nc_utils.forecast_points(
            str(path), context["variable"], context["spec"]["dimensions"]["rename"], day_dim="day"
        )
        if points is not None and len(points):
            kept = float(domain.with_ids(points, context["convention"])["id"].isin(context["keep_ids"]).mean())
    return int(values * kept * _FORECAST_BYTES_PER_VALUE) + _FORECAST_FIXED_BYTES


def forecast(
    config: BlendConfig,
    model: ForecastModel,
    *,
    keep_ids: frozenset[str] | None = None,
    workers: int = 1,
    budget_bytes: int | None = None,
) -> ForecastOnsets:
    """A model's forecast onset probabilities for every one of its files.

    keep_ids, when given, limits the rows to those units (blendable_ids under
    the inner join); the results for them are unchanged.
    """
    paths = inputs.list_files(model.files, config, f"model {model.name}")
    context = {
        **_context(config, _specs.forecast_spec(config, model)),
        "variable": model.variable,
        "unit_scale": float(model.unit_scale),
        "keep_ids": keep_ids,
    }
    weights = _regrid_weights(config, model)
    with domain.on_units(paths, weights, workers=workers, budget_bytes=budget_bytes) as unit_paths:
        count = _parallel.workers_within([_forecast_bytes(p, context) for p in unit_paths], workers, budget_bytes)
        results = _parallel.map_ordered(_forecast_file, unit_paths, context, count)
    return ForecastOnsets(
        model=model.name,
        parts=[r["wide"] for r in results],
        has_spread=any(r["has_spread"] for r in results),
        member_counts=[c for r in results for c in r["member_counts"]],
    )


def _regrid_weights(config: BlendConfig, model: ForecastModel | None) -> Path | None:
    """The weights that put a model's (or the observations') grid on the units."""
    regrid = config.domain.regrid
    if regrid is None:
        return None
    supplied = (model.regrid_weights if model is not None else None) or regrid.weights
    if supplied is None:
        raise NotImplementedError(
            "Building regrid weights from domain.regrid.shapefile arrives later in phase 2; "
            "give domain.regrid.weights for now"
        )
    return config.path(supplied)
