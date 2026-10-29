"""Forecast files in their own formats.

The files of one model need not be laid out alike: the archives a blend
trains on and the live files it forecasts from may name their dimensions
differently and store rainfall in other units. Each file's format is
resolved when it is read, by the rules ai-almanac registers datasets by:

- a dimension's role (time, day, number, lat, lon, adm3_name) comes from
  the model's configured `dims`, else its name (case-insensitive aliases),
  else its coordinate (a lead dimension counted in days);
- rainfall is converted to millimetres from the variable's declared
  `units`; the model's `unit_scale` applies only to files declaring none.

Everything downstream, the forecast signature included, sees the resolved
dimension names and millimetres.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

# Millimetres per unit of a rainfall total's declared (CF) units.
MILLIMETRES_PER_UNIT = {
    "m": 1000.0,
    "metres": 1000.0,
    "meters": 1000.0,
    "m/day": 1000.0,
    "mm": 1.0,
    "mm/day": 1.0,
    "mm day-1": 1.0,
    "mm d-1": 1.0,
    "kg m-2": 1.0,
    "kg m**-2": 1.0,
    "kg/m2": 1.0,
}

# Dimension names by role, lower case.
_ALIASES = {
    "time": "time", "init_time": "time", "initial_time": "time", "forecast_reference_time": "time",
    "day": "day", "lead_day": "day", "lead_days": "day", "prediction_timedelta_daily": "day",
    "number": "number", "member": "number", "members": "number", "ensemble": "number",
    "ensemble_member": "number", "realization": "number", "sample": "number",
    "lat": "lat", "latitude": "lat",
    "lon": "lon", "longitude": "lon",
    "adm3": "adm3_name", "adm3_name": "adm3_name",
}
_DAY_UNITS = ("day", "days")


@dataclass(frozen=True)
class ForecastFormat:
    """How one file is read: its dimensions' roles and its rainfall scale."""

    rename: dict[str, str]   # file dimension -> role, for every dimension with one
    scale: float             # multiplies the file's rainfall into millimetres
    declared_units: str | None

    @property
    def units(self) -> str:
        """The rainfall units the blend reads: millimetres, or how they were taken."""
        return "mm" if self.declared_units is not None else f"undeclared (x{self.scale:g})"


def forecast_format(ds, variable: str, dims: Mapping[str, str], unit_scale: float, where="") -> ForecastFormat:
    """The format of an open netCDF4 Dataset's forecast variable."""
    if variable not in ds.variables:
        raise ValueError(f"{where}: no {variable!r} variable")
    var = ds.variables[variable]
    configured = {str(k).lower(): v for k, v in dims.items()}
    rename: dict[str, str] = {}
    for dim in var.dimensions:
        role = configured.get(dim.lower()) or _ALIASES.get(dim.lower())
        if role is None and dim in ds.variables:
            units = str(getattr(ds.variables[dim], "units", "")).strip().lower()
            if units in _DAY_UNITS:
                role = "day"
        if role is not None:
            rename[dim] = role
    roles = list(rename.values())
    repeated = sorted({role for role in roles if roles.count(role) > 1})
    if repeated:
        raise ValueError(f"{where}: several dimensions of {variable!r} read as {', '.join(repeated)}")
    declared = getattr(var, "units", None)
    declared = str(declared).strip() if declared is not None and str(declared).strip() else None
    return ForecastFormat(rename=rename, scale=rainfall_scale(declared, unit_scale, where), declared_units=declared)


def rainfall_scale(units: str | None, fallback: float, where="") -> float:
    """Millimetres per unit of rainfall declared in `units`; fallback when undeclared."""
    if units is None:
        return float(fallback)
    try:
        return MILLIMETRES_PER_UNIT[units.lower()]
    except KeyError:
        supported = ", ".join(repr(u) for u in MILLIMETRES_PER_UNIT)
        raise ValueError(f"{where}: unrecognised rainfall units {units!r} (supported: {supported})") from None


def of_file(path, variable: str, dims: Mapping[str, str], unit_scale: float) -> ForecastFormat:
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        return forecast_format(ds, variable, dims, unit_scale, where=str(path))
