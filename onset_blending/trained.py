"""TrainedBlend: everything operations needs from training, in one file.

A trained blend holds its recipe (the BlendConfig, with every derived default
frozen), the fitted model, the reference data training resolved
(thresholds, reference onsets, regrid and cell weights, as arrays), the
observed onsets the climatology is fitted on, and each model's forecast
signature. A forecast takes only live forecast files and an
issue date; everything else comes from here, so operations cannot run with
other parameters than training did.

The file is a zip of JSON, YAML, NumPy and the reference files: no pickles,
so it loads across library versions and is safe to read. It is written to a
temporary file and renamed into place, and every member is checked against
its digest on load.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from .core import io as core_io
from .core.onsets import ObservedOnsets
from .core.reference import Reference
from .config import BlendConfig, ConfigError, ForecastModel

# Bump whenever the numbers the features are computed from change: a blend
# trained under another version is refused rather than fed different features.
FEATURES_VERSION = 1
FORMAT = "onset-blending/trained-blend"
FORMAT_VERSION = 1
_MANIFEST = "manifest.json"


# ---------------------------------------------------------------------------
# Forecast signatures
# ---------------------------------------------------------------------------


def forecast_signature(path: str | Path, model: ForecastModel, *, selected: bool = True) -> dict:
    """What a forecast file holds, read in its own format (formats), apart
    from the issue dates: the variable, the units its rainfall is read in
    and how its daily values were aggregated, other dimension sizes, member
    ids, lead days, the grid (as sets: order is the file's own) and the
    issue time of day. selected: after the model's lead-day and member
    selection, as training reads it; else all a (live) file holds."""
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        if model.variable not in ds.variables:
            raise ValueError(f"{Path(path).name} has no {model.variable!r} variable")
        fmt = core_io.forecast_format(ds, model.variable, model.dims, model.unit_scale, where=Path(path).name)
        variable = ds.variables[model.variable]
        names = {d: fmt.rename.get(d, d) for d in variable.dimensions}

        def values(dim):
            if dim in ds.variables:
                raw = ds.variables[dim][:]
                return np.asarray(raw.filled(np.nan) if np.ma.isMaskedArray(raw) else raw)
            return np.arange(len(ds.dimensions[dim]))

        signature: dict = {"variable": model.variable, "units": fmt.units, "dims": {}}
        lead_dim = next((d for d, name in names.items() if name == "day"), None)
        signature["aggregation"] = _aggregation(ds, variable, lead_dim)
        for dim, name in names.items():
            if name == "time":
                times = ds.variables[dim]
                stamps = netCDF4.num2date(
                    times[:], times.units, getattr(times, "calendar", "standard"),
                    only_use_cftime_datetimes=False,
                )
                signature["issue_hours"] = sorted({int(t.hour) for t in np.atleast_1d(stamps)})
                continue
            vals = values(dim)
            if name == "day":
                days = [int(v) for v in vals]
                lead = model.lead_days
                signature["lead_days"] = [d for d in days if lead.min <= d <= lead.max] if selected else days
            elif name == "number":
                members = [int(v) for v in vals]
                if model.max_member is not None and selected:
                    members = [m for m in members if m <= model.max_member]
                signature["members"] = members
            elif name in ("lat", "lon"):
                signature[name] = sorted(round(float(v), 6) for v in vals)
            elif name == "adm3_name":
                signature["unit_ids"] = sorted(str(v) for v in vals)
            else:
                signature["dims"][name] = len(vals)
    return signature


def issue_dates(path: str | Path, model: ForecastModel) -> list[dt.date]:
    """The issue dates a forecast file holds."""
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        if model.variable not in ds.variables:
            return []
        rename = core_io.forecast_format(ds, model.variable, model.dims, model.unit_scale, where=Path(path).name).rename
        for dim in ds.variables[model.variable].dimensions:
            if rename.get(dim) == "time" and dim in ds.variables:
                times = ds.variables[dim]
                stamps = netCDF4.num2date(
                    times[:], times.units, getattr(times, "calendar", "standard"),
                    only_use_cftime_datetimes=False,
                )
                return sorted({dt.date(t.year, t.month, t.day) for t in np.atleast_1d(stamps)})
    return []


def _aggregation(ds, variable, lead_dim: str | None) -> dict:
    """How the file says its daily values were made: the cell method over
    its lead dimension (e.g. sum) and the window its lead coordinate states
    (e.g. the UTC day ending at the issue time plus the lead), or None
    where it does not say."""
    method = None
    if lead_dim is not None:
        for name, how in re.findall(r"([^\s:]+):\s*([^\s:]+)", str(getattr(variable, "cell_methods", "") or "")):
            if name in (lead_dim, "day") or lead_dim.startswith(name):
                method = how.lower()
    window = None
    if lead_dim is not None and lead_dim in ds.variables:
        coordinate = ds.variables[lead_dim]
        text = getattr(coordinate, "description", None) or getattr(coordinate, "long_name", None)
        window = " ".join(str(text).lower().split()) if text else None
    return {"method": method, "window": window}


def _signature_differences(trained: dict, live: dict) -> list[str]:
    """How a live file fails to cover the trained signature. It may hold
    more (longer leads, more members, a larger domain such as the globe):
    the blend reads only what it was trained on. It may not hold less, nor
    another resolution, units, aggregation, issue hour or variable."""
    out = []
    for key, label in (
        ("variable", "variable"), ("units", "rainfall units"), ("aggregation", "daily aggregation"),
        ("issue_hours", "issue hour (UTC)"), ("dims", "other dimensions"),
    ):
        if trained.get(key) != live.get(key):
            out.append(f"{label} {_brief(live.get(key))}, trained on {_brief(trained.get(key))}")
    for key, label in (("lead_days", "lead days"), ("members", "members"), ("unit_ids", "unit ids")):
        have, need = live.get(key), trained.get(key)
        if (have is None) != (need is None):
            out.append(f"{label} {_brief(have)}, trained on {_brief(need)}")
        elif need is not None and set(need) - set(have):
            missing = sorted(set(need) - set(have))
            out.append(f"{label} {_brief(have)}, trained on {_brief(need)} (missing {_brief(missing)})")
    for key, label in (("lat", "latitudes"), ("lon", "longitudes")):
        have, need = live.get(key), trained.get(key)
        if (have is None) != (need is None):
            out.append(f"{label} {_brief(have)}, trained on {_brief(need)}")
        elif need is not None:
            have_step, need_step = _grid_step(have), _grid_step(need)
            if None not in (have_step, need_step) and have_step != need_step:
                out.append(f"{label} every {_degrees(have_step)}, trained on every {_degrees(need_step)} (another resolution)")
            elif set(need) - set(have):
                out.append(
                    f"{label} {_span(have)}, trained on {_span(need)}: the trained domain is not covered"
                )
    return out


def _grid_step(values: list[float]) -> float | None:
    """A grid's spacing (the median gap between its distinct values)."""
    distinct = np.unique(values)
    return round(float(np.median(np.diff(distinct))), 6) if len(distinct) > 1 else None


def _degrees(step: float | None) -> str:
    return "?" if step is None else f"{step:g}°"


def _span(values: list[float]) -> str:
    return f"{min(values):g}–{max(values):g}" if values else "none"


def _brief(value) -> str:
    if isinstance(value, list) and len(value) > 6:
        if all(isinstance(v, int) for v in value) and value == list(range(value[0], value[-1] + 1)):
            return f"{value[0]}–{value[-1]}"
        return f"{len(value)} values {value[0]}…{value[-1]}"
    return repr(value)


# ---------------------------------------------------------------------------
# Arrays without pickles
# ---------------------------------------------------------------------------


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _npz_load(data: bytes):
    return np.load(io.BytesIO(data), allow_pickle=False)


# ---------------------------------------------------------------------------
# The trained blend
# ---------------------------------------------------------------------------


@dataclass
class TrainedBlend:
    config: BlendConfig                 # the recipe, derived defaults frozen
    model: object                       # model.FittedModel
    observed: ObservedOnsets            # what the climatology is fitted on
    units: tuple[str, ...]              # the units the blend was trained and scores on
    signatures: dict[str, dict]         # model name -> forecast signature
    reference: Reference                # thresholds, reference onsets, regrid and cell weights
    provenance: dict = field(default_factory=dict)

    # -- building -----------------------------------------------------------

    @classmethod
    def from_training(cls, config: BlendConfig, inputs, model, *, provenance: dict | None = None) -> "TrainedBlend":
        """The blend a training run produced (inputs: training.Inputs)."""

        climatology = inputs.climatology
        resolved = dataclasses.replace(
            config,
            climatology=dataclasses.replace(
                config.climatology, train_years=climatology.train_years, test_years=climatology.test_years
            ),
        )
        signatures, problems = {}, []
        for model_config in config.models:
            files = core_io.list_files(model_config.files, config, f"model {model_config.name}")
            seen = {}
            for path in files:
                seen.setdefault(json.dumps(forecast_signature(path, model_config), sort_keys=True), []).append(path.name)
            if len(seen) > 1:
                groups = "; ".join(f"{', '.join(names[:4])}{'…' if len(names) > 4 else ''}" for names in seen.values())
                problems.append(
                    f"{model_config.name}: its training files differ in dimensions, members, lead days or grid "
                    f"({groups}); a blend needs one forecast structure to forecast with"
                )
            signatures[model_config.name] = json.loads(next(iter(seen)))
        if problems:
            raise ConfigError(problems)

        return cls(
            config=resolved,
            model=model,
            observed=inputs.observed,
            units=tuple(sorted({str(unit) for unit in inputs.weekly.units})),
            signatures=signatures,
            reference=inputs.reference,
            provenance={
                "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "training_years": list(config.evaluation.years.training),
                **(provenance or {}),
            },
        )

    # -- files --------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write the blend to path, atomically."""
        from importlib.metadata import PackageNotFoundError, version

        path = Path(path)
        members: dict[str, bytes] = {"config.yml": self.config.to_yaml().encode()}
        model_meta, model_arrays = self.model.to_arrays()
        members["model.json"] = json.dumps(model_meta, indent=1).encode()
        members["model.npz"] = _npz_bytes(model_arrays)
        observed = self.observed.data
        members["observed.npz"] = _npz_bytes({
            "unit": observed["unit"].values.astype(str),
            "year": observed["year"].values,
            **{name: observed[name].values for name in ("onset_day", "onset_idx", "present")},
        })
        members["units.json"] = json.dumps(list(self.units)).encode()
        members["signatures.json"] = json.dumps(self.signatures, indent=1).encode()
        reference_meta, reference_arrays = self.reference.to_arrays()
        members["reference.json"] = json.dumps(reference_meta).encode()
        members["reference.npz"] = _npz_bytes(reference_arrays)
        try:
            package_version = version("onset-blending")
        except PackageNotFoundError:
            package_version = None
        manifest = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "features_version": FEATURES_VERSION,
            "package_version": package_version,
            "provenance": self.provenance,
            "config_sha256": hashlib.sha256(members["config.yml"]).hexdigest(),
            "members": {name: hashlib.sha256(data).hexdigest() for name, data in members.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(handle, "wb") as raw, zipfile.ZipFile(raw, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(_MANIFEST, json.dumps(manifest, indent=1))
                for name, data in members.items():
                    archive.writestr(name, data)
                raw.flush()
                os.fsync(raw.fileno())
            os.replace(temp, path)
        except BaseException:
            Path(temp).unlink(missing_ok=True)
            raise
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TrainedBlend":
        """Read a saved blend; refuses a damaged file or other feature numerics."""
        from .blend.logit import FittedModel

        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read(_MANIFEST))
            if manifest.get("format") != FORMAT or manifest.get("format_version") != FORMAT_VERSION:
                raise ValueError(f"{path} is not a trained blend this version can read")
            members = {name: archive.read(name) for name in manifest["members"]}
        damaged = [n for n, d in members.items() if hashlib.sha256(d).hexdigest() != manifest["members"][n]]
        if damaged:
            raise ValueError(f"{path} is damaged: {', '.join(damaged)} do not match their digests")
        if manifest["features_version"] != FEATURES_VERSION:
            raise ValueError(
                f"{path} was trained with feature version {manifest['features_version']}; this package "
                f"computes version {FEATURES_VERSION}. Retrain it, or forecast with the version it was trained with."
            )
        config = BlendConfig.from_dict(yaml.safe_load(members["config.yml"]))
        arrays = _npz_load(members["observed.npz"])
        observed = ObservedOnsets(
            data=xr.Dataset(
                {name: (("unit", "year"), arrays[name]) for name in ("onset_day", "onset_idx", "present")},
                coords={"unit": arrays["unit"].astype(object), "year": arrays["year"]},
            ),
            season_start=config.onset.season_start,
        )
        model = FittedModel.from_arrays(json.loads(members["model.json"]), dict(_npz_load(members["model.npz"])))
        return cls(
            config=config,
            model=model,
            observed=observed,
            units=tuple(json.loads(members["units.json"])),
            signatures=json.loads(members["signatures.json"]),
            reference=Reference.from_arrays(json.loads(members["reference.json"]), _npz_load(members["reference.npz"])),
            provenance=manifest.get("provenance", {}),
        )

    # -- operations -----------------------------------------------------------

    def check_forecast_files(self, forecasts: dict[str, list[Path]]) -> list[str]:
        """Every way the live files differ from what the blend was trained on."""
        problems = []
        trained = set(self.config.model_names)
        given = set(forecasts)
        if given != trained:
            missing, extra = sorted(trained - given), sorted(given - trained)
            problems.append(
                "forecasts must be given for exactly the trained models ("
                + ", ".join(sorted(trained)) + ")"
                + (f"; missing {', '.join(missing)}" if missing else "")
                + (f"; not trained on {', '.join(extra)}" if extra else "")
            )
        for name in sorted(trained & given):
            model = self.config.model(name)
            for path in forecasts[name]:
                try:
                    live = forecast_signature(path, model, selected=False)
                except (OSError, ValueError) as exc:
                    problems.append(f"{name}: {Path(path).name}: {exc}")
                    continue
                differences = _signature_differences(self.signatures[name], live)
                if differences:
                    problems.append(f"{name}: {Path(path).name}: " + "; ".join(differences))
        return problems

    def forecast(self, forecasts: dict[str, str | Path | list], *, issue_date=None, resources=None):
        """Blended onset probabilities from live forecast files (see forecast.run)."""
        from . import forecast as forecast_module

        return forecast_module.run(self, forecasts, issue_date=issue_date, resources=resources)
