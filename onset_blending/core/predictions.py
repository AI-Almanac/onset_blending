"""A model's probabilities of the outcome bins for samples, with the samples' keys.

`Predictions.data` is an xarray Dataset over `sample` (and `bin`):
`probability` (sample, bin) over the interval bins (week1..weekN, later),
optionally `earlier` (sample: an onset before the issue, which the RPS
counts) and `daily` (sample, lead_day: a model's daily probabilities); the
keys are coordinates over sample: id, year, time, lat, lon (the last
three when the samples have them), and `outcome`
(positions into the bins' labels; -1 without one, len(labels) for another
label). CV predictions, baselines and external predictions are all held so,
scored from the arrays (core.metrics), and saved as NetCDF.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

__all__ = ["Predictions", "outcome_codes", "outcome_series"]


def outcome_codes(values, labels) -> np.ndarray:
    """Outcome labels as positions into `labels`: -1 where missing, len(labels)
    for a label that is not a bin."""
    index = {label: k for k, label in enumerate(labels)}
    other = len(labels)
    out = np.empty(len(values), dtype=np.int8)
    for i, value in enumerate(values):
        if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
            out[i] = -1
        else:
            out[i] = index.get(value, other)
    return out


def outcome_series(codes: np.ndarray, labels: list[str]) -> pd.Series:
    """The legacy table's outcome column: the labels, missing where code is -1
    (and for a code past the labels)."""
    values = np.array([*labels, None, None], dtype=object)[np.where(codes < 0, len(labels) + 1, codes)]
    return pd.Series(values, dtype=object).astype(str).where(pd.Series(values).notna(), None)


@dataclass
class Predictions:
    data: xr.Dataset

    # -- building -------------------------------------------------------------

    @classmethod
    def of_samples(cls, weekly, rows: np.ndarray, probabilities: np.ndarray, labels, *,
                   earlier: np.ndarray | None = None, daily: np.ndarray | None = None) -> Predictions:
        """Predictions for weekly samples (features.Weekly; rows are positions)."""
        rows = np.asarray(rows, dtype=np.int64)
        coords = {
            "id": ("sample", weekly.take("id", rows).astype(str)),
            "year": ("sample", weekly.take("year", rows)),
            "outcome": ("sample", weekly.outcome_codes(rows, labels)),
            "bin": list(labels),
        }
        if "time" in weekly:
            coords["time"] = ("sample", _times(weekly, rows))
        for key in ("lat", "lon"):
            if key in weekly:
                coords[key] = ("sample", np.asarray(weekly.take(key, rows), dtype=float))
        return cls._build(coords, probabilities, earlier, daily)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, labels) -> Predictions:
        """A legacy CV-prediction table: keys and cv_<bin> (and cv_earlier) columns."""
        labels = list(labels)
        coords = {
            "id": ("sample", frame["id"].astype(str).to_numpy()),
            "year": ("sample", frame["year"].to_numpy()),
            "outcome": ("sample", outcome_codes(frame["outcome"].to_numpy(dtype=object), labels)),
            "bin": labels,
        }
        if "time" in frame:
            coords["time"] = ("sample", pd.to_datetime(frame["time"]).to_numpy())
        for key in ("lat", "lon"):
            if key in frame:
                coords[key] = ("sample", frame[key].to_numpy(dtype=float))
        probabilities = frame[[f"cv_{b}" for b in labels]].to_numpy(dtype=float)
        earlier = frame["cv_earlier"].to_numpy(dtype=float) if "cv_earlier" in frame else None
        return cls._build(coords, probabilities, earlier, None)

    @classmethod
    def _build(cls, coords, probabilities, earlier, daily) -> Predictions:
        variables = {"probability": (("sample", "bin"), np.asarray(probabilities, dtype=float))}
        if earlier is not None:
            variables["earlier"] = ("sample", np.asarray(earlier, dtype=float))
        if daily is not None:
            variables["daily"] = (("sample", "lead_day"), np.asarray(daily, dtype=float))
            coords = {**coords, "lead_day": np.arange(1, daily.shape[1] + 1)}
        labels = list(coords["bin"])
        data = xr.Dataset(variables, coords=coords)
        data["outcome"].attrs["labels"] = labels
        return cls(data)

    @classmethod
    def concat(cls, parts: list[Predictions]) -> Predictions:
        data = xr.concat([p.data for p in parts], dim="sample", data_vars="minimal", coords="different",
                         compat="equals", join="exact")
        data["outcome"].attrs["labels"] = parts[0].labels
        return cls(data)

    def take(self, positions: np.ndarray) -> Predictions:
        """These samples (positions), in this order."""
        return Predictions(self.data.isel(sample=np.asarray(positions, dtype=np.int64)))

    def with_probabilities(self, probabilities: np.ndarray, *, earlier: np.ndarray | None = None) -> Predictions:
        """Other probabilities (over the same bins) for these samples."""
        data = self.data.drop_vars(list(self.data.data_vars))
        data["probability"] = (("sample", "bin"), np.asarray(probabilities, dtype=float))
        if earlier is not None:
            data["earlier"] = ("sample", np.asarray(earlier, dtype=float))
        if "lead_day" in data.dims:
            data = data.drop_dims("lead_day")
        data["outcome"].attrs["labels"] = self.labels
        return Predictions(data)

    # -- reading --------------------------------------------------------------

    def __len__(self) -> int:
        return self.data.sizes["sample"]

    @property
    def labels(self) -> list[str]:
        return list(self.data["bin"].values)

    @property
    def probabilities(self) -> np.ndarray:
        return self.data["probability"].values

    @property
    def earlier(self) -> np.ndarray | None:
        return self.data["earlier"].values if "earlier" in self.data else None

    @property
    def outcome(self) -> np.ndarray:
        return self.data["outcome"].values

    @property
    def ids(self) -> np.ndarray:
        return self.data["id"].values

    @property
    def year(self) -> np.ndarray:
        return self.data["year"].values

    def frame(self) -> pd.DataFrame:
        """The legacy CV-prediction table: id, time, year, lat, lon, outcome
        and cv_<bin> (cv_earlier, p_day_<k> when present)."""
        labels = self.labels
        out = {"id": pd.Series(self.ids, dtype=str)}
        if "time" in self.data.coords:
            out["time"] = pd.Series(pd.DatetimeIndex(self.data["time"].values).date, dtype=object)
        out["year"] = self.year
        for key in ("lat", "lon"):
            if key in self.data.coords:
                out[key] = self.data[key].values
        out["outcome"] = outcome_series(self.outcome, labels)
        table = pd.DataFrame(out)
        for j, label in enumerate(labels):
            table[f"cv_{label}"] = self.probabilities[:, j]
        if self.earlier is not None:
            table["cv_earlier"] = self.earlier
        if "daily" in self.data:
            daily = self.data["daily"].values
            for d in range(daily.shape[1] - 1):
                table[f"p_day_{d + 1}"] = daily[:, d]
            table["p_later"] = daily[:, -1]
        return table

    def to_netcdf(self, path) -> None:
        data = self.data.copy()
        data["outcome"].attrs["labels"] = ",".join(self.labels)
        data.to_netcdf(path)

    @classmethod
    def open(cls, path) -> Predictions:
        with xr.open_dataset(path) as stored:
            data = stored.load()
        labels = str(data["outcome"].attrs["labels"]).split(",")
        data["outcome"].attrs["labels"] = labels
        return cls(data)


def _times(weekly, rows) -> np.ndarray:
    values = weekly.take("time", rows)
    return pd.to_datetime(pd.Series(values)).to_numpy() if values.dtype == object else values
