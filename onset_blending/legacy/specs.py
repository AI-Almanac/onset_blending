"""Convert a set of today's stage specs into one BlendConfig.

The old pipeline spreads one blend over six specs: regrid, the ground-truth
raw_data spec (which also holds the climatology runs), one raw_data spec per
forecast model, combine, connect and cv_models. Many facts are restated across
them. `convert_specs` reads a set, checks that every restated fact agrees, and
returns one BlendConfig:

- Conflicting facts are reported together as one ConfigError.
- Keys that only wire one stage's output to the next (intermediate paths,
  column prefixes, out_stems) are derived in the new config, so they are
  dropped.
- Keys no code reads are dropped too.

Every dropped key is listed in `Conversion.notes`, so nothing disappears
silently. The key-by-key inventory this follows is in docs/explore1.md
(cv_models and the prediction scripts), docs/explore2.md (combine and connect)
and docs/explore3.md (regrid, raw_data, climatology).

    python -m onset_blending.legacy_specs --name india \\
        --ground-truth archive/specs/raw_data/gt.yml --forecast aifs=archive/specs/raw_data/aifs.yml \\
        --combine archive/specs/combine/c.yml --connect archive/specs/2025_blend/connect.yml \\
        --cv-models archive/specs/2025_blend/cv_models.yml -o configs/india.yml
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .. import config as cfg
from ..config import BlendConfig, ConfigError

__all__ = ["Conversion", "convert_specs", "convert_spec_files"]

DERIVED = "derived from the config"
UNUSED = "no code reads it"
RUNTIME = "a run setting, not part of the blend"

# Old defaults that differ from the new config's, written out explicitly.
_OLD_TRAINING_YEARS = list(range(2000, 2025))
_OLD_FIXED_CUTOFF = "06-02"
_OLD_FORECAST_VARIANTS = {"base": "", "ref_onset": "_ref", "fixed_cutoff": "_fixed_cutoff"}
_CONNECT_VARIANTS = {"fixed_cutoff": "fixed_cutoff", "ref": "ref_onset"}
# Combine daily columns the connect stage reads, by raw output column.
_DAILY_OUT = {
    "predicted_prob": "p_onset",
    "predicted_prob_fixed_cutoff": "p_onset_fixed_cutoff",
    "predicted_prob_ref": "p_onset_ref",
    "forecast_rain": "rain_mean",
}


@dataclass(frozen=True)
class Conversion:
    config: BlendConfig
    notes: tuple[str, ...]

    def to_yaml(self, sources: dict[str, str] | None = None) -> str:
        """The config as YAML, headed by where it came from and what was dropped."""
        lines = ["# Converted from the stage specs by onset_blending.legacy_specs."]
        for label, path in (sources or {}).items():
            lines.append(f"#   {label}: {path}")
        if self.notes:
            lines.append("# Not carried over:")
            lines += [f"#   - {note}" for note in self.notes]
        return "\n".join(lines) + "\n" + self.config.to_yaml()


class _Spec:
    """One old spec that remembers which keys the conversion has read."""

    def __init__(self, label: str, data: dict | None, notes: list[str]):
        self.label = label
        self.data = data or {}
        self.read: set[str] = set()
        self._notes = notes

    def get(self, path: str, default=None):
        self.read.add(path)
        node = self._lookup(path)
        return default if node is None else node

    def has(self, path: str) -> bool:
        return self._lookup(path) is not None

    def drop(self, path: str, reason: str) -> None:
        if self.has(path):
            self.read.add(path)
            self._notes.append(f"{self.label}: {path} ({reason})")

    def unread(self) -> list[str]:
        """Leaf keys nothing read; reading a key covers everything below it."""
        leaves: list[str] = []

        def walk(node, path):
            if isinstance(node, dict) and node:
                for key, value in node.items():
                    walk(value, f"{path}.{key}" if path else str(key))
            elif path and not self._covered(path):
                leaves.append(path)

        walk(self.data, "")
        return leaves

    def _covered(self, path: str) -> bool:
        parts = path.split(".")
        return any(".".join(parts[:i]) in self.read for i in range(1, len(parts) + 1))

    def _lookup(self, path: str):
        node = self.data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node


def _same_path(a: str | None, b: str | None) -> bool:
    return a is not None and b is not None and os.path.normpath(a) == os.path.normpath(b)


def _without_none(mapping: dict) -> dict:
    return {k: v for k, v in mapping.items() if v is not None}


class _Converter:
    def __init__(self, *, name, ground_truth, forecasts, combine, connect, cv_models, regrid, connect_onset_spec,
                 data_root=None):
        self.problems: list[str] = []
        self.notes: list[str] = []
        self.name = name
        self.data_root = data_root
        self.gt = _Spec("ground_truth", ground_truth, self.notes)
        self.forecasts = {m: _Spec(f"forecast[{m}]", spec, self.notes) for m, spec in forecasts.items()}
        self.combine = _Spec("combine", combine, self.notes)
        self.connect = _Spec("connect", connect, self.notes)
        self.cv = _Spec("cv_models", cv_models, self.notes)
        self.regrid = _Spec("regrid", regrid, self.notes) if regrid is not None else None
        self.connect_onset = connect_onset_spec

    # -- helpers -----------------------------------------------------------

    def agree(self, what: str, values: dict[str, Any]):
        """One fact stated in several specs: the stated values must match."""
        stated = {where: v for where, v in values.items() if v is not None}
        distinct = []
        for value in stated.values():
            if value not in distinct:
                distinct.append(value)
        if len(distinct) > 1:
            listed = "; ".join(f"{where} = {value!r}" for where, value in stated.items())
            self.problems.append(f"{what} disagrees between specs: {listed}")
        return distinct[0] if distinct else None

    def years_between(self, spec: _Spec) -> tuple[int, ...]:
        low, high = spec.get("options.min_year"), spec.get("options.max_year")
        if low is None and high is None:
            return ()
        if low is None or high is None:
            self.problems.append(
                f"{spec.label}: options.min_year and max_year must be given together to convert"
            )
            return ()
        return tuple(range(int(low), int(high) + 1))

    @property
    def raw_specs(self) -> list[_Spec]:
        return [self.gt, *self.forecasts.values()]

    # -- onset -------------------------------------------------------------

    @staticmethod
    def onset_definition(spec: _Spec | dict) -> dict:
        """The onset definition as read_onset_params resolves it."""
        if isinstance(spec, dict):
            spec = _Spec("onset", spec, [])
        wet = float(spec.get("options.onset_definition.wet_day_min_mm", 1.0))
        base = "options.onset_definition"
        return {
            "window": int(spec.get("options.window", 5)),
            "trigger_rule": str(spec.get(f"{base}.trigger_rule", "all_days_wet")).lower(),
            "wet_day_min_mm": wet,
            "wet_day_comparison": str(spec.get(f"{base}.wet_day_comparison", "gte")).lower(),
            "follow_days": int(spec.get(f"{base}.follow_days", 21)),
            "followup_anchor": str(spec.get(f"{base}.followup_anchor", "after_trigger")).lower(),
            "dry_spell": {
                "mode": str(spec.get(f"{base}.dry_spell.mode", "consecutive_dry")),
                "min_dry_days": int(spec.get(f"{base}.dry_spell.min_dry_days", 7)),
                "dry_day_min_mm": float(spec.get(f"{base}.dry_spell.dry_day_min_mm", wet)),
                "sum_window": int(spec.get(f"{base}.dry_spell.sum_window", 10)),
                "sum_min_mm": float(spec.get(f"{base}.dry_spell.sum_min_mm", 5.0)),
            },
        }

    def threshold(self, spec: _Spec) -> dict | None:
        """The threshold as onset_utils resolves it, in the new config's shape."""
        rule = spec.get("thresholds.rule") or {}
        rule_type = str(rule.get("type", "")).lower()
        if spec.get("thresholds.constant") is not None:
            return {"mm": float(spec.get("thresholds.constant"))}
        if rule_type == "constant":
            return {"mm": float(rule["value"])}
        source = spec.get("thresholds.file")
        if source is None:
            return None
        scale = None
        if rule_type == "scale":
            scale = _without_none(
                {k: rule.get(k) for k in ("factor", "offset", "min", "max")}
            )
        try:
            value = float(source)
        except (TypeError, ValueError):
            value = None
        if value is not None:
            if scale:
                value = value * float(scale.get("factor", 1.0)) + float(scale.get("offset", 0.0))
                value = max(value, scale["min"]) if scale.get("min") is not None else value
                value = min(value, scale["max"]) if scale.get("max") is not None else value
            return {"mm": value}
        out: dict[str, Any] = {"file": source}
        t = "thresholds."
        ext = os.path.splitext(str(source))[1].lower()
        before = set(spec.read)
        if ext in (".nc", ".nc4", ".netcdf"):
            out["lat_var"] = spec.get(t + "lat_var") or spec.get(t + "lat_col") or "lat"
            out["lon_var"] = spec.get(t + "lon_var") or spec.get(t + "lon_col") or "lon"
            out["value_var"] = spec.get(t + "thresh_var") or spec.get(t + "thresh_col") or "MWmean"
        elif ext == ".mat":
            out["mat_lat_name"] = spec.get(t + "mat_lat_name", "lat")
            out["mat_lon_name"] = spec.get(t + "mat_lon_name", "lon")
            out["mat_value_name"] = spec.get(t + "mat_thresh_name", "onset.day.thres")
        else:
            out["id_col"] = spec.get(t + "id_col") or spec.get(t + "adm3_col")
            out["value_col"] = spec.get(t + "thresh_col", "onset_thresh")
            out["lat_col"] = spec.get(t + "lat_col", "lat")
            out["lon_col"] = spec.get(t + "lon_col", "lon")
        used = spec.read - before
        for key in ("thresh_var", "lat_var", "lon_var", "mat_lat_name", "mat_lon_name",
                    "mat_thresh_name", "id_col", "adm3_col", "thresh_col", "lat_col", "lon_col"):
            if t + key not in used:
                spec.drop(t + key, f"does not apply to a {ext or 'CSV'} threshold file")
        if scale:
            out["scale"] = scale
        return _without_none(out)

    @staticmethod
    def reference_onset(spec: _Spec) -> dict | None:
        month_day = spec.get("ref_onset.constant_month_day") or spec.get("ref_onset.date")
        if month_day:
            for key in ("file", "date_col", "year_col", "day_col", "base_date", "unit_col"):
                spec.drop(f"ref_onset.{key}", "a constant month-day takes precedence")
            return {"month_day": str(month_day)}
        if not spec.get("ref_onset.file"):
            return None
        return _without_none(
            {
                "file": spec.get("ref_onset.file"),
                "date_col": spec.get("ref_onset.date_col"),
                "year_col": spec.get("ref_onset.year_col"),
                "day_col": spec.get("ref_onset.day_col"),
                "base_date": spec.get("ref_onset.base_date"),
                "unit_col": spec.get("ref_onset.unit_col"),
            }
        )

    def onset(self, uses_ref_series: bool) -> dict:
        definitions = {s.label: self.onset_definition(s) for s in self.raw_specs}
        definition = self.agree("the onset definition (options.window, options.onset_definition)", definitions)
        self.definition = definition
        thresholds = {s.label: self.threshold(s) for s in self.raw_specs}
        threshold = self.agree("the onset threshold", thresholds)
        if threshold is None:
            self.problems.append(
                "no onset threshold in any raw_data spec: the old pipeline then finds no onsets at all"
            )

        references = {s.label: self.reference_onset(s) for s in self.raw_specs}
        reference = references[self.gt.label]
        stated = {k: v for k, v in references.items() if v is not None}
        if reference is None and stated:
            self.problems.append(
                "ref_onset is set in " + ", ".join(stated) + " but not in ground_truth; one reference "
                "onset now sets both the observed onset search start and the forecasts' ref series"
            )
        elif reference is not None:
            missing = [s.label for s in self.forecasts.values() if references[s.label] is None]
            if missing and uses_ref_series:
                self.problems.append(
                    f"ref_onset is missing from {', '.join(missing)}, whose ref series the evaluation "
                    "uses; there it equals the plain series, while the new config computes it"
                )
            elif missing:
                self.notes.append(
                    f"{', '.join(missing)}: no ref_onset, so their unused ref series was the plain series"
                )
            self.agree("ref_onset", {k: v for k, v in references.items() if v is not None})

        season_start = self.gt.get("options.cutoff_month_day")
        if season_start is None:
            self.problems.append("ground_truth: options.cutoff_month_day is required")
        for spec in self.forecasts.values():
            spec.drop("options.cutoff_month_day", "only ground-truth processing reads it")
        fixed = self.agree(
            "options.fixed_cutoff_month_day",
            {s.label: str(s.get("options.fixed_cutoff_month_day", _OLD_FIXED_CUTOFF)) for s in self.forecasts.values()},
        )
        gt_fixed = self.gt.get("options.fixed_cutoff_month_day")
        if gt_fixed is not None:
            self.notes.append(
                "ground_truth: options.fixed_cutoff_month_day (only forecast processing reads it"
                + (")" if str(gt_fixed) == fixed else f"; it said {gt_fixed!r}, the forecasts {fixed!r})")
            )

        out: dict[str, Any] = {"threshold": threshold, "season_start": season_start}
        if fixed is not None:
            out["fixed_cutoff"] = fixed
        if reference is not None:
            out["reference_onset"] = reference
        if definition is not None:
            definition = {**definition, "dry_spell": dict(definition["dry_spell"])}
            if definition["dry_spell"]["dry_day_min_mm"] == definition["wet_day_min_mm"]:
                del definition["dry_spell"]["dry_day_min_mm"]
            out["definition"] = definition
        return out

    # -- inputs ------------------------------------------------------------

    def files(self, spec: _Spec, regrid_source: dict | None, years: tuple[int, ...]) -> dict:
        folder = spec.get("input.nc_folder")
        pattern = spec.get("input.file_regex")
        if regrid_source is not None:
            # The regrid writes <stem>_adm3.nc beside each source file; the new
            # pipeline regrids in memory, so it reads the sources.
            folder = regrid_source.get("nc_folder")
            pattern = regrid_source.get("file_regex") or r"\.nc$"
        out = {"folder": folder, "pattern": pattern}
        if years:
            out["years"] = list(years)
        return out

    @staticmethod
    def dims(spec: _Spec) -> dict:
        rename = dict(spec.get("dimensions.rename", {}) or {})
        day_dim = spec.get("input.wide_day_dim", "day")
        if day_dim != "day":
            rename = {old: ("day" if new == day_dim else new) for old, new in rename.items()}
        return {str(k): str(v) for k, v in rename.items()}

    def cell_transform(self, spec: _Spec) -> str | None:
        enabled = bool(spec.get("options.cell_transform_enabled", False))
        if enabled:
            path = spec.get("options.cell_transform_file")
            if path is None:
                self.problems.append(f"{spec.label}: options.cell_transform_enabled needs cell_transform_file")
            return path
        spec.drop("options.cell_transform_file", "cell_transform_enabled is false")
        return None

    def regrid_entry(self, folder: str | None) -> dict | None:
        if self.regrid is None:
            return None
        for entry in self.regrid.get("forecasts", []) or []:
            if _same_path(entry.get("nc_folder"), folder):
                return entry
        return None

    def observations(self) -> dict:
        gt = self.gt
        regrid_source = None
        if self.regrid is not None:
            regrid_source = {
                "nc_folder": self.regrid.get("ground_truth.nc_folder"),
                "file_regex": self.regrid.get("ground_truth.file_regex"),
            }
            if not _same_path(regrid_source["nc_folder"], gt.get("input.nc_folder")):
                self.problems.append(
                    "regrid ground_truth.nc_folder and ground_truth input.nc_folder differ "
                    f"({regrid_source['nc_folder']!r} vs {gt.get('input.nc_folder')!r}); the raw_data "
                    "spec reads the regridded files beside the sources"
                )
            regrid_variable = self.regrid.get("ground_truth.value_col")
            if regrid_variable and str(regrid_variable).lower() != str(gt.get("input.value_col")).lower():
                self.problems.append(
                    f"regrid ground_truth.value_col {regrid_variable!r} is not ground_truth "
                    f"input.value_col {gt.get('input.value_col')!r}"
                )
        missing = gt.get("options.missing_rain_policy") or gt.get("input.missing_rain_policy") or "keep"
        out = {
            "files": self.files(gt, regrid_source, self.years_between(gt)),
            "variable": gt.get("input.value_col"),
            "dims": self.dims(gt),
            "missing_rain": str(missing).lower(),
            "cell_transform": self.cell_transform(gt),
        }
        return _without_none(out)

    def models(self, connect_models: list[dict], onset_definition: dict | None) -> list[dict]:
        combine_models = self.combine.get("forecasts", {}) or {}
        names = [str(entry.get("name")) for entry in connect_models]
        for name in self.forecasts:
            if name not in names:
                self.problems.append(f"forecast[{name}] is not in connect forecast_models ({', '.join(names)})")
        for name in combine_models:
            if name not in names:
                self.notes.append(f"combine: forecasts.{name} (connect never reads it)")
        top_weights = self.regrid.get("weights_in") if self.regrid is not None else None
        token_source = self.connect_token_definition()
        models = []
        for index, entry in enumerate(connect_models):
            name = str(entry.get("name"))
            raw = self.forecasts.get(name)
            combined = combine_models.get(name)
            if raw is None:
                self.problems.append(f"connect forecast_models[{index}] {name!r} has no raw_data spec")
                continue
            if combined is None:
                self.problems.append(f"connect forecast_models[{index}] {name!r} has no combine forecasts entry")
                combined = {}
            model: dict[str, Any] = {"name": name}

            years = self.years_between(raw)
            sources = combined.get("sources") or []
            if len(sources) > 1:
                self.problems.append(
                    f"combine forecasts.{name}: {len(sources)} sources; the new config reads each "
                    "model from one set of files"
                )
            if sources and sources[0].get("years") not in (None, ""):
                parse_problems: list[str] = []
                combine_years = cfg._years(sources[0]["years"], f"combine forecasts.{name}.sources[0].years", parse_problems)
                self.problems += parse_problems
                if combine_years:
                    years = tuple(sorted(set(years) & set(combine_years))) if years else combine_years
            if any("prob_col" in source for source in sources):
                self.notes.append(f"combine: forecasts.{name}.sources[].prob_col ({UNUSED})")

            regrid_source = self.regrid_entry(raw.get("input.nc_folder"))
            if self.regrid is not None and regrid_source is None:
                self.notes.append(
                    f"forecast[{name}]: the regrid spec does not list its folder, so its files are "
                    "read as they are (already on the units)"
                )
            model["files"] = self.files(raw, regrid_source, years)
            model["variable"] = raw.get("input.value_col")
            model["dims"] = self.dims(raw)
            if raw.get("filter.max_number") is not None:
                model["max_member"] = int(raw.get("filter.max_number"))

            min_day, max_day = raw.get("options.min_day"), raw.get("options.max_day")
            if min_day is None or max_day is None:
                self.problems.append(f"forecast[{name}]: options.min_day and max_day are required")
            else:
                model["lead_days"] = {"min": int(min_day), "max": int(max_day)}
                combine_max = combined.get("max_day")
                if combine_max is not None and int(combine_max) < int(max_day):
                    self.problems.append(
                        f"combine forecasts.{name}.max_day ({combine_max}) is below forecast[{name}] "
                        f"options.max_day ({max_day}); the old combine stops on that"
                    )

            rain_day_max = self.agree(
                f"{name}: rain_day_max",
                {f"forecast[{name}] options": raw.get("options.rain_day_max"), "connect": entry.get("rain_day_max")},
            )
            policy = self.agree(
                f"{name}: rain_horizon_policy",
                {
                    f"forecast[{name}] options": _lower(raw.get("options.rain_horizon_policy")),
                    "connect": _lower(entry.get("rain_horizon_policy")),
                },
            )
            horizon = _without_none({"max_day": rain_day_max, "policy": policy})
            if horizon.get("policy") == ("exact" if rain_day_max is not None else "legacy"):
                del horizon["policy"]
            if horizon:
                model["rain_horizon"] = horizon

            predictors = []
            for p_index, predictor in enumerate(entry.get("rain_predictors") or []):
                parsed = self.rain_predictor(predictor, token_source, onset_definition, f"connect forecast_models[{index}].rain_predictors[{p_index}]")
                if parsed is not None:
                    predictors.append(parsed)
            if predictors:
                model["rain_predictors"] = predictors
            transform = entry.get("rain_transform")
            if transform is not None:
                model["rain_transform"] = _rain_transform(transform)
            for variant in entry.get("variants") or []:
                if variant not in _CONNECT_VARIANTS:
                    self.problems.append(
                        f"connect forecast_models[{index}].variants: {variant!r} is not one of "
                        f"{', '.join(_CONNECT_VARIANTS)}"
                    )
            model["cell_transform"] = self.cell_transform(raw)
            weights = (regrid_source or {}).get("weights_in") or top_weights
            if weights is not None:
                model["regrid_weights"] = weights
            self.check_combine_columns(name, combined, first=index == 0)
            models.append(_without_none(model))
        return models

    def check_combine_columns(self, name: str, combined: dict, first: bool) -> None:
        for item in combined.get("daily") or []:
            col, out = item.get("col"), item.get("out")
            expected = _DAILY_OUT.get(col)
            if expected is None:
                if out in _DAILY_OUT.values():
                    self.problems.append(
                        f"combine forecasts.{name}.daily: {col!r} is named {out!r}, which connect reads; "
                        "the new pipeline names each series by what it is"
                    )
                else:
                    self.notes.append(f"combine: forecasts.{name}.daily {col} -> {out} (connect never reads it)")
            elif out != expected:
                self.problems.append(
                    f"combine forecasts.{name}.daily maps {col!r} to {out!r}; connect reads it as {expected!r}"
                )
        constants = {item.get("col"): item.get("out") for item in combined.get("constants") or []}
        for col, out in constants.items():
            if col == "onset_thresh" and out != "onset_thresh":
                self.problems.append(f"combine forecasts.{name}.constants maps onset_thresh to {out!r}")
            elif col not in ("onset_thresh", "ref_onset_date"):
                self.notes.append(f"combine: forecasts.{name}.constants {col} (connect never reads it)")
        if first and "onset_thresh" not in constants:
            self.problems.append(
                f"combine forecasts.{name} (connect's first model) has no onset_thresh constant, "
                "which the old connect stage reads the threshold feature from"
            )
        if any(item.get("add_plus") for item in combined.get("daily") or []):
            self.notes.append(f"combine: forecasts.{name}.daily[].add_plus (connect never reads the plus columns)")

    def connect_token_definition(self) -> dict:
        """The onset definition connect resolved window tokens against."""
        if self.connect.get("onset_spec"):
            if self.connect_onset is not None:
                return self.onset_definition(self.connect_onset)
            return self.onset_definition(_Spec("onset", self.gt.data, []))
        inline = {"options": _without_none({
            "window": self.connect.get("options.window"),
            "onset_definition": self.connect.get("options.onset_definition"),
        })}
        return self.onset_definition(inline)

    def rain_predictor(self, predictor, tokens: dict, onset_definition: dict | None, where: str) -> dict | None:
        if isinstance(predictor, dict):
            agg = str(predictor.get("agg", "")).lower()
            window = predictor.get("window", "trigger")
        else:
            text = str(predictor)
            agg, _, rest = text.partition("_")
            last = text.rsplit("_", 1)[-1]
            digits = re.fullmatch(r"(\d+)day", last)
            window = int(digits.group(1)) if digits else last
        if agg not in ("diff", "max", "min"):
            self.problems.append(f"{where}: agg {agg!r} is not diff, max or min")
            return None
        if isinstance(window, str) and window.strip().isdigit():
            window = int(window)
        if isinstance(window, str):
            token = window.lower()
            try:
                resolved = cfg.resolve_window(cfg.RainPredictor(agg=agg, window=token), _definition(tokens))
            except ValueError:
                self.problems.append(f"{where}: window {window!r} is not a number or a known token")
                return None
            if onset_definition is not None:
                now = cfg.resolve_window(cfg.RainPredictor(agg=agg, window=token), _definition(onset_definition))
                if now != resolved:
                    return {"agg": agg, "window": resolved}
            return {"agg": agg, "window": token}
        return {"agg": agg, "window": int(window)}

    # -- domain ------------------------------------------------------------

    def grid_ids(self) -> dict:
        def read(spec: _Spec | None) -> tuple | None:
            if spec is None:
                return None
            base = "geometry." if isinstance(spec.data.get("geometry"), dict) else ""
            digits = spec.get(base + "grid_id_decimal_digits")
            fmt = spec.get(base + "grid_id_format")
            if digits is None and fmt is None:
                return None
            return (digits, str(fmt).lower() if fmt is not None else "trimmed")

        stated = {s.label: read(s) for s in [self.regrid, *self.raw_specs, self.connect, self.cv] if s is not None}
        agreed = self.agree("the grid-id convention (grid_id_decimal_digits, grid_id_format)", stated)
        if agreed is None:
            return {}
        return _without_none({"decimal_digits": agreed[0], "format": agreed[1] if agreed[1] != "trimmed" else None})

    def domain(self) -> dict:
        out: dict[str, Any] = {}
        grid_ids = self.grid_ids()
        if grid_ids:
            out["grid_ids"] = grid_ids

        raw_dissemination = {}
        for spec in self.raw_specs:
            path = spec.get("filter.dissemination_cells_file")
            if path is not None:
                id_col = spec.get("filter.id_col") or spec.get("filter.dissemination_id_col")
                raw_dissemination[spec.label] = _without_none({"file": path, "id_col": id_col})
        cv_file = self.cv.get("cell.dissemination")
        cv_dissemination = None
        if cv_file:
            cv_id = self.cv.get("cell.id_col") or self.cv.get("cell.dissemination_id_col")
            cv_dissemination = _without_none({"file": cv_file, "id_col": cv_id})
        else:
            self.cv.drop("cell", "cell.dissemination is empty")
        stated = dict(raw_dissemination)
        if cv_dissemination is not None:
            stated["cv_models cell"] = cv_dissemination
        dissemination = self.agree("the dissemination units", stated)
        if dissemination is not None:
            out["dissemination"] = dissemination
            if not raw_dissemination:
                out["dissemination_scope"] = "training"
        if self.regrid is not None:
            report_file = self.regrid.get("dissemination_cells_file")
            self.regrid.get("dissemination_id_col")
            if report_file is not None and not _same_path(report_file, (dissemination or {}).get("file")):
                self.notes.append(
                    f"regrid: dissemination_cells_file {report_file!r} (only the coverage report used it, "
                    "and it is not the blend's dissemination file)"
                )

        bboxes = {s.label: s.get("filter.bbox") for s in self.raw_specs}
        bbox = self.agree("filter.bbox", bboxes)
        if bbox:
            out["area_of_interest"] = {"bbox": {k: float(v) for k, v in bbox.items() if v is not None}}

        centroids = {}
        for spec in self.raw_specs:
            path = spec.get("filter.centroids_file")
            if path is not None:
                centroids[spec.label] = _without_none({"file": path, "id_col": spec.get("filter.centroids_id_col")})
        centroid_table = self.agree("filter.centroids_file", centroids)
        written = self.regrid.get("centroids_out") if self.regrid is not None else None
        if centroid_table is not None and _same_path(centroid_table["file"], written):
            self.notes.append("ground_truth: filter.centroids_file (the regrid writes it)")
        elif centroid_table is not None:
            out["centroids"] = centroid_table

        if self.regrid is not None:
            out["regrid"] = self.regrid_section()
        return out

    def regrid_section(self) -> dict:
        r = self.regrid
        out = {
            "shapefile": r.get("geometry.shapefile"),
            "unit_id_col": r.get("geometry.region_id_col") or r.get("geometry.region_key_col") or "adm3_name",
            "unit_name_col": r.get("geometry.region_name_col"),
            "parent_id_col": r.get("geometry.parent_key_col"),
            "crs": r.get("geometry.crs", "EPSG:4326"),
            "grid_lat_var": r.get("geometry.grid_lat_var"),
            "grid_lon_var": r.get("geometry.grid_lon_var"),
            "clip_to_coverage": bool(r.get("clip_to_coverage", True)),
            "weights": r.get("ground_truth.weights_in") or r.get("weights_in"),
        }
        r.drop("geometry.region_label", UNUSED)
        for key in ("weights_out", "centroids_out", "report_out"):
            r.drop(key, "written by result.save")
        if any("value_col" in entry for entry in r.get("forecasts", []) or []):
            self.notes.append("regrid: forecasts[].value_col (it only named a weights file)")
        return _without_none(out)

    # -- climatology and features -------------------------------------------

    def climatology(self, window_tags: list) -> tuple[dict, list[tuple[int, int]]]:
        runs = self.gt.get("climatologies", {}) or {}
        prefixes: dict[str, str] = {}
        for key, entry in (self.combine.get("input.climatologies", {}) or {}).items():
            prefixes[(entry or {}).get("out_prefix") or f"{key}_p_onset"] = key
        if not prefixes and self.combine.get("input.climatology_rds"):
            prefixes["clim_p_onset"] = "clim"
        base_prefix = self.connect.get("climatology.base_prefix", "clim")
        unc_prefix = self.connect.get("climatology.unconditional_prefix", "clim_unc")

        def run_for(prefix: str, what: str) -> tuple[str, dict] | None:
            key = prefixes.get(f"{prefix}_p_onset")
            if key is None:
                self.problems.append(
                    f"connect reads the {what} climatology as {prefix}_p_onset_*, which no combine "
                    "input.climatologies entry writes"
                )
                return None
            if key not in runs:
                self.problems.append(f"combine climatology {key!r} has no ground_truth climatologies.{key} run")
                return None
            return key, runs[key]

        base = run_for(base_prefix, "conditional")
        unc = run_for(unc_prefix, "unconditional")
        used = {k for k, _ in (base, unc) if k}
        out: dict[str, Any] = {}
        if base is None or unc is None:
            return out, []
        base_run, unc_run = base[1], unc[1]
        if not base_run.get("conditional", True):
            self.problems.append(f"climatologies.{base[0]} feeds the conditional climatology but is unconditional")
        if unc_run.get("conditional", True):
            self.problems.append(f"climatologies.{unc[0]} feeds the unconditional climatology but is conditional")

        def shared(run: dict) -> dict:
            horizons = run.get("horizons")
            return {
                "onset_col": run.get("onset_col", "onset_day"),
                "season_start": str(run.get("season_start_md")),
                "issue_end": str(run.get("issue_end_md")),
                "forecast_window": run.get("forecast_window"),
                "horizons": [
                    {"start": str(h["start_md"]), "end": str(h["end_md"]), "forecast_window": int(h["forecast_window"])}
                    for h in horizons
                ] if horizons else None,
                "cv_by_year": bool(run.get("cv_by_year", True)),
                "min_onset_years": int(run.get("min_onset_years", 10)),
                "test_years": (run.get("test_year_min"), run.get("test_year_max")),
            }

        def train_years(run: dict) -> tuple:
            return (run.get("train_year_min"), run.get("train_year_max"))

        settings = self.agree(
            "climatology settings",
            {f"climatologies.{base[0]}": shared(base_run), f"climatologies.{unc[0]}": shared(unc_run)},
        )
        self.agree(
            "climatology training years",
            {f"climatologies.{base[0]}": train_years(base_run), f"climatologies.{unc[0]}": train_years(unc_run)},
        )
        windows: list[tuple[int, int]] = []
        for tag in window_tags:
            match = re.fullmatch(r"(\d{4})_(\d{4})", str(tag))
            if match is None:
                hint = " (an unquoted 2000_2024 in YAML reads as the number 20002024; quote it)" if isinstance(tag, int) else ""
                self.problems.append(f"connect climatology.window_tags: {tag!r} is not '<start>_<end>'{hint}")
                continue
            window = (int(match.group(1)), int(match.group(2)))
            found = run_for(f"{base_prefix}_{tag}", f"{tag} windowed")
            if found is None:
                continue
            used.add(found[0])
            self.agree(f"climatology settings for window {tag}", {"main": settings, f"climatologies.{found[0]}": shared(found[1])})
            if train_years(found[1]) != window:
                self.problems.append(
                    f"climatologies.{found[0]} trains on {train_years(found[1])}, but its tag says {window}"
                )
            windows.append(window)
        for key in runs:
            if key not in used:
                self.notes.append(f"ground_truth: climatologies.{key} (connect never reads it)")
        for key in ("paths.climatology_out_dir", "paths.climatology_out_stem", "input.gt_path"):
            self.gt.drop(key, DERIVED)

        if settings is None:
            return out, windows
        if settings["onset_col"] != "onset_day":
            self.problems.append(f"climatologies use onset_col {settings['onset_col']!r}; only onset_day is supported")
        if settings["season_start"] != str(self.gt.get("options.cutoff_month_day")):
            self.problems.append(
                f"climatologies season_start_md {settings['season_start']!r} is not ground_truth "
                f"options.cutoff_month_day {self.gt.get('options.cutoff_month_day')!r} (both set the onset-day origin)"
            )
        out = {
            "issue_end": settings["issue_end"],
            # The default is features.probability_days; say it only when it differs.
            "forecast_window": None
            if settings["forecast_window"] == int(self.connect.get("day_max", 28))
            else settings["forecast_window"],
            "horizons": settings["horizons"],
            "cv_by_year": settings["cv_by_year"],
            "min_onset_years": settings["min_onset_years"],
            "train_years": list(train_years(base_run)),
            "test_years": list(settings["test_years"]),
            "windows": [list(w) for w in windows] or None,
            "workers": self.gt.get("climatology_runtime.workers"),
        }
        return _without_none(out), windows

    def features(self) -> dict:
        c = self.connect
        n_bins = c.get("n_bins") if c.has("n_bins") else c.get("n_weeks", 4)
        cv_bins = self.cv.get("n_bins")
        self.agree("the number of week bins", {"connect": int(n_bins), "cv_models": cv_bins})
        days = c.get("days_per_bin") if c.has("days_per_bin") else c.get("days_per_week", 7)
        join = self.combine.get("options.join", "inner")
        if join == "full":
            self.problems.append(
                "combine: options.join 'full' (an outer join of the day-level table) has no counterpart: "
                "the samples are the units and issue dates every source covers"
            )
        elif join != "inner":
            self.notes.append(f"combine: options.join {join!r} (the old combine read anything but 'full' as inner)")
        return {
            "bins": {"count": int(n_bins), "days": int(days)},
            "probability_days": int(c.get("day_max", 28)),
            "trim_after_true_onset": bool(self.combine.get("options.trim_forecasts_after_true_onset", False)),
            "rain_transform": _rain_transform(c.get("rain_transform", "identity")),
        }

    # -- evaluation ----------------------------------------------------------

    def rename_prefix(self, text: str) -> str:
        prefix = self.connect.get("climatology.output_prefix", cfg.CLIMATOLOGY_FEATURE)
        if prefix == cfg.CLIMATOLOGY_FEATURE:
            return text
        return re.sub(rf"(?<![A-Za-z0-9_]){re.escape(prefix)}(?=_)", cfg.CLIMATOLOGY_FEATURE, text)

    def evaluation(self, windows: list[tuple[int, int]]) -> dict:
        cv = self.cv
        if cv.has("run.input_rds_override"):
            self.problems.append("cv_models: run.input_rds_override was removed (the old code rejects it)")
        cv.drop("run.MR", UNUSED)
        cv.drop("run.pipeline_input_dir", DERIVED)
        cv.drop("run.pipeline_output_dir", "set outputs.dir instead")
        years = {
            "training": list(cv.get("run.training_years", _OLD_TRAINING_YEARS)),
            "cv_holdout": list(cv.get("run.cv_holdout_years", [])),
            "true_holdout": list(cv.get("run.true_holdout_years", [])),
        }
        out: dict[str, Any] = {
            "years": years,
            "cutoff_mode": cv.get("run.cutoff_mode", "ref_onset"),
            "cv_methods": self.cv_methods(),
            "cluster_var": cv.get("cv.cluster_var", "cluster"),
        }
        formulas = {}
        for name, entry in (cv.get("models.formulas", {}) or {}).items():
            entry = entry or {}
            formulas[str(name)] = {
                "text": self.rename_prefix(str(entry.get("text", ""))),
                "enabled": bool(entry.get("enabled", True)),
            }
        out["formulas"] = formulas
        out["window_variants"] = self.window_variants(windows)
        out["climatology_baselines"] = self.climatology_baselines()
        out["forecasts"] = self.forecast_evaluations()
        out["mme"] = self.mme()
        external = cv.get("extras.external_predictions")
        if external:
            out["external_predictions"] = [_without_none(dict(e)) for e in external]
        return out

    def cv_methods(self) -> list[str]:
        methods = []
        for method in self.cv.get("cv.methods", ["global"]):
            method = str(method)
            if method in ("global", "local"):
                methods.append(method)
            elif method.startswith("cluster"):
                methods.append("clusters")
            elif method == "neighbors":
                self.problems.append("cv_models cv.methods: 'neighbors' is not implemented (the old code raises)")
            else:
                self.notes.append(f"cv_models: cv.methods {method!r} (the old code warned and skipped it)")
        return methods

    def window_variants(self, windows: list[tuple[int, int]]) -> list[dict]:
        raw = self.cv.get("models.window_variants")
        if not raw:
            return []
        entries = raw if isinstance(raw, list) else [raw]
        out = []
        for index, entry in enumerate(entries):
            where = f"cv_models models.window_variants[{index}]"
            noted = f"cv_models: models.window_variants[{index}]"
            if not entry.get("enabled", False):
                self.notes.append(f"{noted} (disabled)")
                continue
            replace = entry.get("replace") or {}
            source, target = str(replace.get("from", "")), str(replace.get("to", ""))
            if not source.endswith("_qx") or target != source[:-3] + "{window}_qx":
                self.problems.append(
                    f"{where}.replace: only 'X_qx' -> 'X{{window}}_qx' is supported, got {source!r} -> {target!r}"
                )
                continue
            end = entry.get("end_year")
            variant_windows = [[int(s), int(end)] for s in entry.get("start_years") or [] if int(s) != 1900]
            for window in variant_windows:
                if tuple(window) not in windows:
                    self.problems.append(
                        f"{where}: window {window[0]}_{window[1]} needs that climatology "
                        "(connect climatology.window_tags)"
                    )
            if entry.get("include_baseline_1900") is not None:
                self.notes.append(f"{noted}.include_baseline_1900 ({UNUSED})")
            self.notes.append(
                f"{noted}.replace (the old replace never matched, so the variant repeated its base formula; "
                "it now swaps the windowed climatology in)"
            )
            out.append(_without_none({
                "base": entry.get("base_name"),
                "windows": variant_windows,
                "replace": self.rename_prefix(source[:-3]),
                "only_if_cutoff_mode": entry.get("only_if_cutoff_mode"),
            }))
        return out

    def climatology_baselines(self) -> dict:
        prefix = self.connect.get("climatology.output_prefix", cfg.CLIMATOLOGY_FEATURE)
        tasks = self.cv.get("extras.clim_logits", []) or []
        out = {"unconditional": True, "conditional": False, "windows": []}
        names = [t.get("name") for t in tasks]
        if "unc_clim_raw" not in names:
            self.notes.append("cv_models: no unc_clim_raw clim_logits task; it is always scored now (the pooled summary needs it)")
        for index, task in enumerate(tasks):
            name, column = task.get("name"), task.get("base_col_prefix")
            where = f"cv_models extras.clim_logits[{index}]"
            if name == "unc_clim_raw":
                if column != f"{prefix}_unc" or task.get("earlier_col") != f"{prefix}_unc_earlier" or not task.get("earlier_is_logit", True):
                    self.problems.append(
                        f"{where}: unc_clim_raw must read {prefix}_unc with earlier_col {prefix}_unc_earlier "
                        "as logits, which connect writes"
                    )
            elif name == "clim_raw":
                if column != prefix:
                    self.problems.append(f"{where}: clim_raw must read {prefix}, which connect writes")
                starts = [int(y) for y in task.get("window_start_years") or []]
                end = task.get("window_end_year")
                if not starts or end is None:
                    out["conditional"] = True
                    continue
                for start in starts:
                    if start == 1900:
                        out["conditional"] = True
                    else:
                        out["windows"].append([start, int(end)])
            else:
                self.problems.append(
                    f"{where}: {name!r} is not unc_clim_raw or clim_raw; windowed climatologies are "
                    "clim_raw with window_start_years"
                )
        return out

    def forecast_evaluations(self) -> list[dict]:
        variants = self.cv.get("extras.forecast_variants") or _OLD_FORECAST_VARIANTS
        for variant, suffix in variants.items():
            if _OLD_FORECAST_VARIANTS.get(variant) != suffix:
                self.problems.append(
                    f"cv_models extras.forecast_variants: {variant!r} -> {suffix!r}; only "
                    + ", ".join(f"{k} -> {v!r}" for k, v in _OLD_FORECAST_VARIANTS.items())
                    + " exist"
                )
        out = []
        for index, entry in enumerate(self.cv.get("extras.forecasts", []) or []):
            variant = entry.get("variant") or "base"
            if variant not in variants:
                self.problems.append(f"cv_models extras.forecasts[{index}]: variant {variant!r} is not in forecast_variants")
            if "fair_brier" in entry:
                self.notes.append(f"cv_models: extras.forecasts[{index}].fair_brier ({UNUSED})")
            out.append({
                "model": entry.get("name"),
                "variant": variant,
                "raw": bool(entry.get("raw", False)),
                "calibrated": bool(entry.get("calibrated", False)),
                "platt_weights": bool(entry.get("export_platt_weights", False)),
            })
        return out

    def mme(self) -> dict:
        cv = self.cv
        cv.drop("mme.mc_cores", UNUSED)
        sources = []
        for index, model in enumerate(cv.get("mme.blend_models", []) or []):
            if model.get("source") not in ("clim", "forecast"):
                self.notes.append(f"cv_models: mme.blend_models[{index}] (source {model.get('source')!r}; the old code skipped it)")
                continue
            sources.append({"name": model.get("name"), "source": model.get("source"),
                            "cal_variant": model.get("cal_variant", "base")})
        return _without_none({
            "enabled": bool(cv.get("mme.enabled", False)),
            "variants": list(cv.get("mme.variants", ["fixed_cutoff"])),
            "sources": sources,
            "optimize": "auto" if cv.get("mme.optimize_if_holdout_is_full_2000_2024", True) else "never",
            "weights_year_tag": cv.get("mme.weights_year_tag"),
        })

    # -- assembly ------------------------------------------------------------

    def convert(self) -> Conversion:
        for spec in self.raw_specs:
            for key in ("input.parallel", "input.workers"):
                spec.drop(key, RUNTIME)
            for key in ("output", "input.wide_prefix", "id"):
                spec.drop(key, DERIVED)
            expected = "ground_truth_rainfall" if spec is self.gt else "rainfall_forecast"
            if spec.get("type", expected) != expected:
                self.problems.append(f"{spec.label}: type is {spec.get('type')!r}, expected {expected!r}")
        for key in ("input.ground_truth_wide_rds", "input.climatology_rds", "output"):
            self.combine.drop(key, DERIVED)
        for key in ("input_rds", "output_rds"):
            self.connect.drop(key, DERIVED)
        if self.connect.get("onset_spec"):
            self.notes.append(
                f"connect: onset_spec {self.connect.get('onset_spec')!r} (rain windows now resolve "
                "against the one onset definition)"
            )
        self.connect.drop("climatology.output_prefix", "the climatology features are always prob_clim")

        data: dict[str, Any] = {"name": self.name}
        if self.data_root is not None:
            data["data_root"] = self.data_root
        connect_models = list(self.connect.get("forecast_models", []) or [])
        if not connect_models:
            self.problems.append("connect: forecast_models is required")
        data["onset"] = self.onset(self.uses_ref_series())
        data["domain"] = self.domain()
        data["observations"] = self.observations()
        data["models"] = self.models(connect_models, self.definition)
        data["climatology"], windows = self.climatology(list(self.connect.get("climatology.window_tags", []) or []))
        data["features"] = self.features()
        data["evaluation"] = self.evaluation(windows)

        for spec in [self.regrid, *self.raw_specs, self.combine, self.connect, self.cv]:
            if spec is not None:
                self.notes += [f"{spec.label}: {path} ({UNUSED})" for path in spec.unread()]
        if self.problems:
            raise ConfigError(self.problems)
        try:
            config = BlendConfig.from_dict(data)
        except ConfigError as exc:
            raise ConfigError([f"the converted config: {p}" for p in exc.problems]) from None
        return Conversion(config=config, notes=tuple(self.sorted_notes()))

    def sorted_notes(self) -> list[str]:
        """Notes grouped by spec, in pipeline order."""
        order = ["regrid", "ground_truth", *(s.label for s in self.forecasts.values()), "combine", "connect", "cv_models"]
        rank = {label: i for i, label in enumerate(order)}
        return sorted(self.notes, key=lambda note: rank.get(note.split(":", 1)[0], len(order)))

    def uses_ref_series(self) -> bool:
        if self.cv.data.get("run", {}).get("cutoff_mode", "ref_onset") == "ref_onset":
            return True
        extras = self.cv.data.get("extras") or {}
        if any((f or {}).get("variant") == "ref_onset" for f in extras.get("forecasts") or []):
            return True
        if any((m or {}).get("cal_variant") == "ref_onset" for m in (self.cv.data.get("mme") or {}).get("blend_models") or []):
            return True
        texts = [str((f or {}).get("text", "")) for f in ((self.cv.data.get("models") or {}).get("formulas") or {}).values()]
        return any("_p_onset_ref_" in text for text in texts)


def _definition(resolved: dict) -> cfg.OnsetDefinition:
    spell = resolved["dry_spell"]
    return cfg.OnsetDefinition(
        window=resolved["window"],
        follow_days=resolved["follow_days"],
        dry_spell=cfg.DrySpell(min_dry_days=spell["min_dry_days"], sum_window=spell["sum_window"]),
    )


def _lower(value):
    return None if value is None else str(value).strip().lower()


def _rain_transform(value):
    if isinstance(value, dict) and "power" in value:
        return f"power:{value['power']}"
    return value


def convert_specs(
    *,
    name: str,
    ground_truth: dict,
    forecasts: dict[str, dict],
    combine: dict,
    connect: dict,
    cv_models: dict,
    regrid: dict | None = None,
    connect_onset_spec: dict | None = None,
    data_root: str | None = None,
) -> Conversion:
    """One BlendConfig from a set of stage specs (as loaded from YAML).

    `forecasts` maps each model name (the combine and connect key) to its
    raw_data spec. `connect_onset_spec` is the raw_data spec connect's
    `onset_spec` names, when that is not the ground-truth spec. The specs'
    paths are relative to the directory the old scripts ran from: give it as
    `data_root`.
    """
    return _Converter(
        name=name, ground_truth=ground_truth, forecasts=forecasts, combine=combine,
        connect=connect, cv_models=cv_models, regrid=regrid, connect_onset_spec=connect_onset_spec,
        data_root=data_root,
    ).convert()


def _load(path: str | Path | None) -> dict | None:
    if path is None:
        return None
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def convert_spec_files(
    *,
    name: str,
    ground_truth: str | Path,
    forecasts: dict[str, str | Path],
    combine: str | Path,
    connect: str | Path,
    cv_models: str | Path,
    regrid: str | Path | None = None,
    data_root: str | None = None,
) -> Conversion:
    """`convert_specs` on YAML files; connect's onset_spec is looked up beside
    the ground-truth spec."""
    connect_spec = _load(connect)
    onset_spec = None
    onset_id = (connect_spec or {}).get("onset_spec")
    if onset_id and onset_id != Path(ground_truth).stem:
        candidate = Path(ground_truth).parent / f"{onset_id}.yml"
        onset_spec = _load(candidate) if candidate.exists() else None
    return convert_specs(
        name=name,
        ground_truth=_load(ground_truth),
        forecasts={model: _load(path) for model, path in forecasts.items()},
        combine=_load(combine),
        connect=connect_spec,
        cv_models=_load(cv_models),
        regrid=_load(regrid),
        connect_onset_spec=onset_spec,
        data_root=data_root,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--ground-truth", required=True, help="raw_data spec of the observations and climatologies")
    parser.add_argument("--forecast", action="append", required=True, metavar="MODEL=SPEC",
                        help="a model's raw_data spec (repeat per model)")
    parser.add_argument("--combine", required=True)
    parser.add_argument("--connect", required=True)
    parser.add_argument("--cv-models", required=True)
    parser.add_argument("--regrid")
    parser.add_argument("--data-root", help="where the specs' relative paths start, relative to the output file")
    parser.add_argument("-o", "--output", help="write the config here (default: stdout)")
    args = parser.parse_args(argv)

    forecasts = {}
    for item in args.forecast:
        model, sep, path = item.partition("=")
        if not sep:
            parser.error(f"--forecast expects MODEL=SPEC, got {item!r}")
        forecasts[model] = path
    sources = {"ground_truth": args.ground_truth, **{f"forecast[{m}]": p for m, p in forecasts.items()},
               "combine": args.combine, "connect": args.connect, "cv_models": args.cv_models}
    if args.regrid:
        sources["regrid"] = args.regrid
    try:
        conversion = convert_spec_files(
            name=args.name, ground_truth=args.ground_truth, forecasts=forecasts, combine=args.combine,
            connect=args.connect, cv_models=args.cv_models, regrid=args.regrid,
            data_root=args.data_root,
        )
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    text = conversion.to_yaml(sources)
    if args.output:
        Path(args.output).write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
