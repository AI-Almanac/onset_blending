# ==============================================================================
# File: read_spec.py
# ==============================================================================
# Purpose
#   Read YAML "spec" files and provide lightweight validation utilities.
#
# Function summary
#   get_arg(key, default=None)
#     Retrieve a CLI argument of the form --key value from sys.argv.
#
#   load_spec(spec_id, type)
#     Load YAML from specs/<type>/<spec_id>.yml.
#
#   validate_spec(spec, required_top=(), required_paths=(), checks=(), label="spec")
#     Generic validator:
#       * required_top: top-level keys that must exist
#       * required_paths: dotted paths (e.g. "input.nc_folder") that must exist
#       * checks: list of callables(spec) run after structural checks
#     Returns spec unchanged on success; raises ValueError on failure.
#
#   outcome_levels_from_spec(spec)
#     Build ordered outcome labels: day_1..day_max plus spec["targets"]["plus_label"].
#
#   clim_cols_from_spec(spec)
#     Construct climatology column names from spec.
# ==============================================================================

import sys
import os
import yaml
from ._shared_helpers import _has_path, _get_path


def get_arg(key, default=None):
    """
    Retrieve a CLI argument of the form --key value from sys.argv.
    Also supports key=value positional form for backward compat with R style.
    """
    args = sys.argv[1:]
    # --key value form
    for i, a in enumerate(args):
        if a == f"--{key}" and i + 1 < len(args):
            return args[i + 1]
        # key=value form
        if a.startswith(f"{key}="):
            return a[len(key) + 1:]
    return default


def load_spec(spec_id, spec_type):
    """Load YAML from specs/<spec_type>/<spec_id>.yml."""
    path = os.path.join("specs", spec_type, f"{spec_id}.yml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Spec file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def validate_spec(spec, required_top=(), required_paths=(), checks=(), label="spec"):
    """
    Validate a spec dict.

    Parameters
    ----------
    spec : dict
    required_top : iterable of str
        Top-level keys that must be present.
    required_paths : iterable of str
        Dotted paths (e.g. "input.nc_folder") that must be present.
    checks : iterable of callables
        Each callable receives spec and should raise on failure.
    label : str
        Label used in error messages.

    Returns
    -------
    spec : dict  (unchanged)
    """
    # --- 1) top-level fields ---
    missing_top = [k for k in required_top if k not in spec]
    if missing_top:
        raise ValueError(f"{label} missing top-level fields: {', '.join(missing_top)}")

    # --- 2) nested dotted paths ---
    missing_paths = [p for p in required_paths if not _has_path(spec, p)]
    if missing_paths:
        raise ValueError(f"{label} missing required fields: {', '.join(missing_paths)}")

    # --- 3) extra checks ---
    for fn in checks:
        fn(spec)

    return spec


def outcome_levels_from_spec(spec):
    """Build ordered outcome labels: day_1..day_max plus plus_label."""
    max_day = spec["targets"]["max_day"]
    plus_label = spec["targets"]["plus_label"]
    return [f"day_{k}" for k in range(1, max_day + 1)] + [plus_label]


def clim_cols_from_spec(spec):
    """
    Construct climatology column names from spec.
    Important: 'plus' must be last (downstream code depends on this ordering).
    """
    max_day = spec["targets"]["max_day"]
    pref = spec["climatology"]["prefix"]
    plus = spec["climatology"]["plus_col"]
    cols = [f"{pref}{k}" for k in range(1, max_day + 1)]
    return cols + [plus]
