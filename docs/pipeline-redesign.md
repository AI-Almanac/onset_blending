# Redesign: onset_blending owns the blend pipeline, in memory

Status: phase 1 (config) done; phase 2 next.

## Goals

- **One pipeline, in one process.** onset_blending runs the whole blend:
  inputs → onsets → climatology → features → evaluation → final fit, and
  applying a trained blend to a new season. Stages hand each other Python
  objects (xarray, numpy, small DataFrames); nothing is written between them.
- **One authoritative config.** A single `BlendConfig` replaces the six spec
  stages (regrid, raw_data ground truth and forecasts, combine, connect,
  cv_models). Anything derivable (column prefixes, artifact names, cutoff and
  year tags, stage-to-stage paths) is computed, never configured. It is
  validated once, up front.
- **Efficient data model.** Cells are integer positions into one index;
  dates are `datetime64`; members, issue dates, lead days are array axes.
  Strings and Python objects appear only at the edges.
- **Intermediates are optional.** Any stage's result can be saved (for
  debugging, consistency checks, extra diagnostics); by default only the
  outputs consumers read are written.
- **No lost functionality.** Every option the specs and scripts offer today
  keeps a home in the config (see "Functionality retained").

Decisions already made:
- Verification is *scientific equivalence* against the cached references
  (criteria below), not bit-identity.
- The script/spec interface (`1_blend_evaluation.py`, `3_fit_final_model.py`,
  `apply_blend_model.py`, the `run_*`/pipeline scripts, `specs/*.yml`) is
  replaced, not wrapped.
- ai-almanac moves its **local** path to the new API. Modal keeps the
  current workflow on the pinned onset_blending ref until a later change.

## What consumers need (the output contract)

From a trace of every reader in ai-almanac:

| Output | Read by | Must keep |
|---|---|---|
| `summary_models_pooled_<tag>.csv` | results page, chat tool, manifest summary | columns `model, auc, brier, rps, brier_skill, rps_skill, pietra, n, auc_week1..N, auc_later, brier_week1..N, brier_later`; models `unc_clim_raw`, `clim_raw`, `blended_model` |
| `summary_models_<tag>.csv` | skill map, chat tool | per-cell rows `id, model, lat, lon, n, rps, brier` plus an `ALL` row |
| `yearly_metrics_global_<tag>.csv` | "Scores by held-out year" | `year, model, brier, rps, auc` |
| final coefficient bundle | live-season scoring | now an onset_blending object; ai-almanac only passes it back to `apply` |
| day-level forest | live scoring (onset-forest) | unchanged (onset-forest owns it) |
| live forecast CSVs | forecast map | weekly `id, time, onset_threshold, lat, lon, cv_week1..N, cv_later`; daily `p_day_1..28, p_later` |

Everything else is download-only or unread: `combined_wide.pkl` (only the
day-level blend's input, which can take the in-memory table instead),
`weekly_training_input.pkl`, per-year coefficient pickles,
`formula_sample_support_*.json`, the `.pkl` twins of the CSVs,
`training_spec.yml` (becomes the saved `BlendConfig`).

## Config: one source of truth

Today the same facts are restated across stages: the onset definition in
both raw_data specs; the dissemination-units file 3 times; the unit-id column
5 times; year windows in 5 places; `n_bins` in connect and cv_models; model
names and column prefixes that must agree across combine, connect and
cv_models; and stage outputs re-typed as the next stage's inputs.

`BlendConfig` is organised by concept, not by stage:

```yaml
domain:            # where: grid or polygon units (regrid weights), unit ids,
                   # dissemination units, centroids, area of interest
observations:      # files, variable, dims, missing-rain policy, year range
models:            # each forecast model once: name, files, variable, dims,
  - name: aifs     # units, horizon (rain_day_max + policy), members
onset:             # the definition, once: window, wet-day rule, follow-up,
                   # dry spell, thresholds (scalar or per-unit table),
                   # season cutoff, reference onset (fixed or table), lead days
climatology:       # KDE settings, train/test windows (default: derived from years)
features:          # weeks (n_bins, days per bin, day_max), rain aggregates
evaluation:
  years:           # training, cv_holdout, true_holdout: stated once
  cv:              # methods: global | local | neighbors | clusters (+ params)
  formulas:        # named, with templating over models ({model}, {week})
  baselines:       # climatology logits (windows), raw + calibrated forecasts
  mme:             # optional optimised weights
  external:        # external CV predictions
outputs:           # what to write; `intermediates: [...]` for debugging
```

Derived, never configured: every column name and prefix (from model names
and weeks), artifact names and tags (from the cutoff mode and years), the
climatology windows (from the years unless overridden), what each stage
reads from the one before.

## Data model

| Stage | Today | Proposed |
|---|---|---|
| cells | `"lat_lon"` / district strings on every row | `Cells` index once; integer positions elsewhere |
| observation onsets | long table of every cell-day, then a wide `(id, year)` table | `xr.Dataset[cell, year]`: onset day / date |
| forecast onsets | 19.5M-row wide table per India file, then member aggregation | `xr.Dataset[issue, cell, lead]` per model (member stats on read) |
| climatology | pickled frames per chunk | `xr.Dataset[issue, cell, lead]` |
| combine + connect | 12 GB daily table → 1.2 GB weekly table, via pickles and row chunks | feature arrays `[row, feature]` + outcome codes; one slim DataFrame only for formula evaluation |
| evaluation | spawned scripts reading pickles and YAML | functions returning a `BlendResult` (CV predictions, coefficients, metrics tables, final model) |

## API (sketch)

```python
from onset_blending import BlendConfig, run_blend, apply_blend

cfg = BlendConfig.from_yaml("blend.yml")         # or built in code
result = run_blend(cfg)                          # in memory, stage by stage:
#   inputs = load_inputs(cfg); onsets = compute_onsets(inputs, cfg)
#   clim = build_climatology(onsets.obs, cfg); feats = build_features(...)
#   result = evaluate(feats, cfg)
result.save("out/", intermediates=["onsets", "features"])   # optional extras
season = apply_blend(result.final_model, cfg, live_forecasts)
```

ai-almanac's local runner builds a `BlendConfig` from the job, calls
`run_blend`, and publishes `result.save(...)`. Its intermediates cache becomes
a thin wrapper over the stage functions.

## Functionality retained

CV methods global, local, clusters (`neighbors` raises NotImplementedError
today and stays out); formula window variants;
climatology logits (unconditional, windowed); raw and Platt-calibrated
forecast evaluation, fair Brier, Platt-weight export; MME optimisation (and
reading saved weights); external CV predictions; regrid to polygon units
(weights, clipping, centroids), dissemination filtering, areas of interest;
regrid onto the observation grid's cells when there is no shapefile;
dissemination units that restrict the inputs or only training and scoring;
per-unit thresholds and reference-onset tables; missing-rain policies;
rain-horizon policies; both cutoff modes; final fit and application to
future years; country templates (ported to the new config).

## Migrating spec sets

`onset_blending.legacy_specs` converts a spec set (regrid, ground truth with
its climatology runs, one raw_data spec per model, combine, connect,
cv_models) into a `BlendConfig`:

- every fact the specs restate must agree, or the conversion fails, listing
  all conflicts at once;
- old defaults that differ from the new ones are written out (for example
  `trim_forecasts_after_true_onset: false`, `cutoff_mode: ref_onset`, no
  forecast scoring without `extras.forecasts`);
- dropped keys are listed in the output header with the reason (derived,
  a run setting, or read by no code).

The templates are ported to `configs/`; `tests/test_legacy_specs.py` keeps
each port equal to its spec set until the specs are removed. The key
inventories the converter follows are `docs/explore1.md` (cv_models and the
prediction scripts), `docs/explore2.md` (combine, connect) and
`docs/explore3.md` (regrid, raw_data, climatology).

Defects in today's code that the inventory found, and what the new pipeline
does about each:

| Today | New |
|---|---|
| MME with two or more sources raises KeyError 'lat' at the merge | fixed in phase 2 |
| formula `window_variants` never match, so a variant repeats its base formula | the variant swaps the windowed climatology in (the converter notes it) |
| two `diff` predictors on one model write one column | rejected by validation |
| combine `join` treats anything but `full` as inner | `inner` or `full` only |
| no threshold configured gives no onsets, silently | the threshold is required |
| unquoted `window_tags: [2000_2024]` reads as 20002024 | windows are year pairs; the converter explains the old value |
| `run.MR`, `include_baseline_1900`, `fair_brier`, `mc_cores`, `plus_label_template`, `default_prob_col`, combine `climatology.*`, `sources[].prob_col`, connect `mode`, `region_label` are read by nothing | dropped |
| `3_produce_figures.py` reads file names nothing writes any more | removed with the scripts in phase 4 |

The 2026 sets (`configs/fixed_cutoff_*2026*.yml`) are operational: their
forecasts cover 2026 only, so they describe applying a blend trained
elsewhere rather than training one.

## Verification: scientific equivalence

Against the cached references (Ethiopia and India, both bit-identical across
two machines), on every stage change:
- onset days: identical (integers);
- CV probabilities: max |Δ| ≤ 1e-6; coefficients: relative Δ ≤ 1e-6;
- pooled and yearly skill (Brier, RPS, AUC, Pietra): |Δ| ≤ 1e-6;
- per-cell metrics: |Δ| ≤ 1e-6 for at least 99.9% of cells, with any
  exceptions listed;
- time and peak memory recorded for each run.
A change that misses a criterion is either fixed or brought to review with
the differences shown.

## Phases

1. **Config.** `BlendConfig` with validation; a converter from today's spec
   sets so existing templates and tests carry over; port the templates.
2. **In-memory pipeline on today's functions.** Stage functions calling the
   existing logic, no files or subprocesses between stages; ai-almanac's local
   runner switches to it. Results should stay bit-identical at this phase,
   which isolates plumbing from numerics.
3. **Data model, stage by stage.** Observation onsets, forecast onsets,
   climatology, features, then evaluation move to the array model; each step
   verified and measured.
4. **Remove the old interface.** Scripts, specs and their helpers go; docs and
   tests updated.

Each phase ends with commits in both repos and a verification report.
