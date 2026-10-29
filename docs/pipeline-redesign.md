# Redesign: onset_blending owns the blend pipeline, in memory

Status: phase 1 (config) done. Phases 2 and 3: the package side is done;
ai-almanac's side (the live writer, switching its runners) is next.

Phase 2 verification, against the cached references:
- Ethiopia: `train()` reproduces every artifact exactly (combined and weekly
  tables, CV predictions, fold and final coefficients, per-cell, pooled and
  yearly metrics) in 44 s, against 525 s for the original code. A forecast from
  the saved blend, replaying 2025 as a live season, gives the old apply path's
  probabilities exactly.
- India: the combined and weekly tables are identical (inputs in 242 s, peak
  66 GB, against 48 GB when the stages hand over files: phase 3 targets this).
  The evaluation and final fit, run on the reference weekly table, give all 36
  artifacts byte-identical.
- The old script-based run on India took 7,462 s (peak 94 GB) and matches the
  branch's run on 38 of 39 files; the 39th differs only in log text.

Phase 3 (data model), against the same references:
- Every stage is arrays: onsets and climatology as xarray Datasets, the
  weekly features as `features.Weekly` (feature arrays over samples,
  outcome codes, ids and coordinates per unit; `frame()` rebuilds the
  legacy table exactly). Design matrices come from the arrays
  (`design.Plan`, Patsy's columns and bits, 6x faster than Patsy); global
  CV and the final fit share one design in shared memory.
- The evaluation is arrays too, driven by the config: every model's CV
  predictions are `Predictions` (an xarray Dataset over samples, with the
  samples' keys), the baselines, Platt calibration, MME and scores are
  computed from the arrays (`core.calibration`, `blend.mme`,
  `core.metrics`), and local and cluster CV fit from the arrays (a cluster
  once per year rather than once per unit). Artifacts are NetCDF (CV
  predictions, fold coefficients) and CSV (scores, weights); no pickles.
  The legacy script, run on `_specs.cv_spec(config)`, stays the oracle in
  the tests: every artifact has the same content.
- Ethiopia: 21 s; India: 197 s (evaluation and final fit 138 s; the system
  peak was +13.8 GB before the evaluation moved to arrays, against 469 s and
  66 GB in phase 2). Both match the
  references on every artifact; the one exception is the Platt-calibrated
  baselines, whose fits now run on one BLAS thread so results do not
  depend on the core count: their Brier and RPS move by < 1e-13, their AUC
  by < 3e-4 (near-ties reorder).

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
| combine + connect | 12 GB daily table → 1.2 GB weekly table, via pickles and row chunks | `features.Weekly`: feature arrays over `sample`, outcome codes, ids/lat/lon per unit (done) |
| evaluation | spawned scripts reading pickles and YAML | `evaluate(weekly, config)` → `Evaluation`: every model's CV predictions as `Predictions` (xarray over samples), fold models, metrics from the arrays, final model; design matrices built from the arrays, one shared per formula; NetCDF and CSV artifacts (done) |

## Package layout

onset_blending is an installable package (`pyproject.toml`, `pip install -e .`).
Nothing in it edits `sys.path`, changes directory, or finds files relative to
the repository: modules import each other as `onset_blending.<module>`, and
every path comes from the config or the caller. Relative paths in a config
file resolve against that file's directory, or against its `data_root` if it
sets one.

```
onset_blending/
  config.py          BlendConfig: the one recipe
  core/              the stages every blend shares, on arrays and xarray
    io.py            files, each file's format, reading rainfall
    domain.py        units, ids, regridding, areas of interest
    reference.py     thresholds, reference onsets, weights (resolved once)
    onsets.py        observed and forecast onsets
    climatology.py   KDE onset climatology
    features.py      the weekly samples (Weekly)
    design.py        formulas, and their design matrices from the arrays
    predictions.py   a model's probabilities for samples (Predictions, NetCDF)
    metrics.py       Brier, RPS, AUC, Pietra per unit, pooled and per year
    calibration.py   Platt calibration (cross-validated, weight export)
    parallel.py      worker pools, shared memory
  blend/             blending methods behind one interface
    logit.py         the multinomial logistic blend; global, local, cluster CV + final fit
    random_forest.py the CRPS random forest over daily rain and the climatology
    _crps_tree.py    its trees, from onset-forest (same numerics, stored as arrays)
    mme.py           the multi-model ensemble of baselines (RPS-optimised weights)
  evaluation.py      cross-validation, baselines, calibration, scores
  training.py        train(config) -> TrainingResult, holding a TrainedBlend
  trained.py         TrainedBlend: recipe + fitted model + reference data
  forecast.py        TrainedBlend.forecast(...) -> Forecast: the operational path
  legacy/            spec-set converter (convert-specs)
archive/             the retired scripts, spec sets and their reference data
```

Each stage is a function from inputs to a result object, with no files, globals
or working-directory state between stages. Extension points are plain tables
in one place: a new rain-predictor aggregate, CV method or output writer is
one function registered next to the others, not a new script.

`predict/` becomes `forecast.py` and `maps.py`: no separate scripts, specs or
temporary `_op.yml` copies.

## Training and operational forecasting

```python
from onset_blending import BlendConfig, TrainedBlend, train

result = train(BlendConfig.from_yaml("configs/india_aifs.yml"))
result.save("out/")                       # evaluation tables (the output contract)
result.blend.save("india_aifs.blend")     # what operations needs, in one file

blend = TrainedBlend.load("india_aifs.blend")
# One file per raw model, each holding one initialisation: the issue date.
forecast = blend.forecast({"aifs_ensv2": "live/ens_2026-06-09.nc", "aifsv2": "live/det_2026-06-09.nc"})
forecast.save("out/2026-06-09/")          # weekly + daily CSVs, maps if asked
```

**The operational path cannot take blending parameters.** `forecast()` accepts
where the live forecast files are (and optionally an issue date), nothing
else. Operationally each raw model gives one file with one initialisation,
which is the issue date forecast; every model must be initialised on the
same date, inside the season. Files holding a season of issues (a replay)
forecast each of them, or the one `issue_date` names. Every
parameter comes from the `TrainedBlend`, which is written by training and
holds, in one file:

- the `BlendConfig` it was trained with (the recipe);
- the fitted model: formula, feature names, scaler, coefficients, classes;
- reference data, resolved at training time so operations cannot resolve it
  differently: the units, per-unit thresholds, the observed onset days that
  the climatology is fitted on, the regrid weights per model grid, and the
  dissemination units;
- each model's forecast signature (below);
- provenance: package version, `FEATURES_VERSION`, config digest, training
  years, and fingerprints of the training inputs.

Today a coefficient pickle travels separately from the five specs that
produced its features, and nothing checks that the specs used for operations
are the ones used for training. The onset definition, thresholds,
climatology years, rain windows and horizons, bins and cutoff mode can all
drift silently; only the feature names are checked. Keeping the recipe and
reference data inside the trained blend removes that possibility rather than
detecting it.

**A forecast must cover what the blend was trained on.** Files need not share
a layout: each file is read in its own format (`formats.py`, the rules
ai-almanac registers datasets by). A dimension's role comes from the model's
`dims`, else its name (aliases such as `prediction_timedelta_daily`,
`member`, `latitude`), else a lead coordinate counted in days; rainfall is
converted to millimetres from the variable's declared `units` (`m` ×1000,
`mm`, `kg m-2`), the model's `unit_scale` applying only to files that
declare none. Training records each model's forecast signature, what the
blend reads in that resolved form:

- the variable and the units its rainfall is read in (mm, or undeclared);
- how its daily values were aggregated: the cell method over the lead
  dimension (`cell_methods`, e.g. sum) and the window the lead coordinate
  states (e.g. "End of the UTC calendar-day aggregation window (00:00 UTC
  of the following date)");
- the ensemble member ids;
- the lead days;
- the grid: its lat/lon values as sets, or unit ids for files already on units;
- other dimensions' sizes and the issue time of day (00Z or 12Z shifts every
  lead day).

Every training file must give the same signature, apart from the issue dates.
Otherwise training refuses and names the files that differ: a blend fitted on
a mix of ensemble sizes has no single ensemble to forecast with.

A live file may hold more than the blend was trained on and is trimmed to
it: longer leads, more members (only the trained member ids are read), a
larger domain such as the globe (only the trained units are read). It is
refused when it holds less, or something else:
- lead days, members or grid points the training had are missing;
- an ensemble where training was deterministic, or the reverse;
- the grid has another resolution (its spacing differs), even where its
  points include the trained ones;
- rainfall in other units after conversion (e.g. declared against
  undeclared), another aggregation method or window, another issue hour,
  another variable or other dimension sizes.
A file that declares no aggregation is accepted only against training files
that declared none.

The other checks, made before any work with every problem reported at once:
- exactly the trained models are supplied;
- the issue date is inside the season;
- a per-year reference-onset table has the live year.

**Code version.** `FEATURES_VERSION` is bumped whenever feature numerics
change. A blend trained under another version is refused: retrain, or run the
version it was trained with. The same parameters on changed code is the same
failure in another form.

Atomicity:
- a trained blend is saved and loaded as a whole, written to a temporary
  file and renamed, and verified against its digest on load;
- an operational run writes its whole output set or nothing;
- no pickles in the artifact (JSON, NumPy and NetCDF only), so it loads
  across library versions and is safe to read.

ai-almanac stores the one `.blend` file per trained blend and calls
`forecast()` with one single-initialisation file per model. Registered
archives keep their own layout: no copies are written, and `model_layouts`
is not needed by this path (each file says its format). India's and
Ethiopia's AIFS, Aurora and GraphCast archives use
`prediction_timedelta_daily` and metres, declaring `cell_methods:
prediction_timedelta: sum` and the UTC-day window on the lead coordinate;
the live writer writes `day` and mm.

The live writer must still produce what the archives hold, which the
signature check now enforces. A survey of ai-almanac's writer found two
defects, inferred from the code and the files' attributes:
- **Lead days.** Live day d sums the 6-hourly steps at lead hours
  24d … 24d+18, i.e. hours (24d−6, 24d+18], so it overlaps archive day d+1
  by 18 h. Live forecasts are scored about a day late against blends trained
  on the archives. Archive day k is the UTC day ending at issue + k days,
  hours (24(k−1), 24k] for a 00Z issue.
- **Members.** Live ensemble runs average their 8 members into one trajectory
  before writing, while the archives keep 26 members. The ensemble onset
  probabilities (the share of members with an onset) then come from a single
  mean trajectory.
Decision: live ensemble runs keep all 26 members separate, and the live
writer uses the archive lead-day convention and declares it (the archive's
`cell_methods` and lead-window description) with `units: mm`. Its live path
today also re-supplies parameters from the job record, and the
weekly-feature settings fall back to code defaults (day_max, days_per_week,
n_weeks, rain_window). Those match training only because training never
overrides them.

**The random forest blend.** `evaluation.random_forest` adds the CRPS random
forest (`blend.random_forest`, from onset-forest) beside the logistic blend.
It reads, per weekly sample, the unit's lat/lon, each model's ensemble-mean
rainfall for every lead day and the conditional climatology for lead days
1..max_day (`features.daily_inputs`, from the stage arrays: no day-level
table), learns the onset's lead day, and predicts daily probabilities whose
week sums are scored with the other models. Its folds are the true holdouts,
then blocks of CV years; each trains on every sample of the dissemination
cells in its training years and predicts the samples the formulas score.
The final forest is stored in the trained blend as arrays, and a forecast
then also gives daily probabilities. On Ethiopia, with the same parameters,
its feature columns, medians and daily and weekly probabilities equal
onset-forest's on the workflow's forest frame exactly.

**Dates of observations and forecasts.** Forecast lead day d is the UTC day
ending at issue + d (the archives declare it; the reforecast inference sums the
native accumulations over (24(d−1), 24d] hours), and the blend scores it
against an observed onset on issue + d. Observation files do not say which
window their date covers, so each observation dataset declares it
(`observations.date_marks`): `end` when the date ends its window (IMD: the
24 h to 08:30 IST), `start` when the date is the day the rain fell
(calendar-day products such as CHIRPS), which moves its dates a day later to
the forecasts' labels; each day keeps the season of the day it covers.

A lag check settled the two registered datasets: forecast lead day d
correlates best with IMD's date issue + d (0.69, against 0.61 a day earlier)
and with CHIRPS's issue + d − 1 (0.70, against 0.59). India's blend was aligned
already; Ethiopia's was scoring forecasts a day late. Aligned (`start`),
Ethiopia's pooled RPS skill against the unconditional climatology rises for the
blend (0.141 → 0.157) and for every forecast (AIFS-ENS raw −0.015 → +0.040,
calibrated −0.020 → +0.029; AIFS v2 raw −0.647 → −0.574), while the
observation-only climatology stays put (0.076 → 0.078).

Existing operational blends (a coefficient pickle plus a spec set) import once
with `TrainedBlend.from_legacy(coef_pkl, config)`. It builds the reference data
from the config's inputs and refuses if the coefficients' feature names are
not exactly what the config builds.

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
| MME with two or more sources raises KeyError 'lat' at the merge | sources aligned on their samples' id and issue date (phase 3) |
| formula `window_variants` never match, so a variant repeats its base formula | the variant swaps the windowed climatology in (phase 3; the converter notes it) |
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
2. **Package and in-memory pipeline on today's functions.** `pyproject.toml`
   and the module layout above; stage functions that call the existing logic,
   with no files or subprocesses between them; `train()`, `TrainedBlend` and
   `forecast()` replace `1_blend_evaluation.py`, `3_fit_final_model.py` and
   `predict/`. ai-almanac's local runner and live scoring switch to it.
   Results should stay bit-identical at this phase, which isolates the
   plumbing from the numerics. The operational path is checked against
   today's `run_operational_pipeline.py` on a replayed season.
3. **Data model, stage by stage.** Observation onsets, forecast onsets,
   climatology, features, then evaluation move to the array model; each step
   verified and measured.
4. **Remove the old interface.** Scripts, specs and their helpers go; docs and
   tests updated.

Each phase ends with commits in both repos and a verification report.
