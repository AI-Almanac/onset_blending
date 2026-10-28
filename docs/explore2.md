I've inventoried every combine and connect spec key the code reads. Seven template keys are never read: in combine `type`, `options.plus_label_template`, `options.default_prob_col`, `forecasts.<m>.sources[].prob_col` and the whole top-level `climatology:` block (`max_day`, `prob_col`); in connect, `mode`.

  There are no tests covering combine or connect. All paths below are relative to `/home/hcwf/code/onset_blending`. Spec loading:
  - **Combine:** `load_spec(spec_id,"combine")` reads `specs/combine/<id>.yml` relative to the current directory (`python/pipelines/prepare_data/3_combine_datasets.py:38`, `python/pipelines/_shared/read_spec.py:50-56`).
  - **Connect:** `yaml.safe_load` reads `specs/2025_blend/<spec_id>.yml` directly; the default `--spec_id connect_ref` does not exist (`python/pipelines/blending_process/0_connect_prepare_data_to_2025_pipeline.py:29,42-44`).
  - **Wrappers:** `python/run_training.py` and `predict/run_operational_pipeline.py` load both specs, rewrite keys, and write temporary `<id>_op.yml` copies. Their rewrites are listed under section 1C.

  ## 1. Spec keys read

  Abbreviations: CFU = `python/prepare_data/combine_forecasts_utils.py`, 3CD = `python/pipelines/prepare_data/3_combine_datasets.py`, CU = `python/blending_process/connect_utils.py`, RHU = `python/rain_horizon_utils.py`, RT = `python/run_training.py`, ROP = `predict/run_operational_pipeline.py`.

  ### A. Combine (`specs/combine/*.yml`)

  | Key | Type | Default if absent | Allowed | Read at | Meaning |
  |---|---|---|---|---|---|
  | `type` | str | – | – | **UNUSED** | – |
  | `input.ground_truth_wide_rds` | path | required (KeyError) | pkl with `id`, `year`; optional `onset_day`, `onset_date` | 3CD:41 → CFU:105-128 | Ground truth, left-joined on (id, year) |
  | `input.climatologies` | map name→{rds, out_prefix} | falsy → falls back to `climatology_rds` | – | 3CD:45-52 | Climatology tables to join |
  | `input.climatologies.<k>.rds` | path | required | wide pkl with `id,time,year,predicted_prob_day_<k>`, no `day` column | 3CD:49 → CFU:141-158 | Climatology file |
  | `input.climatologies.<k>.out_prefix` | str | `"<k>_p_onset"` (coalesce) | – | 3CD:50 | Renames `predicted_prob_day_` to `<out_prefix>_day_` |
  | `input.climatology_rds` | path | none | – | 3CD:53-58 | Legacy single climatology, only used if `climatologies` is falsy; prefix hardcoded `clim_p_onset` |
  | `options.join` | str | inner | only the literal `"full"` gives an outer join; anything else is inner | 3CD:99 | How climatology and forecast daily tables are joined |
  | `options.trim_forecasts_after_true_onset` | bool | False | – | 3CD:112-116 | Drops rows with time > true_onset_date |
  | `options.plus_label_template` | str | – | – | **UNUSED**: `format_forecast_family` takes the parameter (CFU:237), never uses it, and 3CD:63 does not pass it | – |
  | `options.default_prob_col` | str | – | – | **UNUSED** | – |
  | `forecasts` | map model→conf | `{}` | – | 3CD:62-65 | Forecast families; the key becomes the column prefix |
  | `forecasts.<m>.max_day` | int > 0 | required (KeyError) | – | CFU:243-245, checked at CFU:299-306 | Ceiling check only: errors if any `<col>_day_k` has k > max_day; nothing is truncated |
  | `forecasts.<m>.sources` | list | required (KeyError at CFU:264) | – | CFU:262-266 | Files concatenated together |
  | `forecasts.<m>.sources[].file` | path | required | – | CFU:189 | Forecast wide pkl (needs `id,time,year`) |
  | `forecasts.<m>.sources[].years` | str or int | `""` (no filter) | `"a:b"` inclusive, comma lists, mixed | CFU:193, parsed at CFU:32-51 | Year filter |
  | `forecasts.<m>.sources[].prob_col` | str | – | – | **UNUSED** (set in the 2026 `aifs` entries) | – |
  | `forecasts.<m>.constants` | list | `[]` | – | CFU:254-257 | Per (id, time, year) columns |
  | `forecasts.<m>.constants[].col` / `.out` | str | required | col must exist in the source (CFU:212-214) | CFU:255, renamed at CFU:282 | `col` becomes `<m>_<out>` |
  | `forecasts.<m>.daily` | list | required non-empty (CFU:247-248) | – | CFU:250-253 | Day-indexed variables |
  | `forecasts.<m>.daily[].col` / `.out` | str | required | source must have `^<col>_day_\d+$` (CFU:217-221) | CFU:251, renamed at CFU:309 | `<col>_day_k` becomes `<m>_<out>_day_k` |
  | `forecasts.<m>.daily[].add_plus` | bool | False | – | CFU:251, CFU:314-315 | Adds a `..._day_<N+1>plus` column |
  | `climatology.max_day`, `climatology.prob_col` | – | – | – | **UNUSED** (no code reads a top-level `climatology` block in combine) | – |
  | `output.out_dir` | path | required (KeyError) | – | 3CD:127 | Output directory |
  | `output.basename` | str | spec_id | – | 3CD:130-131 | Output is `<basename>_combined_wide.pkl` |

  ### B. Connect (`specs/2025_blend/connect_*.yml`)

  | Key | Type | Default | Allowed | Read at | Meaning |
  |---|---|---|---|---|---|
  | `mode` | str | – | – | **UNUSED** (no reader anywhere) | – |
  | `input_rds` | path | required | – | CU:293, 301-302 | Combined wide pkl |
  | `output_rds` | path | required | – | CU:294, 595-600; the script prints it at 0_connect:51 | Output pkl, written verbatim |
  | `day_max` | int | 28 | – | CU:295 | Probability days 1..day_max summed into bins; also sets the legacy rain horizon (+10) |
  | `days_per_bin` (alias `days_per_week`) | int | 7 | – | CU:297-298 | Bin width; also used for outcome binning |
  | `n_bins` (alias `n_weeks`) | int | 4 | – | CU:299 | Number of week bins (not counting earlier/later) |
  | `onset_spec` | str (raw_data spec id) | none | must exist as `specs/raw_data/<id>.yml` | CU:56-58 | Source for symbolic rain windows |
  | `options.window`, `options.onset_definition` (inline) | – | none | – | CU:60-62 | Inline onset definition, used only if there is no `onset_spec` |
  | `climatology.base_prefix` | str | `clim` | – | CU:354 | Reads `<bp>_p_onset_day_1..day_max` |
  | `climatology.unconditional_prefix` | str | `clim_unc` | – | CU:355 | Reads `<up>_p_onset_day_0..day_max` (day_0 required, CU:151-153) |
  | `climatology.output_prefix` | str | `prob_clim` | – | CU:356-358 | Prefix of the output logit columns |
  | `climatology.window_tags` | list | `[]` | – | CU:364-377 | Each tag reads `<bp>_<tag>_p_onset_day_*` |
  | `rain_transform` (top level) | see per-model | `identity` | same | CU:484 | Default rain-feature transform |
  | `forecast_models` | list | required (KeyError CU:337) | – | CU:337, 388-393, 405, 472 | Forecast models; the first one supplies the onset threshold |
  | `forecast_models[].name` | str | required | – | CU:337-338, 391, 406, 473 | Model column prefix |
  | `forecast_models[].variants` | list[str] | `[]` (the base series is always added) | suffix tokens such as `fixed_cutoff`, `ref` | CU:392, 407 | Reads `<name>_p_onset[_<v>]_day_*` |
  | `forecast_models[].rain_predictors` | list of `{agg, window}` or legacy strings | `[]` (skips the whole rain block, including horizon validation) | agg: `diff`, `max`, `min`; window: int, digit string, or token | CU:474-476, parsed at CU:515-525 | Rain features |
  | `forecast_models[].rain_transform` | str, number or `{power:p}` | spec-level, then identity | `identity`/`none`/`""`, `sqrt`, `fourth_root`, `log1p`, `"power:<p>"`, `{power:p}`, bare number | CU:482-485, built at CU:171-232 | Per-model override |
  | `forecast_models[].rain_day_max` | positive int | None, which means legacy N = day_max + 10 | – | CU:487 → RHU:31-36 | Rain horizon |
  | `forecast_models[].rain_horizon_policy` | str | `exact` if `rain_day_max` is set, else `legacy` | `legacy`, `exact`, `truncate` (lowercased) | RHU:37-54 | Validation mode. `legacy` with an explicit `rain_day_max` is an error, and so is a non-legacy policy without one |
  | `geometry.grid_id_decimal_digits` / `geometry.grid_id_format` (or the same keys at top level when there is no `geometry` block) | int / str | none | – | CU:306-308 → `python/prepare_data/spatial_id_utils.py:44-67, 313-339` | Only used if the combined table lacks both `id` and `adm3_name`. Combine always writes `id`, so this never happens in practice |

  Not a spec key but it controls behaviour: the CLI flag `--required_probability_prefixes` (0_connect:31-39 → CU:383-429).

  ### C. Keys the wrappers read or rewrite
  - **RT combine patches:**
    - `output.out_dir` is set to work_dir and `output.basename` to the combine spec id (RT:757-758).
    - Every `input.climatologies.<k>.rds` whose key `k` is also a run in the clim spec's `climatologies` is rewired to `<work_dir>/<out_stem or paths.climatology_out_stem+"_"+k, default stem "climatology_issue">.pkl` (RT:467-502). Keys `clim` and `clim_unc` must exist in both specs (RT:487-495).
    - `input.ground_truth_wide_rds` is chosen in this order: `--gt_path`, then `<work_dir>/<clim_spec>_wide.pkl`, then the existing work-dir artifact, then the spec's own value (RT:505-516, 763-770).
    - `forecasts.<MODEL>.sources[0].file` is set to `<work_dir>/<raw_spec_id>_wide.pkl`. Only `sources[0]` changes; `years` and other sources are untouched. Legacy role keys `aifs`/`aifs_ens` are renamed to the model name (RT:519-560, 771-777).
  - **RT connect patches:**
    - Every `--forecast` model needs a `forecast_models` entry, and names must be unique (RT:261-287).
    - `day_max` is read as `probability_day_max` (RT:707).
    - `rain_day_max` and `rain_horizon_policy` are reconciled with the raw spec's `options.*` and written into both specs (RT:290-320).
    - `input_rds` becomes `<work_dir>/<combine_spec>_combined_wide.pkl`. `output_rds` becomes `--blend_input` if given, else `<work_dir>/<basename(output_rds)>`; it must be non-empty (RT:455-464, 782-800).
    - Required prefixes come from the cv spec (RT:358-417).
  - **ROP:**
    - Same as RT, except climatology rds is only `clim`/`clim_unc`, from the mandatory `climatologies.<k>.out_stem` with no fallback (ROP:418-425).
    - `ground_truth_wide_rds` is rewritten only if `--gt_path` is given (ROP:433-434).
    - Required prefixes come from the saved coefficient features matching `<name>_p_onset(_x)?_week\d+` (ROP:188-223, 458).

  ## 2. Cross-stage dependencies (what must match)

  **Ground-truth / climatology raw spec → combine**
  - `input.ground_truth_wide_rds` must be the ground-truth raw output `<output.out_dir>/<output.basename|spec_id>_wide.pkl` (`python/prepare_data/nc_utils.py:2356,2407`). It must be the same file as the clim spec's `input.gt_path`; wrappers enforce this (RT:741-750, 763-770).
  - It must contain `onset_date`, because connect needs `true_onset_date` (CU:344-345).
  - `input.climatologies.<k>.rds` must equal `<paths.climatology_out_dir>/<climatologies.<k>.out_stem | paths.climatology_out_stem_<k>>.pkl` (`python/pipelines/prepare_data/2_build_climatology.py:133-140`).
    - Combine climatology keys must equal the raw `climatologies` keys for wrapper rewiring.
    - The 2026 combine specs reference `imd_clim_mok_date_clim_issue.pkl`, but `ref_rain_fixed_cutoff_2026` uses `out_stem: ref_rain_fixed_cutoff_clim_issue`. They only line up when a wrapper rewires them.
  - The climatology run feeding connect's `unconditional_prefix` must be `conditional: false`, because only that writes `predicted_prob_day_0` (`python/prepare_data/climatology_utils.py:567-573`) and connect requires day_0 (CU:151-153).
  - Each climatology run's `forecast_window` must be ≥ connect `day_max` (CU:135-138).
  - Climatology `test_year_min/max` and `season_start_md`/`issue_end_md` must cover the forecast (id, time, year) rows; otherwise the inner join (3CD:99-103) drops them.

  **Raw forecast spec → combine**
  - `sources[].file` must equal `<raw out_dir>/<basename|spec_id>_wide.pkl` (RT:736-738).
  - `forecasts.<m>.max_day` must be ≥ the raw spec's `options.max_day`. Raw output keeps days [min_day, max_day] (`python/prepare_data/nc_utils.py:880-911`), and a larger day raises at CFU:305-306.
  - `daily[].col` must be one of the raw output prefixes: `predicted_prob`, `predicted_prob_fixed_cutoff`, `predicted_prob_ref`, `forecast_rain`, `forecast_rain_sd`, `frac_raining` (`nc_utils.py:1683-1694, 1826-1836`).
  - `constants[].col` must be one of `onset_thresh`, `ref_onset_date` (`nc_utils.py:1683-1686, 1923`).

  **Combine → connect**
  - Connect `input_rds` must equal `<combine output.out_dir>/<output.basename|spec_id>_combined_wide.pkl` (3CD:127-131).
  - Combine `forecasts` keys must equal connect `forecast_models[].name`.
  - The rain daily `out` must be literally `rain_mean`: connect reads `<name>_rain_mean_day_` (CU:491).
  - Probability `out` names must be `p_onset` (base) and `p_onset_<v>` for every connect variant `v` (CU:388-393, 408-412). The base series is required unless `--required_probability_prefixes` excludes it.
  - The first connect `forecast_models` entry must have a combine constant with `out: onset_thresh` (CU:337-341). The 2026 `aifs` combine entries have no constants, so they only work while listed second in connect.
  - `out_prefix` values must equal `<base_prefix>_p_onset`, `<unconditional_prefix>_p_onset`, and `<base_prefix>_<tag>_p_onset` for each window tag. With the default `<k>_p_onset`, this means combine key `clim_<tag>` pairs with connect tag `<tag>`.
  - The forecast day horizon (raw `options.max_day`, and the combine `max_day` ceiling) must be ≥ connect `day_max` for probabilities and ≥ the rain horizon N. In legacy mode N = day_max + 10 = 38 (RHU:31-36).

  **Raw forecast spec ↔ connect (rain horizon)**
  - Raw `options.rain_day_max/rain_horizon_policy` and connect `forecast_models[].rain_day_max/rain_horizon_policy` must resolve to the same pair if both are set; RT copies whichever is set into the other (RT:290-320).
  - Strict mode on the raw side also needs `options.min_day: 1` and `options.max_day ≥ rain_day_max` (`nc_utils.py:75-93`).
  - Exact mode makes connect reject rain columns beyond N, so raw `max_day` must not exceed N.

  **`onset_spec`**
  - It names the ground-truth raw spec (for example `ref_rain_fixed_cutoff_2026`), not a forecast spec. Tokens resolve from its `options.window` and `options.onset_definition.*` (`python/prepare_data/onset_utils.py:374-389`).
  - Nothing checks that the forecast raw specs use the same onset definition.

  **Connect → cv_models**
  - `climatology.output_prefix` (prob_clim) must equal cv `extras.clim_logits[name=clim_raw].base_col_prefix`. `resolve_climatology_weeks` (`python/blending_process/blend_evaluation_utils.py:41-55`) finds `<prefix>_week<n>`, and `N_BINS = max(weeks)` in `1_blend_evaluation.py:155-156` and `predict/3_fit_final_model.py:279-280`.
    - `<output_prefix>_unc` must equal the `unc_clim_raw` `base_col_prefix`.
    - `<output_prefix>_unc_earlier` must equal `earlier_col`, with `earlier_is_logit: true`.
  - Connect `n_bins` should equal cv `n_bins`. cv `n_bins` is used only when `build_formulas_from_spec` gets no `n_bins` (blend_evaluation_utils.py:64-66, 287), for example in `predict/apply_blend_model.py:170`.
  - Model names must match cv `extras.forecasts[].name`, `mme.blend_models[].name` (source forecast) and the formula tokens.
  - Variants: a cv variant V maps through `extras.forecast_variants` (default `{base:"", ref_onset:"_ref", fixed_cutoff:"_fixed_cutoff"}`, blend_evaluation_utils.py:72-81) to a suffix. The connect variant must be that suffix without the leading underscore, so cv `ref_onset` needs connect `ref`. RT errors on undeclared prefixes (RT:411-416).
  - Rain predictor names must match formula terms `diff_<m>_qx` and `<agg>_<m>_<W>day_qx`, where W is the resolved integer.
  - Window tags must be the string `"<sy>_<ey>"`, matching cv `clim_logits.window_start_years`/`window_end_year` (`1_blend_evaluation.py:449-454` → blend_evaluation_utils.py:458-466) and `models.window_variants` (`start_years`, `end_year`, `replace.to: prob_clim{window}_qx`, blend_evaluation_utils.py:333-341). Start year 1900 means no tag.
  - `output_rds` vs cv `run.cutoff_mode` and `run.pipeline_input_dir`: without `--input_path`, the evaluator looks for:
    - `<work_dir>/cv_data_[<variant>_]new_pipeline.pkl`, or
    - `Monsoon_Data/Processed_Data/pipeline_input/<run.pipeline_input_dir>/cv_data_[<variant>_]new_pipeline.pkl` (blend_evaluation_utils.py:119-156).
    - `export_blend_output.py:247-254` defaults to the root `Monsoon_Data/Processed_Data/2025_pipeline_input`.
    - The 2026 connect outputs `cv_data_fixed_cutoff_new_pipeline_2026*.pkl` do not match the derived name, so they rely on wrappers passing `--input_path`.
  - Connect `mode` looks paired with cv `run.cutoff_mode`, but no code links them.
  - Connect `days_per_bin` must match `2_2025_evaluation.py --days_per_bin` (default 7, CLI only, lines 314, 153).

  ## 3. How names are derived

  **Combine**
  - Climatology columns: `predicted_prob_day_<k>` (including day_0) becomes `<out_prefix>_day_<k>` (CFU:156,170). Only id, time, year and those columns are kept (CFU:173-174).
  - Plus columns: `<prefix>_day_<maxk+1>plus` is hardcoded, = max(0, 1−Σ day columns), including day_0, with skipna (CFU:66-98, esp. 86, 96-97). For example `clim_p_onset_day_29plus` or `aifs_p_onset_day_46plus`. `plus_label_template` is not consulted, and more than one `*plus` column is an error (CFU:58-63).
  - Forecast daily columns: `<col>_day_<k>` becomes `<m>_<out>_day_<k>` (CFU:309). With `add_plus`, `<m>_<out>_day_<N+1>plus` is added (CFU:315).
  - Constants: `<col>` becomes `<m>_<out>`, e.g. `aifs_ens_onset_thresh` (CFU:282).
  - Truth columns: `true_onset_day`, `true_onset_date` (CFU:122-127).
  - Output file: `<out_dir>/<basename|spec_id>_combined_wide.pkl` (3CD:130-131). Wrappers use `<work_dir>/<combine_spec>_combined_wide.pkl` (RT:782).

  **Connect** (output columns, CU:581-589)
  - Base columns: `id, time, year, lat, lon, onset_threshold, outcome`. `outcome` is `week1..weekN`, `later`, or None (`python/pipelines/_shared/misc.py:159-181`; stored as str/None at CU:590).
  - Climatology logits:
    - `<op>_week<w>` from `<bp>_p_onset_day_*` (CU:452-454; the intermediate `<bp>_p_onset_week<w>` comes from CU:145).
    - `<op>_unc_week<w>` (CU:455-456).
    - `<op>_unc_earlier` from `<up>_p_onset_day_0` (CU:451).
    - Window variants: `<op>_<tag>_week<w>` (CU:166-167, 368-373).
  - Forecast bins are probabilities, not logits: `<name>_p_onset_week<w>` and `<name>_p_onset_<v>_week<w>` (CU:408-411, 125).
  - Rain features: `diff_<name>_week<w>` has no window in the name (CU:571). Others are `<agg>_<name>_<W>day_week<w>` (CU:576). The same names are used as keys in `attrs["unavailable_rain_predictors"]` (CU:554-564).
  - `attrs["rain_horizons"]` holds `{name: {rain_day_max, strict, policy}}` (CU:504-508, 592).

  **Blend helpers** (`python/blending_process/blend_evaluation_utils.py`)
  - `_qx` expands to `_week1.._weekN`, and `a*b` expands to all interaction subsets (189-245).
  - `_qx`-generated terms whose columns appear in `unavailable_rain_predictors` are dropped. Explicit `_weekN` terms are never dropped (248-270).
  - `get_forecast_variant_suffix` (72-81) returns the suffix described in section 2.
  - `forecast_prob_cols` (84-87) builds `<name>_p_onset<suffix>_week<w>` but is never called; it is only imported at `1_blend_evaluation.py:497`.
  - `forecast_label` (90-91) returns `name` for base, else `name_variant`. It feeds `<label>_raw` and the Platt file names (`1_blend_evaluation.py:462-468, 516`).
  - `make_year_tag` (94-100) returns `""`, `_Y`, or `_min_max` from cv `run.true_holdout_years + cv_holdout_years` (103-108).
  - `make_cutoff_tag` (111-116) returns `_fixed_cutoff`, `_no_ref_filter`, or `""`. It takes cv `run.cutoff_mode`, which defaults to `ref_onset` in `1_blend_evaluation.py:100` and `3_fit_final_model.py:215` and is required in apply/export.
  - `input_rds_from_cutoff` (119-126) returns `cv_data_[<res>_][<variant>_]new_pipeline.pkl`. The `resolution` argument is never passed.
  - `resolve_blend_input_path` (129-156) resolves in this order: `input_path`, then an error if `run.input_rds_override` is present, then `work_dir`, then the root plus `run.pipeline_input_dir`.
  - `make_window_suffix` (273-274) returns `_<sy>_<ey>`. Windowed formulas are named `<base_name>_<sy>_<ey>` (340); climatology window models are `clim_raw_<sy>_<ey>` (`1_blend_evaluation.py:453`).
  - These tag and name helpers take their inputs from the cv spec, not from combine or connect, apart from the column prefixes listed above.

  ## 4. Non-obvious behaviour

  **Combine**
  - `join`: only `"full"` gives an outer join, so `"outer"` or a typo silently becomes inner (3CD:99). The chain order is climatologies, then forecasts (3CD:97-103). Constants are always outer-joined across models and then left-joined onto the daily table (3CD:119-124). Truth is always a left join on (id, year) (3CD:109). `year` is recomputed from `time` after the joins (3CD:106).
  - Trimming keeps rows where `true_onset_date` is NaN or `time <= true_onset_date` (3CD:113-116). Rows where time equals the onset date survive, but connect gives them outcome None because lead_day 0 ≤ 0.
  - Sources are concatenated, and duplicate (id, time, year) rows keep the first with a warning (CFU:266, 273-276). Columns not declared in `daily` or `constants` (e.g. `forecast_rain_sd`) are dropped (CFU:224-225). Daily variables within a family are outer-merged (CFU:312).
  - Plus bins use skipna, so a row whose probabilities are all NaN gets plus = 1.0 (CFU:96-97). Connect never uses the plus columns.

  **Connect**
  - The base probability series is always configured alongside the listed variants (CU:392, 407).
    - Without the `--required_probability_prefixes` flag, every configured series must be complete.
    - With the flag (even empty), unlisted incomplete series are skipped with a message, and listed but undeclared ones raise (CU:383-429).
    - So the country template, whose combine has only `p_onset_fixed_cutoff`, fails when connect runs standalone and only works through RT.
  - The onset threshold comes from the first model only and is used in every model's `diff` (CU:337-341, 572-574).
  - lat/lon are parsed from an id of the form `"<lat>_<lon>"` (exactly one underscore, lat within ±90, lon within -180..360); otherwise they are NaN (CU:314-334).
  - Outcome uses `assign_lead_bin(..., allow_earlier=False)`, with lead = true_onset_date − time (CU:347-351).
  - Week sums use days 1..day_max; day_0 is excluded; bin w covers columns (w−1)·dpb to w·dpb (CU:115-125).
    - There is no check that day_max ≥ n_bins·dpb. Short slices sum fewer days or give 0, and days beyond n_bins·dpb are ignored.
    - The numpy sum propagates NaN.
  - `logit_winsor` clips to [1e-4, 0.9999] and applies only to climatology (CU:97-104). Window tags apply only to the conditional base climatology.
  - Rain transform precedence is per-model, then top level, then identity. Power and log transforms clip at 0 and keep NaN (CU:194-232). For `diff`, the value is tf(max rolling sum) − tf(threshold) (CU:572-574).
  - Rain predictor parsing (CU:515-523):
    - The dict form lowercases `agg`.
    - Legacy strings `"<agg>_<N>day"` take the window from the digits of the last `_`-separated token, otherwise a token lookup. `agg` is not lowercased.
    - So `"min_dry_spell"` fails, because the last token is `spell`.
    - An unknown agg raises (CU:550-551).
  - Window tokens are case-insensitive (CU:66-73, 84-89):
    - `trigger`/`window` map to `options.window` (default 5).
    - `dry_spell`/`sum_window` map to `onset_definition.dry_spell.sum_window` (default 10), even when `mode: consecutive_dry`. They do not map to `min_dry_days`.
    - `follow` maps to `follow_days` (21); `min_dry` maps to `min_dry_days` (7).
    - Explicit integers win.
  - `diff` and `max` are computed the same way: the max over the week's start days of the rolling W-day rain sum (CU:542-545). Two `diff` predictors with different windows on one model overwrite each other, because the name has no window.
  - Rain horizon (RHU:23-54, 119-219):
    - Legacy: N = day_max + 10, checks only that columns exist, allows NaN and later days.
    - Exact: rejects columns > N and ≤ 0, and requires finite rain in every row, so a `join: full` gap raises.
    - Truncate: like exact but allows columns > N.
    - Every mode uses only days 1..N (RHU:173; CU:499-503).
    - Start days are limited to ≤ N−W+1 (RHU:208-219). Partial validity uses NaN-ignoring `np.fmax`/`np.fmin` (CU:250-271). No valid start gives an all-NaN column plus an `unavailable_rain_predictors` entry (CU:552-564), which downstream drops `_qx` terms on (blend_evaluation_utils.py:255-263) or turns into an error (816-848).

  **YAML gotcha (verified with `yaml.safe_load`)**
  - An unquoted `window_tags: [2000_2024]`, as written in the template comment at `specs/2025_blend/connect_country_template.yml:19`, parses as the integer 20002024. Connect would then look for `clim_20002024_p_onset_day_*`.
  - `years: 2000:2024` stays a string.

  **Current spec facts**
  - The three 2026 connect specs set no `rain_day_max`, so they run in legacy mode with N = 38. Raw `max_day: 45` and combine `max_day: 45` are consistent with that.
  - `connect_fixed_cutoff_2026_{gencast,ngcm}.yml` have no `onset_spec` and use integer windows.
  - The country template chain is consistent: forecast_template `max_day 40, rain_day_max 40 truncate`, combine `max_day 40`, connect `rain_day_max 40 truncate`, and `dry_spell` resolves to 10, `trigger` to 5, giving `min_forecast_model_10day_*` and `max_forecast_model_5day_*`, which match the cv formula.