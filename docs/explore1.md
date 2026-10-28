 I inventoried every spec key and CLI option read by the evaluation and prediction stages, under /home/hcwf/code/onset_blending. Four findings matter before migrating, because carrying them over as-is would keep them broken:

  - **MME cannot run with two or more blend models.** The `merge(on=id_vars)` at 1_blend_evaluation.py:588 raises KeyError 'lat', because the climatology predictions carry no lat/lon. I checked this with in-memory pandas. It is not inside a try, so the script would abort. Every spec has `mme.enabled: false`, so this path has never run.
  - **Formula `window_variants` cannot change anything.** The `replace` is applied to the formula after `_qx` has already been expanded, so `from: prob_clim_qx` never matches. The variant is the base formula under a new name. I confirmed this in memory.
  - **Unused template keys:** `run.MR`, `models.window_variants.include_baseline_1900` and `extras.forecasts[].fair_brier`. `compute_fair_brier5` is never called.
  - **3_produce_figures.py reads file names the current scripts no longer write** (details in section 4).

  Paths below are relative to /home/hcwf/code/onset_blending. "1_blend" = python/pipelines/blending_process/1_blend_evaluation.py; "utils" = python/blending_process/blend_evaluation_utils.py.

  ## 1. Spec keys (specs/2025_blend/cv_models_*.yml)

  | Key | Type / allowed values | Default when absent | Read at | Meaning |
  |---|---|---|---|---|
  | `n_bins` (top level) | int | 4 | utils:64-66, used at :287 | Only used when `build_formulas_from_spec` gets no `n_bins`. That happens only on apply_blend_model.py:170 (old coefficient files without a saved formula). 1_blend and 3_fit work the bin count out from the data (1_blend:155-156; 3_fit:279-280). Only the country template sets it. |
  | `geometry.grid_id_decimal_digits` | int 0..15 | none: taken from the input file's IDs, or inferred from lat/lon | spatial_id_utils.py:52-67, 217-246 | Grid-ID precision. Errors if it conflicts with the IDs already in the input (:225-232). If there is no `geometry:` block, the top-level `grid_id_decimal_digits` is read instead (:44-49). No cv spec sets it. |
  | `geometry.grid_id_format` | `trimmed`\|`fixed` | `trimmed` | spatial_id_utils.py:56-66 | Requires `grid_id_decimal_digits`. |
  | `run.cutoff_mode` | str: `fixed_cutoff`→`_fixed_cutoff`, `no_ref_filter`→`_no_ref_filter`, anything else→`""` | `ref_onset` in 1_blend:100 and 3_fit:215; **required** (KeyError) in apply_blend_model.py:360 and export_blend_output.py:181 | utils:111-126, 318-320 | Sets the output-name tag, the default input file name `cv_data_[<variant>_]new_pipeline.pkl`, and the `window_variants.only_if_cutoff_mode` gate. |
  | `run.MR` | bool | — | — | **UNUSED** (no read anywhere). |
  | `run.training_years` | list[int] | 2000..2024 (1_blend:101, 3_fit:216-219); **required** in apply:362 but the value is unused there | utils:938-942, 634, 529; 3_fit:223-225 | Years for global-CV training, the Platt CV data subset, Platt export fitting, and the final fit. |
  | `run.cv_holdout_years` | list[int] | [] | utils:107 | Test years (one fold each). |
  | `run.true_holdout_years` | list[int] | [] | utils:106; 1_blend:102; 3_fit:220-225 | Put in front of the CV holdout years to form `holdout_years` (utils:108). Excluded from training in global CV (:941), Platt CV (:571, :589) and the final fit. Not excluded from Platt export (:529) unless absent from training_years. |
  | `run.pipeline_input_dir` | str | "" | utils:153-155 | Standalone input = `Monsoon_Data/Processed_Data/pipeline_input/<dir>/cv_data_..._new_pipeline.pkl`. Ignored when `--input_path` (or `--work_dir` in 1_blend) is given. export does not read it. |
  | `run.pipeline_output_dir` | str | "" | 1_blend:95-96 only | Results go to `Monsoon_Data/results/<dir>` only when `--results_dir` is absent. run_training always passes `--results_dir`. 3_fit, apply and export ignore it (the comment in cv_models_fixed_cutoff_one_2026.yml:15 notes this). |
  | `run.input_rds_override` | — | — | utils:142-146 | Removed: its presence raises ValueError. |
  | `cv.methods` | list: `global`, `local`, `neighbors`, `cluster*` | ["global"] | 1_blend:105, 293-334 | `neighbors` raises NotImplementedError (utils:1127), re-raised as RuntimeError. Unknown methods warn and are skipped. |
  | `cv.cluster_var` | str (a column name) | `cluster` | 1_blend:329 | Grouping column for cluster CV. |
  | `cell` | mapping | required in 1_blend:127 and 3_fit:249; optional in apply:127 | | |
  | `cell.dissemination` | CSV path | 1_blend: "" (fails); 3_fit/apply: `--dissem_file` > spec > `Monsoon_Data/dissemination_cells.csv` | 1_blend:128-129; 3_fit:250-255; apply:128-133 | Allowed cells: training is restricted to them, scoring too. |
  | `cell.id_col`, alias `cell.dissemination_id_col` | str | none | 1_blend:130-141; 3_fit:256-267; apply:134-143 | Column renamed to `id`; error if missing. Otherwise `id`, then `adm3_name`, then lat/lon (spatial_id_utils:310-348). |
  | `models.formulas` | mapping name→{enabled,text} | required, non-empty | utils:283-285 | |
  | `models.formulas.<n>.enabled` | bool | **True** in utils:291, but treated as False by run_training.py:350, 382, 435 | | |
  | `models.formulas.<n>.text` | str | required if enabled | utils:292-302 | Terms split on `+`. A term containing `_qx` is repeated per week (`_week1.._weekN`). `a*b*c` becomes all interaction subsets, within each week only (:189-229). Terms without `_qx` appear once. `_qx` terms whose columns the connect stage marked unavailable (`attrs["unavailable_rain_predictors"]`) are dropped (:248-270). |
  | `models.window_variants` | mapping or list of mappings | none | utils:312-315 | |
  | `…enabled` | bool | False | :316 | |
  | `…only_if_cutoff_mode` | str | none (always) | :318-320 | |
  | `…base_name` | must name an enabled formula | required | :321-323 | |
  | `…start_years` | list[int] (1900 skipped) | required | :324-327, 335-336 | |
  | `…end_year` | int | required | :325-327 | |
  | `…replace.from` / `.to` | str; `{window}`→`_<sy>_<end>` | required | :328-339 | Creates model `<base>_<sy>_<end>`. See §5 (B): currently has no effect. |
  | `…include_baseline_1900` | bool | — | — | **UNUSED.** |
  | `mme.enabled` | bool | False | 1_blend:529; run_training.py:402 | Also needs at least one successful clim_logits task. |
  | `mme.variants` | list[str] | ["fixed_cutoff"] | 1_blend:531 | Only changes labels and file names; does not select data. |
  | `mme.mc_cores` | int | 1 (`--cores` overrides) | 1_blend:533 | Passed through but ignored by `optimize_mme_weights` (utils:1531-1532). |
  | `mme.blend_models[].name` | str | required | 1_blend:540 | |
  | `mme.blend_models[].source` | `clim`\|`forecast` | anything else: warn and skip the variant | :555-572 | |
  | `mme.blend_models[].cal_variant` | str | `base` | :562; run_training:407 | Looks up `<name>` or `<name>_calibrated_<cv>`. |
  | `mme.optimize_if_holdout_is_full_2000_2024` | bool | True | :544 | Not in any template. |
  | `mme.weights_year_tag` | str (must include the leading `_`) | the run's year tag | :615 | Not in any template. |
  | `extras.clim_logits[].name` | str; `unc_clim_raw` and `clim_raw` are special | required | 1_blend:397; utils:45 | |
  | `extras.clim_logits[].base_col_prefix` | str | required; `prob_clim` used if no `clim_raw` task (utils:48) | 1_blend:398 | |
  | `extras.clim_logits[].earlier_col` | str | required for `unc_clim_raw` | :407-409 | |
  | `extras.clim_logits[].earlier_is_logit` | bool | True | :415 | |
  | `extras.clim_logits[].window_start_years` / `window_end_year` | list[int] / int | [] / None | :400-401, 449-454 | Only used when name is `clim_raw`. |
  | `extras.forecasts[].name` | str | required | :460 | |
  | `extras.forecasts[].variant` | must be a key of `forecast_variants` | `base` | :461 | |
  | `extras.forecasts[].raw` | bool | False | :464 | |
  | `extras.forecasts[].calibrated` | bool | False | :476 | |
  | `extras.forecasts[].export_platt_weights` | bool | False | :495 | Only takes effect inside the calibrated branch. |
  | `extras.forecasts[].fair_brier` | bool | — | — | **UNUSED.** |
  | `extras.forecast_variants` | mapping variant→column suffix | `{base:"", ref_onset:"_ref", fixed_cutoff:"_fixed_cutoff"}`, replaced wholesale if given | utils:72-81; run_training:367-371 | |
  | `extras.external_predictions[].name` / `.file` / `.method` | str / path relative to the results dir / str | required / required / `global` | utils:1405-1422 | Not in
  any template. |

  2_2025_evaluation.py and 3_produce_figures.py read **no spec at all**

  ## 2. CLI arguments

  **1_blend_evaluation.py** (:68-83; reads `specs/2025_blend/<id>.yml` ctory, :85)
  - `--spec_id` (default `cv_models`, which does not exist).
  - `--cores` int (None → 1 CV worker; overrides `mme.mc_cores`; must b
  - `--work_dir` (input becomes `<dir>/cv_data_<variant>_new_pipeline.pkl`; overrides `run.pipeline_input_dir`).
  - `--input_path` (exact connect pickle; highest priority; duplicates
  - `--results_dir` (overrides `run.pipeline_output_dir`; also the base for `external_predictions` files).

  **predict/3_fit_final_model.py** (:189-210)
  - `--spec_id` required.
  - `--model` required: a formula name, including window-variant names; duplicates `models.formulas` keys.
  - `--method` (`global`; only used in the file name; the fit is always
  - `--tag` (`final`).
  - `--out_dir` (`Monsoon_Data/results/2025_model_evaluation`; ignores
  - `--input_path`.
  - `--dissem_file` (overrides `cell.dissemination`).
  - `--pipeline_input_dir` (root, default `Monsoon_Data/Processed_Data/pipeline_input`; `run.pipeline_input_dir` is appended).

  **predict/apply_blend_model.py** (:319-356)
  - `--spec_id`, `--model`, `--year` (int) required.
  - `--year_tag`: written without the leading `_`; replaces the year tag derived from the holdout years.
  - `--method` (`global`).
  - `--coef_tag`: replaces the whole tag, giving `coefs_{model}_{method}_{coef_tag}.pkl`.
  - `--input_path`: switches to **future mode** (no year filter, no meto sample-support filter).
  - `--coef_dir` (`Monsoon_Data/results/2025_model_evaluation`).
  - `--out_dir` (`<coef_dir>/per_year`).
  - `--dissem_file`; `--pipeline_input_dir` (root; historical mode only).

  **predict/export_blend_output.py** (:143-171)
  - `--issue_date` and `--spec_id` required.
  - `--model` (`blended_model`), `--method` (`global`): only used in standard mode.
  - `--coef_dir` (`<repo>/Monsoon_Data/results/2025_model_evaluation`);ports`).
  - `--pipeline_input_dir`: a path relative to the repo with the file name from cutoff_mode appended. Its default is `Monsoon_Data/Processed_Data/2025_pipeline_input`
  (:254), which differs from `DEFAULT_PIPELINE_INPUT_ROOT` in utils:30,e_input_dir`.
  - `--input_path`, `--cv_preds_file`.
  - `--preds_file`: operational mode; the other input options are ignor

  **predict/run_operational_pipeline.py** (:275-352)
  - `--forecast MODEL SPEC_ID` (repeatable). Legacy alternative: `--model_single`, `--model_ens`, `--aifs_spec`, `--aifs_ens_spec`, `--aifs[_ens]_nc_file`,
  `--aifs[_ens]_nc_folder`; the two styles cannot be mixed (run_trainin
  - Required: `--year`, `--issue_date`, `--clim_spec`, `--combine_spec`, `--connect_spec`, `--blend_spec`, `--coef_dir`, `--coef_tag`, `--work_dir`.
  - `--gt_path` overrides clim `input.gt_path` and combine `input.groun
  - `--blend_input` overrides connect `output_rds`.
  - `--map_output_path` is unused.
  - `--blend_model` (`blended_model`; duplicates the formula name), `--region` (`Ethiopia`, passed to run_maps).
  - `--skip_to` (1), `--stop_at`, `--dry_run`, `--regrid_spec` (optiona
  - Steps (:502-553): build climatology → process raw files per forecast → combine → connect (`--required_probability_prefixes` from the saved coefficient features,
  :188-225, 454-460) → apply (future mode) → export (`--preds_file`) →
  - There is no ground-truth processing step: `<work_dir>/<clim_spec>_wide.pkl` must already exist unless `--gt_path` is given (:413).
  - `clim.out_stem` and `clim_unc.out_stem` are required (KeyError othe

  **python/run_training.py** (:618-684)
  - Same forecast arguments as above.
  - Required: `--clim_spec`, `--combine_spec`, `--connect_spec`, `--bleesults_dir` (the latter overrides `run.pipeline_output_dir`).
  - `--gt_path`, `--blend_input`, `--cores` (passed on to 1_blend), `--skip_to`, `--stop_at`, `--dry_run`.
  - Steps (:851-881): process ground truth (1_process_raw_nc_files withast raw spec → build climatology → combine → connect → 1_blend.
  - Run 1_blend with `--spec_id <blend>_op --work_dir --input_path --results_dir`. The `--work_dir` there is redundant because `--input_path` wins.
  - It does **not** run 3_fit_final_model, 2_2025_evaluation or 3_produ
  - It patches copies of the specs to `specs/<type>/<id>_op.yml` and deletes them on exit (:125-204). Patching includes renaming legacy model aliases in the blend spec
  (:323-355).
  - It checks that every forecast model appears in an enabled formula, `extras.forecasts` or `mme.blend_models` (:420-452).
  - It computes which forecast probability series are required (:358-41

  **2_2025_evaluation.py** (:302-318)
  - `--eval_dir` (`Monsoon_Data/evaluation_2025`), `--out_dir` (`…/2025_model_evaluation/evaluation`), `--platt_dir` (`…/2025_model_evaluation`).
  - `--n_bins` (inferred from `week<n>` columns), `--days_per_bin` (7).ins`/`days_per_bin` with no link.

  **3_produce_figures.py**
  - No CLI. Environment variables `BLEND_RESULTS_DIR` and `BLEND_BOUNDARY_PATH` (:168-174). Output goes to `./figures` (:57).

  ## 3. Cross-stage values that must agree

  - **Climatology prefix.** Connect `climatology.output_prefix` = P (default `prob_clim`, connect_utils.py:356-358) writes `P_week<w>`, `P_unc_week<w>` and
  `P_unc_earlier` as logits (:451-456).
    - The cv `clim_logits[clim_raw].base_col_prefix` must equal `P`.
    - The cv `clim_logits[unc_clim_raw].base_col_prefix` must equal `P_P_unc_earlier`.
    - Formula tokens must use `P_qx`.
    - Connect `base_prefix`/`unconditional_prefix` (`clim`/`clim_unc`) imatologies.*.out_prefix` (`clim_p_onset`/`clim_unc_p_onset`).
  - **Climatology windows.** Connect `climatology.window_tags: [T]` writes `P_T_week<w>` (:364-375). cv `window_start_years`/`window_end_year` read `P_<sy>_<end>_week<w>`
  (utils:461-466), so T must equal `<sy>_<end>`.
  - **Bins.** The bin count N comes from the largest `P_week<w>` the connect stage wrote (connect `n_bins`, :299).
    - cv top-level `n_bins` should match it (it matters only on the old
    - 2_2025 `--n_bins`/`--days_per_bin` must match connect.
    - Outcome labels are `week1..weekN`/`later` (:349-351). All N+1 cla rows (utils:952-964; 3_fit:307-319).
    - External predictions must have exactly `cv_week1..N` + `cv_later` (utils:1442-1456).
  - **Forecast probabilities.** Connect `forecast_models[].name`/`varia_week<w>` and `<name>_p_onset_<variant>_week<w>` (:404-437).
    - cv `extras.forecasts[].name` must equal a connect name.
    - `forecast_variants[variant]` must be `""` or `_<connect variant>`
    - An MME `forecast` source needs a matching `extras.forecasts` entry with `variant == cal_variant` and `calibrated: true` (1_blend:484 vs :563).
    - The wrappers enforce required prefixes ⊆ connect-configured prefixes (run_training:411-416; run_operational:219-224; connect_utils:394-402).
  - **Rain predictors.** Column names are `diff_<name>_week<w>` (no window in the name) or `<agg>_<name>_<int window>day_week<w>` (connect_utils:571-577).
    - Symbolic windows (`trigger`/`dry_spell`/`follow`/`min_dry`) are resolved from the connect `onset_spec` raw spec (:40-95). The formula must use the resolved integer, e.g. `min_forecast_model_10day_qx`.
    - Two `diff` predictors with different windows for the same model write to the same column.
  - **Model names.** `--model`/`--blend_model` must be a `models.formulas` key.
    - run_operational hard-codes method `global` in the coefficient path (:456).
    - `unc_clim_raw` is required as the baseline for the pooled summary (utils:1459-1472).
    - `clim_raw` is hard-coded in the figure maps (3_produce_figures:663-664).
    - The figures also hard-code `ngcm_calibrated_fixed_cutoff`, `ngcm_fixed_cutoff_raw`, `mme_fixed_cutoff_clim_raw_opt_rps`, etc. (:66-80, 474-482).
  - **cutoff_mode.** The connect spec's `mode` key is read by nothing. The default input name from cutoff_mode (`cv_data_fixed_cutoff_new_pipeline.pkl`) does not matchthe connect `output_rds` names in the 2026 specs (`..._2026[_gencast|_ngcm].pkl`), so only `--input_path` works with them.
    - 2_2025 hard-codes the onset-label → cutoff-tag / Platt-label mapping (evaluation_2025_utils.py:30-39, 160-181).
  - **Years.** Training and holdout years must exist in the connect output; otherwise CV errors (utils:945-951).
    - The clim raw spec `climatologies.*.train_year_min/max` and `test_year_min/max` (ref_rain_fixed_cutoff_2026.yml:150-176) mirror 2000-2022 / 2026.
    - combine_template_fixed_cutoff_2026.yml:47,66 restricts forecast sources to `2026:2026`, so with that combine spec there are no 2000-2022 forecast rows to train on.
    - Platt file names in 2_2025 hard-code `_2000_2024` (evaluation_2025_utils:215); the MME optimisation hard-codes 2000..2024 (1_blend:545).
  - **Dissemination and IDs.**
    - cv `cell.dissemination`/`id_col` correspond to raw spec `filter.dissemination_cells_file`/`id_col` (nc_utils.py:540-575; both currently point at`Monsoon_Data/dissemination_cells.csv`). Nothing enforces this.
    - The figures hard-code that CSV and join on lat/lon (:650-654).
    - Base columns are `id, time, year, lat, lon, onset_threshold, outcome` (connect_utils:581). The threshold comes from the first forecast model's `<model>_onset_thresh` (:337-341).

  ## 4. Artifacts

  **1_blend** writes to the results dir. `T = {cutoff_tag}{year_tag}`.
  - `formula_sample_support{T}.json` (:269-273): diagnostics only, no reader.
  - `cv_preds_{model}_{method}{T}.pkl` (:347-349): all wide columns plus `cv_*`. Read by export standard mode (export:100-113, 219).
  - `coefs_{model}_global{T}_year{Y}.pkl` (:354-366): `{coefs[test_year,class,feature,coefficient,intercept], scaler, features, formula}`. Read by apply historical mode (apply:383).
  - `platt_weights_{name}[_{variant}]_calibrated_df{T}.pkl` (:514-519): read by 2_2025 (evaluation_2025_utils:213-224).
  - `mme_optimized_weights{cutoff_tag}_{firstClimName}_{mmeVariant}{year_tag}.pkl` (:611-628): read back by 1_blend itself (:616-636).
  - `summary_models{T}.pkl/.csv` (:667-670) and `yearly_metrics_global{T}.pkl/.csv` (:685-688): read by 3_produce_figures.
  - `summary_models_pooled{T}.csv` (:677-678): no reader.
  - Reads: spec, connect pickle, dissemination CSV, external prediction pickles, saved MME weights.

  **3_fit** writes `coefs_{model}_{method}_{tag}.pkl/.csv` (:336-342; the bundle has no `test_year` column). Read by apply `--coef_tag` and by run_operational:454-460.

  **apply** writes to its out dir:
  - `{model}_{method}_year{Y}_coefs.csv` (:398-399).
  - `{model}_{method}_year{Y}_preds.csv/.pkl` (:504-508): input columns plus `cv_*`. Read by export `--preds_file`.
  - `{model}_{method}_year{Y}_metrics.csv` (:497-498, historical mode only).

  **export** writes `blend_output_summary_{YYYYMMDD}.csv` (:304-306): columns `id, time` (M/D/YY), `week1..N, later, clim_week1..N, clim_later`. Read bypredict/run_maps.py `--input_file` and utils/kiremt_onset_messages.py.

  **2_2025** reads from `eval_dir`:
  - `blended_forecasts_2025_sent.csv`, which needs `clim_*` and `clim_con_*` columns. The export output has no `clim_con_*`, so it cannot be used directly.
  - `ngcm_forecasts_2025_sent.csv` and `ngcm_forecasts_2025_sent_no_mok_filter.csv`.
  - `mr_onset_with_{mok_year,mok_clim,may1}_2025.csv` (:93-103). The header comment (:13-19) lists different file names.

  It writes:
  - to `out_dir`: `blended_forecasts_2025_sent_preonset_{lbl}.csv`, `model_metrics_sent_vs_{obs_ref_year|obs_fixed_cutoff|obs_season_start}.pkl`, `metrics_min_brier_rps_auc_models.csv`, `metrics_min_SENT_plus_NGCMcalibrated.csv`;
  - to `platt_dir`: `yearly_metrics_2025{""|_fixed_cutoff|_no_ref_filter}.pkl/.csv` (:146-294).

  **3_produce_figures** reads the hard-coded `PATHS` (:143-163), which use the old naming:
  - `summary_models_2000_2024_clim_mok_date.pkl`; the writer now produces `summary_models_fixed_cutoff_2000_2024.pkl`.
  - `yearly_metrics_global_clim_mok_date_*`.
  - `yearly_metrics_2025_clim_mok_date` / `_no_mok_filter`; 2_2025 now writes `_fixed_cutoff` / `_no_ref_filter`.
  - `model_metrics_sent_vs_mr_mok_year.pkl`; 2_2025 now writes `..._obs_ref_year.pkl`.

  Only the plain `ref_onset` names match the current writers. The old tag is confirmed by an existing file, `Monsoon_Data/results/dry_spell_aifs_aifs_ens/summary_models_clim_mok_date_2022.csv`. Two required inputs are produced by no script in the repo:`cell_metrics_2000_2024.pkl` and `calibration plots/chartdata_{model}_{period}_10bins.csv`. Several of these are hard requirements: `safe_read_pkl` raises if the file is missing. Outputs go to `figures/`, `figures/reliability/` and `figures/maps_2000_2024/`.

  **run_training / run_operational** wire up these files:
  - `<work_dir>/<clim_spec>_wide.pkl`
  - `<work_dir>/<raw_spec>_wide.pkl` per forecast
  - `<work_dir>/<out_stem>.pkl`
  - `<work_dir>/<combine_spec>_combined_wide.pkl`
  - connect output `<work_dir>/basename(output_rds)` or `--blend_input` (run_training:455-464)
  - run_operational additionally: `preds_pkl` (:474) and `export_csv` (:475).

  ## 5. Non-obvious behaviour

  - **(A) MME.**
    - Weights are optimised only if `optimize_if_holdout_is_full_2000_2024` (default True) and the sorted holdout years equal exactly 2000..2024 (:543-546). The optimiser is a 0.1 simplex grid search followed by BFGS on softmax weights (utils:1516-1575).
    - Otherwise it reads `...{weights_year_tag or year_tag}.pkl`, takes the row with `objective=="rps"` and recomputes RPS. A missing file means warn and skip.
    - Weights are optimised on allowed cells and applied to all cells. Only metrics are kept; no `cv_preds` file is written.
    - The file name uses the first `source: clim` name.
    - The merge failure in the opening list applies to every configuration with two or more sources.
  - **(B) Window variants.**
    - Formulas: 1900 is skipped. The replace runs on the already-expanded formula (utils:302, 333), so it never matches and the variant is a duplicate, as noted at thetop. `dropped_terms` is copied from the base formula.
    - clim_logits: only for `clim_raw`. 1900 means the plain prefix and the plain model name; other years give `clim_raw_<sy>_<end>` (1_blend:449-454; utils:461-464).
  - **(C) Holdout years.**
    - `true + cv` are concatenated. The year tag is `_Y` or `_min_max`, a range label even when the years are not contiguous (utils:94-108).
    - An empty list raises "Global CV produced no fold predictions" (:1076-1077).
    - A holdout year outside `training_years` is predicted from all training years.
    - Platt CV trains on (training ∪ holdout) minus the test year and minus true holdouts. Global CV trains on training years only.
    - Climatology and raw baselines are scored on holdout-year rows only.
    - The final fit excludes only `true_holdout_years`.
  - **(D) External predictions.** Entry `{name, file, method?}`. `file` is joined to the results dir (absolute paths pass through). The pickle must be a non-emptyDataFrame with `id, year, outcome, cv_week1..N, cv_later` and no `cv_week>N`. It is scored like a formula model, including yearly metrics, and is not used for Platt or MME (utils:1397-1456; 1_blend:379-385).
  - **(E) Sample support.** Rows are kept only where every column used by any enabled formula (including window variants) is finite. A column that is missing or never finite is an error (utils:804-895). This filtered frame is then used for all later sections: climatology baselines, raw, calibrated, Platt export and MME.
  - **(F) Platt calibration.**
    - CV calibration regresses on raw p (utils:600-607) and is restricted to allowed cells.
    - The export fits on logit(p) (:543) over all cells in training years, and is applied with logit in 2_2025 (evaluation_2025_utils:184-210).
    - Fallback is intercept 0, slope 1 when there are fewer than 10 rows or a single class.
    - Names: calibrated model is `<name>_calibrated_<variant>`, but the Platt file is `platt_weights_<name>_<variant>_calibrated_df…` (variant before "calibrated").
  - **(G) Error handling.**
    - Formula and CV errors abort the run (1_blend:372-373); climatology, raw and calibrated errors only warn.
    - Yearly metrics cover formula models and external predictions only.
    - The pooled summary needs `unc_clim_raw`.
  - **(H) Input path precedence.** `--input_path` > `--work_dir` > `<root>/<run.pipeline_input_dir>/` + a file name derived from cutoff_mode (utils:129-156).
  - **(I) apply.** It uses the formula saved in the coefficient file in preference to the spec (:163-176). The design matrix must match the saved features exactly (:179-209). The saved scaler is applied. Future mode scores every row of the input file.
  - **(J) export.** Operational mode reads one preds file. Standard mode merges `cv_preds` and the wide pickle on `id` (left join from the climatology rows). Climatology probabilities are `expit(P_week<w>)`, with `clim_later = max(0, 1 - sum)`.
  - **(K) Grid-ID precedence.** IDs already in the input win, then the explicit spec value, then inference from lat/lon (spatial_id_utils:217-246).