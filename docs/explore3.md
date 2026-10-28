I inventoried the data-preparation stages: regrid (step 0), raw NetCDF processing (step 1), climatology (step 2), plus combine (step 3), because 3_combine_datasets.py lives under python/pipelines/prepare_data/. Line references are file:line under /home/hcwf/code/onset_blending.

  Files read:
  - python/pipelines/prepare_data/0_regrid_to_shapefile.py, 1_process_raw_nc_files.py, 2_build_climatology.py, 3_combine_datasets.py
  - python/prepare_data/nc_utils.py, onset_utils.py, geometry_utils.py, spatial_id_utils.py, sparse_transform_utils.py, climatology_utils.py, combine_forecasts_utils.py
  - python/rain_horizon_utils.py, python/pipelines/_shared/read_spec.py, _shared_helpers.py, utils/remap_nc.py
  - Callers that patch these specs: python/run_training.py, predict/run_operational_pipeline.py, python/blending_process/connect_utils.py

  sparse_transform_utils.py reads no spec keys. read_spec.load_spec only loads `specs/<type>/<id>.yml` relative to the current directory (read_spec.py:52). validate_spec checks top-level keys, dotted paths and custom checks (read_spec.py:79-93). 0_regrid changes into the repo root (0_regrid:48). Steps 1 and 2 must be run from the repo root.

  # 1. Key inventory

  ## 1a. specs/regrid/*.yml (0_regrid_to_shapefile.py, geometry_utils.get_geometry_cfg)

  | Key | Type | Default | Allowed | Read at | Meaning |
  |---|---|---|---|---|---|
  | geometry.shapefile | path | None | – | geometry_utils:68; 0_regrid:179,196,239; geometry_utils:129-133 | Present: target units are shapefile polygons. Absent: target units are the ground-truth grid cells (IDs `<lat>_<lon>`). |
  | geometry.region_id_col | str | `region_key_col`, else "adm3_name" | column in shapefile | geometry_utils:64-66,137-145 | Stable unit ID. Renamed to internal `adm3_name`. |
  | geometry.region_key_col | str | – | – | geometry_utils:65 | Legacy alias for region_id_col. |
  | geometry.region_name_col | str | None | column in shapefile | geometry_utils:70,148-155,170-171 | Display name, carried as `region_name` into the centroids CSV. |
  | geometry.parent_key_col | str | None | column in shapefile | geometry_utils:72,156-163 | Checked to exist and renamed to `adm2_name`. Written to no output. |
  | geometry.crs | str | "EPSG:4326" | – | geometry_utils:73,177-181,338,518 | Shapefile is reprojected to this CRS; grid boxes and areas are computed in it. |
  | geometry.region_label | str | None | – | geometry_utils:74 | **UNUSED**: copied into cfg, nothing reads it. |
  | geometry.grid_lat_var / grid_lon_var | str | auto: latitude/lat/y/Y and longitude/lon/x/X | – | geometry_utils:75-76,113-114,194(0_regrid),314,503 | NetCDF coordinate names, used only when building weights. |
  | geometry.grid_id_decimal_digits | int 0..15 | None (inferred) | 0..15 | spatial_id_utils:52-67,22-24 | Precision of `<lat>_<lon>` IDs (grid-cell mode, dissemination IDs). |
  | geometry.grid_id_format | str | "trimmed" if digits are set | fixed / trimmed (lowercased) | spatial_id_utils:56-66,25-26 | Setting the format without the digits raises (spatial_id_utils:58-61). |
  | clip_to_coverage | bool | true | – | 0_regrid:150 | Restrict and renormalize weights to cells that have ground-truth data, and give forecasts the same footprint. |
  | ground_truth.nc_folder | path | required | – | 0_regrid:153-156 | Ground-truth source folder. |
  | ground_truth.file_regex | regex | `\.nc$` | – | 0_regrid:155, _list_nc 72-79 | `re.search` on basenames from `glob(*.nc)`; `*_adm3.nc` always excluded. |
  | ground_truth.value_col | str | required if any weights are computed | – | 0_regrid:175-176,194; geometry_utils:249-256 (case-insensitive) | Variable used to find cells with finite data. |
  | ground_truth.weights_in | path | top-level weights_in | – | 0_regrid:162,216-220 | Supplied ground-truth weights; skips building them. |
  | weights_in (top level) | path | None | – | 0_regrid:161 | One supplied weight table for ground truth and every forecast family. |
  | forecasts[] | list | [] | – | 0_regrid:160,271 | Forecast families to regrid. |
  | forecasts[].nc_folder | path | required (KeyError) | – | 0_regrid:273 | Source folder. |
  | forecasts[].file_regex | regex | `\.nc$` | – | 0_regrid:272 | Same matching as ground_truth.file_regex. |
  | forecasts[].value_col | str | "fc" | – | 0_regrid:287 | Used **only** in the derived weight file name. |
  | forecasts[].weights_in | path | top-level weights_in | – | 0_regrid:164-165,277-282 | Supplied weights for this family. |
  | weights_out | path | `<ref_dir>/shared_regrid_weights_<spec_id>.csv` | – | 0_regrid:171,232 | Ground-truth weight CSV. Its directory is `ref_dir` for every other default path; ref_dir defaults to Monsoon_Data/reference. |
  | centroids_out | path | `<ref_dir>/unit_centroids_<spec_id>.csv` | – | 0_regrid:243 | Per-unit centroids, written whenever a shapefile is configured. |
  | report_out | path | `<ref_dir>/regrid_coverage_report_<spec_id>.csv` | – | 0_regrid:256 | Coverage report. |
  | dissemination_cells_file | path | None | – | 0_regrid:251-252,82-87 | Adds a "dissemination" row to the report. A missing file only prints a WARNING. |
  | dissemination_id_col | str | None | – | 0_regrid:89-96 | ID column in that file. Falls back to id / adm3_name / lat+lon, then the first column with a warning (97-104). |

  ## 1b. specs/raw_data/*.yml, step 1 (nc_utils.run_single_pipeline and helpers)

  | Key | Type | Default | Allowed | Read at | Meaning |
  |---|---|---|---|---|---|
  | (top-level presence) | – | required: input, dimensions, output, options, type | – | nc_utils:107 | `dimensions` must exist; a null value is accepted. |
  | type | str | required | rainfall_forecast / ground_truth_rainfall | nc_utils:71,95,2264,2296,2357; other values fail at 2336 | Selects the pipeline. |
  | input.nc_folder | path | required, must be an existing directory | – | nc_utils:102-103,143 | Input folder. |
  | input.file_regex | regex | required | – | nc_utils:144,158-162 | `r No `.nc` filter and no `_adm3` exclusion. |
  | input.value_col | str | required | NetCDF variable (case-sensitive) | nc_utils:113-117,2360 | Rainfall variable. Lowercased for the ground-truth long column and the
  default wide_prefix. |
  | input.wide_day_dim | str | "day" | name after rename | nc_utils:2280 | Lead-day dimension, pandas forecast path only. The tensor path hard-codes "day" (1499). |
  | input.wide_prefix | str | value_col.lower() | – | nc_utils:1888,227>` columns and horizon validation (1902). Does not appear in outputs. |
  | input.parallel | bool | false | must be a bool | nc_utils:64-66,2383 | Use a process pool (spawn). |
  | input.workers | int ≥1 | 5 | must be a positive int (checked even wutils:67-69,2384-2387 | Effective workers = min(workers, number offiles), only when parallel. |
  | input.missing_rain_policy | str | – | – | nc_utils:2299 | Legacy lons.missing_rain_policy is absent. |
  | dimensions.rename | map old→new | {} | – | nc_utils:2361,120-130 (case-insensitive on old); 1395-1402,1491-1498 (both sides lowercased) | Rename NetCDF dimensions to
  lat, lon, time, day, number. `adm3_name`/`adm3` dims become `id` auto
  | output.out_dir | path | required | – | nc_utils:108,2354 | Output directory (created). |
  | output.basename | str | spec_id | – | nc_utils:2356 | Output file se orchestrators set it. |
  | output.write_long | bool | **true** | must be a bool (ground truth only) | nc_utils:100-101,2091,2304,2357 | Write the ground-truth daily long table. |
  | options.window | int | validation requires it for both types; read_ – | nc_utils:72-74,96-97; onset_utils:375 | Trigger window length. |
  | options.onset_definition.wet_day_min_mm | float | 1.0 | – | onset_utils:378 | Wet-day threshold. |
  | …trigger_rule | str | all_days_wet | all_days_wet / first_day_wet (9,396-400 | Which trigger-window days must be wet. |
  | …wet_day_comparison | str | gte | gte / gt (lowercased) | onset_utils:380,401-405 | ≥ or > wet_day_min_mm. |
  | …follow_days | int | 21 | – | onset_utils:381 | Follow-up period ch
  | …followup_anchor | str | after_trigger | after_trigger / onset_day (lowercased) | onset_utils:382,406-410 | Where the follow-up period starts. |
  | …dry_spell.mode | str | consecutive_dry | consecutive_dry / window_ercased) | onset_utils:385,391-395 | Dry-spell veto type. |
  | …dry_spell.min_dry_days | int | 7 | – | onset_utils:386 | consecutive_dry run length. |
  | …dry_spell.dry_day_min_mm | float | = wet_day_min_mm | – | onset_ut |
  | …dry_spell.sum_window | int | 10 | – | onset_utils:388 | window_sum window. Also the connect `dry_spell` token. |
  | …dry_spell.sum_min_mm | float | 5.0 | – | onset_utils:389 | window_
  | options.min_day | int | required for forecasts | must be 1 if rain_day_max is set | nc_utils:72-74,86-89,881,905-907,1936 | First lead day kept. |
  | options.max_day | int | required for forecasts | ≥ rain_day_max | n911,1937 | Last lead day kept, applied at read time. |
  | options.rain_day_max | int>0 | None (legacy) | – | nc_utils:75-94,883-901,1889-1909; rain_horizon_utils:23-54 | Strict rainfall horizon for forecasts. |
  | options.rain_horizon_policy | str | "exact" when rain_day_max is se rejected with an explicit rain_day_max) | rain_horizon_utils:37-54;nc_utils:80,888,1894 | Whether lead days beyond rain_day_max are allowed. Ignored at step 1 when rain_day_max is absent. |
  | options.fixed_cutoff_month_day | "MM-DD" | "06-02" | – | nc_utils:1y) | Start date of the fixed_cutoff probability variant. |
  | options.cutoff_month_day | "MM-DD" | required for ground truth | – | nc_utils:98-99,1417,2076,2185 | Ground-truth search start; onset_day origin. |
  | options.min_year / max_year | int | None | – | nc_utils:174-179 | Fn from the filename. |
  | options.missing_rain_policy | str | keep | keep / drop / zero (lowercased) | nc_utils:2297-2301,1318-1323,1356-1371 | Ground-truth handling of missing rainfall. |
  | options.cell_transform_enabled | bool | false | – | nc_utils:631 | in step 1. |
  | options.cell_transform_file | CSV | required when enabled | columns source_id, target_id, weight | nc_utils:633-660 | Weights must be finite, ≥0, unique pairs,
  positive total per target. |
  | thresholds.constant | number | – | – | onset_utils:200-201 | One threshold for every unit (highest precedence). |
  | thresholds.rule.type | str | – | constant / scale (lowercased) | onshold rule. |
  | thresholds.rule.value | float | required for constant | – | onset_utils:205 | Constant threshold. |
  | thresholds.rule.factor / offset / min / max | float | 1.0 / 0.0 / N:155-158 | scale: clip(t*factor+offset, min, max), applied to the filesource. |
  | thresholds.file | path or number | None | .nc/.nc4/.netcdf, .mat, or numeric string means a scalar | onset_utils:219-240 | Thresholdsource. |
  | thresholds.id_col | str | None | – | onset_utils:275-283 | CSV ID cdm3_col. |
  | thresholds.thresh_col | str | "onset_thresh" | – | onset_utils:273,245 | CSV value column; also the NetCDF variable fallback. |
  | thresholds.adm3_col | str | None | – | onset_utils:286-289 | Legacydm3_name, then id (290-295). |
  | thresholds.lat_col / lon_col | str | lat / lon | – | onset_utils:298-300,243-244 | Legacy lat/lon CSV; also the NetCDF name fallback. |
  | thresholds.lat_var / lon_var / thresh_var | str | lat / lon / "MWme5 | NetCDF threshold variable names. |
  | thresholds.mat_lat_name / mat_lon_name / mat_thresh_name | str | lat / lon / "onset.day.thres" | – | onset_utils:257-259 | .mat threshold variable names. |
  | ref_onset.constant_month_day | "MM-DD" | – | – | onset_utils:67-69 year and unit. |
  | ref_onset.date | "MM-DD" | – | – | onset_utils:67 | **Alias** of constant_month_day. |
  | ref_onset.file | CSV | None (None means no reference onset) | – | oand/or per-unit reference dates. |
  | ref_onset.date_col | str | – | – | onset_utils:79,84-85 | Parseable date column; used if present in the file. |
  | ref_onset.year_col / day_col / base_date | str/str/"MM-DD" | – | – 5-108 | Date = `year-base_date` + day days. year_col also becomes amerge key. |
  | ref_onset.unit_col | str | – | – | onset_utils:80-81,99-104 | Per-unit key (read as str), renamed to `id`. |
  | filter.dissemination_cells_file | CSV | None | – | nc_utils:556-560 | Keep only these IDs. A missing file raises. |
  | filter.id_col | str | None | – | nc_utils:561-570 | ID column in that file. |
  | filter.dissemination_id_col | str | – | – | nc_utils:562 | **Alias** of filter.id_col. Otherwise falls back to id/adm3_name/lat+lon, then the first column with a warning (571-579). |
  | filter.bbox.{lat_min,lat_max,lon_min,lon_max} | float | None | any subset | nc_utils:599-619 | Further spatial restriction. |
  | filter.centroids_file | CSV | None | – | nc_utils:479-502 | Unit lat/lon for bbox (columns lat/center_lat/latitude and lon/center_lon/longitude). |
  | filter.centroids_id_col | str | first of target_id, id, adm3_name, else the first column | – | nc_utils:484-494 | ID column in the centroids file. |
  | filter.max_number | int | None | – | nc_utils:1549-1558,1878,1883-1886 | Keep ensemble members with `number <= max_number`. This compares member index values, it does not count members. |
  | grid_id_decimal_digits / grid_id_format (top level), or geometry.grid_id_* | – | None (inferred) | as in regrid | spatial_id_utils:44-49 (`_geometry_cfg` uses`spec["geometry"]` if present, otherwise the spec itself) | Not documented in any raw_data template. Read wherever step 1 builds IDs from lat/lon: nc_utils:340,428,580,661; spatial_id_utils:326,355. |
  | spec["id"] | injected | = spec_id | – | nc_utils:2351,1904; 2_build:63 | Not a YAML key; any `id:` in the YAML is overwritten. |

  ## 1c. Climatology sections (2_build_climatology.py, climatology_utils.py), same raw_data spec

  | Key | Type | Default | Read at | Meaning |
  |---|---|---|---|---|
  | output.out_dir | path | required | 2_build:81; climatology_utils:53,58 | Base for the default gt_path and out_dir. |
  | input.gt_path | path | `<output.out_dir>/<spec_id>_wide.pkl` (uses spec_id, not output.basename) | climatology_utils:51-54 | Ground-truth wide table to fit. |
  | paths.climatology_out_dir | path | `<dirname(output.out_dir)>/Climatology` | climatology_utils:56-59 | Output directory (created). |
  | paths.climatology_out_stem | str | "climatology_issue" | climatology_utils:60; 2_build:133 | Used only for runs without out_stem, as `<stem>_<run_key>`. |
  | climatology_runtime.workers | int ≥1 | 1; env `CLIM_WORKERS` overrides | 2_build:83-91 | Only used by the paired path (161-171). |
  | climatologies | map run_key→run | required, non-empty | 2_build:66-67,81,101 | One entry per climatology run. |
  | climatologies.<k>.train_year_min/max, test_year_min/max | int | required | 2_build:68-73; climatology_utils:68-71 | Training years (KDE fit) and issue years. |
  | …season_start_md / issue_end_md | "MM-DD" | required | 2_build:69; climatology_utils:72-73,118-136,299-303 | Issue-date window; d0 origin. |
  | …onset_col | str | "onset_day" | climatology_utils:74,92-103 | Column read from the ground-truth table. |
  | …forecast_window | int | exactly one of forecast_window / horizons | 2_build:74-79; climatology_utils:75 | Number of lead days. |
  | …horizons | list of {start_md, end_md, forecast_window} | – | climatology_utils:76,143-159 | Window per issue date. Dates outside every horizon give window -1 and NaN rows. |
  | …conditional | bool | true | climatology_utils:77 | P(onset on d0+k \| no onset by d0) vs unconditional. |
  | …cv_by_year | bool | true | climatology_utils:78,532-551 | Leave-one-year-out KDE. |
  | …min_onset_years | int | 10 | 2_build:116-125 | Drop cells with fewer training onsets. |
  | …out_stem | str | `<paths.climatology_out_stem>_<run_key>` | 2_build:133,140 | Output file stem. |

  ## 1d. specs/combine (3_combine_datasets.py), brief

  Keys read:
  - `input.ground_truth_wide_rds`: required (3_combine:41).
  - `input.climatologies.<n>.rds` (49) and `.out_prefix`, default `<n>_p_onset` (50).
  - `input.climatology_rds`: legacy; prefix `clim_p_onset` (53-58).
  - `forecasts.<f>.max_day`: must be >0 and ≥ every source day (combine_forecasts_utils:243-245,305).
  - `forecasts.<f>.daily[].{col,out,add_plus=false}` (247-253) and `.constants[].{col,out}` (254-257).
  - `forecasts.<f>.sources[].file` (189) and `.years`, given as "a,b,c:d" (193).
  - `options.join`: "full" means outer, anything else inner (3_combine:99).
  - `options.trim_forecasts_after_true_onset`: default false (112).
  - `output.out_dir` (127) and `output.basename`, default spec_id (130).

  **UNUSED** combine keys: `type`, `options.plus_label_template` (the `format_forecast_family` parameter is ignored), `options.default_prob_col`, `climatology.max_day`, `climatology.prob_col`, `sources[].prob_col`.

  ## 1e. Keys in templates that no code reads

  - `filter.years`, `filter.ids`, `filter.cells` (commented in aifs_2026.yml:154-159): never read.
  - `options.fixed_cutoff_month_day` in ground-truth specs (ground_truth_template.yml:54, ref_rain_fixed_cutoff_2026.yml:61): only forecast code reads it (nc_utils:1595,1944). Ground-truth onsets ignore it.
  - `options.cutoff_month_day` in forecast specs (commented in the *_2026 files): only ground-truth code reads it. The one exception is utils/compare_onset.py:707, which reads it from the observation spec.
  - `geometry.region_label`: unused. `geometry.parent_key_col`: checked, never written out.
  - `forecasts[].value_col` in regrid: only names a derived weight file.
  - Template comments that do not match the code:
    - country_template.yml:37 says generated weights have a `target_id` column. They are written as latitude, longitude, adm3_name, weight, because`normalize_regrid_weights` returns a normalized copy that is discarded (geometry_utils:452,527).
    - aifs_2026.yml:140 and ref_rain_2026:111 say the ref variant is "skipped" when ref_onset is omitted. See 4.4: it is computed and equals the raw variant.
    - forecast_template.yml:54-55 says "truncate discards later days". Step 1 actually discards days above options.max_day.
    - prepare_data/README.md:162 says files need days up to max_day+window-1. See 4.2.
    - aifs_2026.yml:110 mentions "R.matlab"; the Python code uses scipy.

  ## 1f. Keys the orchestrators write into raw_data specs (as `specs/raw_data/<id>_op.yml`)

  - Forecast specs: `output.basename` = spec_id, `output.out_dir` = work_dir, `input.nc_folder`, `input.file_regex` (`^<escaped file>$`), `options.rain_day_max`, `options.rain_horizon_policy` (run_training.py:711-735; run_operational_pipeline.py:379-402).
  - Climatology spec: `output.out_dir` (training only), `paths.climatology_out_dir`, `output.basename`, `input.gt_path` (run_training.py:741-750; run_operational_pipeline.py:409-414).
  - Training runs step 1 **and** step 2 with the same climatology spec (run_training.py:850-865), so that file holds both the ground-truth processing keys and the climatology keys.

  # 2. Cross-stage dependencies

  1. **Regrid outputs feed step-1 inputs.** Each source `<stem>.nc` becomes `<stem>_adm3.nc` in the same folder (utils/remap_nc.py:151-153). So raw_data `input.nc_folder` must equal regrid `ground_truth.nc_folder` / `forecasts[].nc_folder`, and raw_data `input.file_regex` must match `_adm3\.nc$`. Step 1 does not exclude non-adm3 files itself (nc_utils:158-162). The filename must also contain a 19xx/20xx year (nc_utils:168).
  2. **Unit IDs must agree everywhere.** Regrid `geometry.region_id_col` values become the `adm3_name` coordinate, then `id` in step 1 (nc_utils:949-1000). They must equal:
     - the values in `thresholds.id_col` (or adm3_col / adm3_name / id);
     - `filter.dissemination_cells_file` with `filter.id_col`;
     - the regrid `dissemination_cells_file` with `dissemination_id_col` (templates use the same file and column `unit_id`);
     - `ref_onset.unit_col`;
     - `filter.centroids_id_col` (target_id).
     Missing threshold IDs raise an error (nc_utils:444-452).
  3. **Centroids.** Regrid `centroids_out` (columns target_id, lat, lon[, region_name]; geometry_utils:384-391) is read by raw_data `filter.centroids_file` with `centroids_id_col: target_id`.
  4. **Grid-ID convention.** Precedence is established IDs, then explicit settings, then inference (spatial_id_utils:217-246).
     - Regrid `geometry.grid_id_*` sets the grid-cell target IDs.
     - Step 1 reads `grid_id_*` at the top level or under `geometry` of the raw_data spec.
     - An explicit convention that conflicts with established IDs (rainfall IDs, cell-transform source_ids, supplied weights) raises.
  5. **Weight formats are not interchangeable.** Regrid weights (latitude, longitude, adm3_name|target_id, weight) cannot be used directly as step-1 `options.cell_transform_file`, which needs source_id, target_id, weight (nc_utils:638-641). Thresholds must be keyed on target IDs when cell_transform is enabled.
  6. **Ground-truth wide table.** It is written to `<output.out_dir>/<output.basename|spec_id>_wide.pkl` (nc_utils:2407). It must equal the same spec's `input.gt_path` (default uses spec_id, climatology_utils:53) and combine `input.ground_truth_wide_rds`. In ref_rain_fixed_cutoff_2026.yml, `gt_path`(Models/dry_spell/imd_clim_mok_date_wide.pkl) is not what its own step 1 writes (Processed_Data/2026/ref_rain_fixed_cutoff_2026_wide.pkl). The orchestrators override gt_path.
  7. **Climatology artifacts.**
     - Output is `<climatology_out_dir>/<out_stem>.pkl`, which must equal combine `input.climatologies.<key>.rds`.
     - Run keys `clim` and `clim_unc` are required by run_training (KeyError, run_training.py:487-495). run_operational_pipeline reads `climatologies.clim.out_stem` and `climatologies.clim_unc.out_stem` directly (run_operational_pipeline.py:417-418), so out_stem is mandatory there.
     - Other run keys are wired when the combine spec uses the same name (run_training.py:498-501).
     - The combine prefix (default `<key>_p_onset`) must match connect `climatology.base_prefix` / `unconditional_prefix`.
     - The 2026 combine specs read `imd_clim_mok_date_clim[_unc]_issue.pkl`, but ref_rain_fixed_cutoff_2026 writes `ref_rain_fixed_cutoff_clim[_unc]_issue.pkl`.
  8. **Onset-day origin.** Ground-truth `onset_day` = onset_date − (year-`options.cutoff_month_day`) (nc_utils:2045,2216). The climatology d0 = issue date −(year-`season_start_md`) (climatology_utils:299-303), and the KDE fits onset_day values. So `climatologies.*.season_start_md` must equal the ground-truth `options.cutoff_month_day` of the spec that produced gt_path. Both templates use 05-01.
  9. **Onset definition.**
     - `options.window` and `options.onset_definition.*` should be identical in the ground-truth and every forecast spec. Nothing enforces this.
     - Connect `onset_spec` loads one raw_data spec (connect_utils:56-73), e.g. `ground_truth_template` or `ref_rain_fixed_cutoff_2026`. It maps tokens: trigger/window → window, dry_spell/sum_window → sum_window, follow → follow_days, min_dry → min_dry_days.
  10. **Start-date keys.** Forecast `fixed_cutoff_month_day` (default 06-02) has no ground-truth counterpart: ground-truth onsets start at max(cutoff_month_day, ref_onset_date). `ref_onset` must be the same in ground truth (start restriction) and forecasts (ref variant).
  11. **Thresholds flow into predictors.** The forecast `onset_thresh` output column goes into combine `constants: {col: onset_thresh}`, becomes `<family>_onset_thresh`, and connect reads it from the **first** forecast model as `onset_threshold` (connect_utils:337-341). That feeds the `diff` rain predictor (573). Ground-truth thresholds define the truth onsets.
  12. **Years.**
      - Ground-truth min_year/max_year must cover the climatology train years.
      - Climatology test years × [season_start_md, issue_end_md] form the issue grid. The combine inner join on (id, time, year) (3_combine:99-103) drops forecast issuedates outside it.
      - Forecast min_year/max_year and combine `sources[].years` should cover the same years.
      - Template values: 2000-2024 everywhere. The 2026 specs train 2000-2022, test 2026, and combine uses "2026:2026".
  13. **Horizons.**
      - Forecast `options.max_day` must be ≤ combine `forecasts.<f>.max_day` (combine_forecasts_utils:305).
      - `options.rain_day_max` and `rain_horizon_policy` must equal connect `forecast_models[].rain_day_max` and `rain_horizon_policy`. The orchestrators reconcile them,raising on mismatch and copying one side to the other (run_training.py:290-320).
      - In legacy mode, connect needs `<model>_rain_mean_day_1..(day_max+10)` (rain_horizon_utils:31-36; connect_utils:487-498), so raw max_day must be ≥ connectday_max+10.
      - The climatology forecast_window must be ≥ connect day_max (connect_utils:135-138).
  14. **Forecast artifact.** `<out_dir>/<basename>_wide.pkl` must equal combine `forecasts.<f>.sources[].file`.

  # 3. Derived names (spec value → name)

  - **Regrid:**
    - `<input stem>_adm3.nc`, coordinate `adm3_name` holding target IDs. Every data variable with lat/lon dims is aggregated, not just value_col(utils/remap_nc.py:114-153).
    - Grid-cell IDs `<lat>_<lon>` formatted by convention (spatial_id_utils:80-104; geometry_utils:410-414).
    - Default CSVs `shared_regrid_weights_<spec_id>.csv`, `unit_centroids_<spec_id>.csv`, `regrid_coverage_report_<spec_id>.csv` in ref_dir (0_regrid:171,232,243,256).
    - Per-unit report = report_out with `.csv` replaced by `_per_unit.csv` (0_regrid:133).
    - Different-grid family weights `<ref_dir>/regrid_weights_<spec_id>_<value_col|fc>.csv`; not configurable (0_regrid:287).
    - Report columns: scope, n_units, n_affected, missing_frac_{min,p05,p25,p50,p75,p95,max} (0_regrid:117-124).
  - **Step 1 files:** `<out_dir>/<output.basename or spec_id>_wide.pkl` and `_long.pkl` (nc_utils:2356,2407,2414).
  - **Step 1 forecast columns:**
    - Intermediate: `<wide_prefix>_day_<lead>` (nc_utils:1010,1248), renamed to bare integers (1916-1918).
    - Output keys: id, time, year, onset_thresh, ref_onset_date.
    - Output day columns, with hard-coded prefixes: `forecast_rain_day_<d>`, `forecast_rain_sd_day_<d>`, `frac_raining_day_<d>`, `predicted_prob_day_<d>`, `predicted_prob_fixed_cutoff_day_<d>`, `predicted_prob_ref_day_<d>` (nc_utils:1688-1697,1826-1836). `<d>` is the NetCDF lead-day integer.
  - **Step 1 ground-truth columns:**
    - Wide: id, year, onset_idx, onset_date, onset_day, cutoff_date (2049-2056).
    - Long: id, time, year, `<value_col.lower()>`, onset_thresh, ref_onset_date, onset_date, onset_flag (2147-2156).
  - **Climatology:** `<out_dir>/<out_stem>.pkl`. Columns time, year, id, model (`clim_kde`, `clim_kde_unc`, `clim_kde_cv`, `clim_kde_unc_cv`, chosen by conditional andcv_by_year), `predicted_prob_day_1..max_h`. The unconditional output adds `predicted_prob_day_0` = CDF(d0) (climatology_utils:554-574).
  - **Combine:**
    - Output `<out_dir>/<basename>_combined_wide.pkl`.
    - Climatology `predicted_prob_day_k` → `<out_prefix>_day_k`, plus `<out_prefix>_day_<max+1>plus` (combine_forecasts_utils:66-98,170).
    - Forecast `<col>_day_k` → `<family>_<out>_day_k`; constants → `<family>_<out>` (282,309).
    - Truth columns: true_onset_day, true_onset_date.
  - **Orchestrator temp specs:** `<id>_op.yml`.

  # 4. Non-obvious behaviour

  1. Validation depends on `type`. Forecasts need options.min_day, max_day and window. Ground truth needs window and cutoff_month_day, and write_long must be a bool. input.workers is validated even when parallel is false (nc_utils:62-103).
  2. Forecast days are cut to [min_day, max_day] **when the file is read** (nc_utils:903-911). The later `[min_day, max_day+window-1]` keep (1939-1940) therefore never adds days. Output probabilities and rain cover lead days min_day..max_day only.
  3. Day columns named "100" or higher are dropped before the onset calculation (nc_utils:1927-1931 keeps names "0".."99" only).
  4. Without ref_onset, the forecast `predicted_prob_ref_*` variant uses start position 1 and equals `predicted_prob_*`. It is not skipped (nc_utils:1746-1756).
  5. Ground-truth start date = max(cutoff_month_day date, ref_onset_date) (nc_utils:2077-2085,2184-2193). Ground truth uses `reject_if_short_followup=True`, forecasts use False (2039,2207 vs 1654, and the default in 1795).
  6. Execution paths are chosen by spec values:
     - Forecast tensor fast path: requires cell-transform weights and dims that become exactly {time, day, lat, lon[, number]} with coordinate variables. Otherwise thepandas path runs (nc_utils:2265-2277,1499-1511).
     - Ground-truth matrix path: requires weights, `write_long: false`, `missing_rain_policy: zero` and lat/lon/time dims. It sets NaN to 0 (2302-2316,1403,1463).
     - Strict finite-rainfall validation (`validate_rain_horizon_frame`) runs only in the pandas path (1900-1909). The tensor path only checks the day coordinate (1516).
  7. `missing_rain_policy`: keep / drop / zero. `drop` removes the whole series containing any NaN, not single dates (1356-1369). It falls back to the legacy`input.missing_rain_policy` (2297-2301).
  8. Thresholds:
     - Precedence: `constant`, then `rule.type: constant`, then `file` (with `rule.type: scale` applied) (onset_utils:200-212). `file` may itself be a number (225-233).
     - CSV ID precedence: id_col, then adm3_col, then adm3_name, then id, then lat_col/lon_col (275-301).
     - Columns are lowercased in `prep_thresholds_id` (nc_utils:403).
     - With no threshold configured, onset_thresh is NaN: no ground-truth onsets and all-zero forecast probabilities, with no error (nc_utils:433-435,1793-1804).
  9. ref_onset precedence: `constant_month_day`, then its alias `date`, then `file`. Within a file, `date_col` (if the column exists) wins over year_col+day_col+base_date. The merge uses whichever of id/year exist on both sides (onset_utils:67-112; nc_utils:530-537).
  10. **`mok_month_day` does not exist anywhere in onset_blending (code or git history).** The only spec-level alias here is `ref_onset.date`. `mok_month_day` exists in ai-almanac `/home/hcwf/code/ai-almanac/modal/blending_app.py:180-197` as the old spelling of `ref_onset_month_day`; the new name wins. It is converted to `{"mode":"constant_month_day", "month_day": ...}` (173-177). That Modal app builds spec dicts in memory (input.value_col, wide_day_dim, wide_prefix, dimensions.rename, options.{min_day, max_day, window, onset_definition, cutoff_month_day}, filter.{dissemination_cells_file, bbox, centroids_file}) and calls `forecast_points`,`read_forecast_points`, `nc_read_*` and `process_*` directly (around 1490-1600, 2173-2260, 2395-2440). `forecast_points` and `read_forecast_points` (nc_utils:1122-1262) have no callers inside onset_blending.
  11. Other aliases:
      - `filter.dissemination_id_col` = `filter.id_col`.
      - `geometry.region_key_col` = `region_id_col`.
      - Dissemination and centroids files without a key column fall back to the first column, with a warning.
      - A missing dissemination file only warns in regrid but raises in step 1.
  12. The year comes from the first 19xx/20xx in the **filename** and overwrites the NetCDF-derived year (nc_utils:168,2288,2325; tensor 1584). It also anchors the cutoff dates. This assumes one season per file.
  13. `filter.max_number` keeps member index values ≤ N, so 0-based numbering keeps N+1 members.
  14. `frac_raining` uses a hard-coded > 1 mm, not wet_day_min_mm (nc_utils:1667,1824).
  15. The hard-coded 10-day / 5 mm arrays in `process_ground_truth_rainfall_id` (2030-2037, 2113-2120) are dead: `find_onset_precomp` ignores them when params are passed(onset_utils:544-548).
  16. Horizon policies:
      - When rain_day_max is absent the policy is "legacy", which keeps the old day_max+10 behaviour.
      - An explicit rain_day_max gives "exact" by default. "truncate" allows later days.
      - Strict mode requires min_day 1 and max_day ≥ rain_day_max (rain_horizon_utils:23-54; nc_utils:75-94).
  17. Regrid modes:
      - With a shapefile, or with grid-cell targets.
      - With `clip_to_coverage` true: weights are restricted and renormalized. A same-grid forecast reuses the ground-truth weights (`grids_equal`). A different grid gets coverage-clipped weights (0_regrid:283-293).
      - With `clip_to_coverage` false: plain area-fraction weights.
      - If every weight table is supplied, there is no overlay and no report, but centroids are still written when a shapefile exists (0_regrid:169,238-246,262-263).
      - Weights below 1e-5 are dropped (geometry_utils:524).
      - Half-cell size falls back to 0.125 for a single-point axis (91).
      - `grid_lat_var`/`grid_lon_var` are honoured when building weights but not when applying them. remap_nc only understands lat/lon/latitude/longitude(utils/remap_nc.py:91-101).
  18. Climatology:
      - Runs differing only in `conditional`, with identical (onset_col, train years, min_onset_years, test years, season_start_md, issue_end_md, forecast_window, horizons, cv_by_year), are computed together and use the workers (2_build:34-51,149-178).
      - `--run <key>` disables pairing.
      - `fit_kde` returns None below 10 samples, a hard-coded limit separate from min_onset_years (climatology_utils:170). With cv_by_year, a cell with exactly 10 yearsgets NaN for in-sample years.
  19. The connect `dry_spell` token resolves to `sum_window` (default 10) even in consecutive_dry mode. For the 2026 specs (min_dry_days 5, no sum_window) it resolves to10 (connect_utils:66-73).
  20. GT `output.write_long` defaults to true, so the long table is written unless it is set to false.