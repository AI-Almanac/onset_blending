# Archive: the retired script interface

What onset_blending was before it became a package, kept for reference and
for converting old spec sets. Nothing here is installed or imported by the
package. The scripts import `onset_blending._legacy`, whose modules now live
in `tests/_legacy` as the tests' oracle, so they are kept to read, not to run.

- `python/`, `predict/`, `utils/`, `benchmark/`: the pipeline scripts (data
  preparation, `1_blend_evaluation.py`, `3_fit_final_model.py`, the
  operational `predict/` path) and analysis scripts.
- `specs/`: the per-stage YAML spec sets they read. Convert one to a
  `BlendConfig` with `python -m onset_blending convert-specs`.
- `Monsoon_Data/`: reference data the specs and the example configs in
  `configs/` point to (thresholds, reference onset dates, dissemination
  cells, grid-to-district weights).
- `requirements.txt`: the scripts' dependencies (the package's are in
  `pyproject.toml`).

The package replaces all of it: see `docs/pipeline-redesign.md` and the
`readme.md` at the root.
