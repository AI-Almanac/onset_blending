# onset_blending

Probabilistic rainy-season onset forecasts that blend AI weather forecasts
(AIFS, GraphCast, Aurora, NeuralGCM, ...) with observed onset climatology.
One installable package owns the whole pipeline in memory: observed and
forecast onsets, the KDE climatology, weekly features, cross-validated
evaluation, the fitted blend, and operational forecasts from live files.

## Install

```bash
pip install -e .            # extras: [regrid] polygon regridding, [maps], [test]
```

## Use

```python
from onset_blending import BlendConfig, TrainedBlend, train

result = train(BlendConfig.from_yaml("configs/india_aifs.yml"))
result.save("out/")                       # scores (CSV), CV predictions (NetCDF), trained.blend

blend = TrainedBlend.load("out/trained.blend")
forecast = blend.forecast({"aifs_ensv2": "live/ens_2026-06-09.nc", "aifsv2": "live/det_2026-06-09.nc"})
forecast.save("out/2026-06-09/")
```

or `python -m onset_blending train | forecast | convert-specs`.

A `BlendConfig` (`configs/`) is the one recipe: data, domain, onset
definition, climatology, features and evaluation. A trained blend carries
its recipe, fitted model, reference data and the forecast structure it was
trained on, and refuses live files that do not cover it (fewer members or
lead days, another grid, other units or daily aggregation); anything more is
trimmed.

The onset threshold is a constant (`onset.threshold.mm`), a per-unit table
(`file`), or computed from the observations (`climatology`: each unit's
mean rainfall over the trigger window, e.g. Moron and Robertson's local
5-day amount over April-October of 1991-2020). With
`definition.threshold_comparison: gte` a window reaching the threshold
triggers; `follow_days: 0` checks no dry spell. Computed thresholds are
kept in `Resources(cache_dir=...)` (`train --cache-dir`) under a key of
their settings and input files, and read back by later runs.

An evaluation writes CSV score tables (`summary_models*`,
`yearly_metrics_global*`), each model's CV predictions and the CV folds'
coefficients as NetCDF (`cv_preds_*.nc`: `Predictions.open`; `coefs_*.nc`:
`blend.logit.fold_models`), Platt and MME weights as CSV, and no pickles.

## Layout

```
onset_blending/
  config.py        BlendConfig
  core/            the stages every blend shares, on arrays and xarray
    io.py            files, each file's format, reading rainfall
    ids.py           unit ids: grid-id conventions, normalisation
    domain.py        areas of interest, dissemination units, regridding
    reference.py     thresholds, reference onsets, weights (resolved once)
    onset_rule.py    the onset search
    onsets.py        observed and forecast onsets
    climatology.py   KDE onset climatology
    features.py      the weekly samples
    design.py        formulas and their design matrices
    predictions.py   a model's probabilities for samples (NetCDF)
    metrics.py       Brier, RPS, AUC, Pietra
    calibration.py   Platt calibration
    parallel.py      worker pools, shared memory
  blend/           blending methods
    logit.py         the multinomial logistic blend; global, local, cluster CV
    random_forest.py the CRPS random forest (daily probabilities too)
    _crps_tree.py    its trees (from onset-forest)
    mme.py           the multi-model ensemble of baselines
  evaluation.py    cross-validation, baselines, scores
  training.py      train(config)
  trained.py       TrainedBlend: what operations applies
  forecast.py      the operational path
  legacy/          the spec-set converter (convert-specs)
configs/           example configs
docs/              design notes (docs/pipeline-redesign.md)
archive/           the retired script interface, its specs and data
tests/             (tests/_legacy: the replaced pipeline code, kept as the oracle)
```

## Develop

```bash
pytest
```

`archive/` holds the scripts, spec sets and reference data the package
replaced (see `archive/README.md`; the old readme, with the onset
definition and spec documentation, is `archive/readme-scripts.md`).
