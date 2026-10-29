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
result.save("out/")                       # evaluation summaries, trained.blend, config.yml

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

## Layout

```
onset_blending/
  config.py        BlendConfig
  core/            the stages every blend shares, on arrays and xarray
    io.py            files, each file's format, reading rainfall
    domain.py        units, ids, regridding, areas of interest
    reference.py     thresholds, reference onsets, weights (resolved once)
    onsets.py        observed and forecast onsets
    climatology.py   KDE onset climatology
    features.py      the weekly samples
    design.py        formula design matrices
    parallel.py      worker pools, shared memory
  blend/           blending methods
    logit.py         the multinomial logistic blend
    random_forest.py the CRPS random forest (daily probabilities too)
    _crps_tree.py    its trees (from onset-forest)
  evaluation.py    cross-validation, baselines, scores
  training.py      train(config)
  trained.py       TrainedBlend: what operations applies
  forecast.py      the operational path
  legacy/          the spec-set converter (convert-specs)
configs/           example configs
docs/              design notes (docs/pipeline-redesign.md)
archive/           the retired script interface, its specs and data
tests/
```

## Develop

```bash
pytest
```

`archive/` holds the scripts, spec sets and reference data the package
replaced (see `archive/README.md`; the old readme, with the onset
definition and spec documentation, is `archive/readme-scripts.md`).
