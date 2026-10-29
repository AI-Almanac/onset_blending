"""The stages every blend shares, on arrays and xarray datasets.

io (files, formats, reading) -> onsets (observed and forecast, by the
onset_rule) -> climatology -> features (the weekly samples), with ids and
domain (units, grid ids, regridding, areas) and reference (thresholds,
reference onsets, weights) resolved once; design (formulas and their
design matrices), predictions, metrics and calibration for the evaluation;
parallel (worker pools, shared memory) beside them.
"""
