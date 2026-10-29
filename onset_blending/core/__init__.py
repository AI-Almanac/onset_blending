"""The stages every blend shares, on arrays and xarray datasets.

io (files, formats, reading) -> onsets (observed and forecast) ->
climatology -> features (the weekly samples), with domain (units, ids,
regridding, areas) and reference (thresholds, reference onsets, weights)
resolved once, design (formula design matrices) and parallel (worker pools,
shared memory) beside them.
"""
