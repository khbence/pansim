"""Average and standard deviation across stochastic runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parse import Run, RunSet


@dataclass
class Summary:
    mean: np.ndarray
    std: np.ndarray
    n: np.ndarray
    runs: list[np.ndarray]


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing window. Empty input days do not enter the average."""
    if window <= 1:
        return values.astype(float, copy=True)
    mask = np.isfinite(values)
    filled = np.where(mask, values, 0.0)
    csum = np.cumsum(filled)
    ccnt = np.cumsum(mask.astype(float))
    total = csum.copy()
    count = ccnt.copy()
    total[window:] -= csum[:-window]
    count[window:] -= ccnt[:-window]
    out = np.full(values.shape, np.nan, dtype=float)
    np.divide(total, count, out=out, where=count > 0)
    return out


def _series(run: Run, key: str) -> np.ndarray | None:
    if key not in run.data:
        return None
    return run.data[key]


def summarize(run_set: RunSet, key: str, smoother: int = 1) -> Summary | None:
    series = []
    for run in run_set.runs:
        values = _series(run, key)
        if values is None:
            continue
        series.append(rolling_mean(values, smoother))
    if not series:
        return None
    length = max(len(s) for s in series)
    stacked = np.full((len(series), length), np.nan)
    for i, values in enumerate(series):
        stacked[i, : len(values)] = values
    count = np.sum(np.isfinite(stacked), axis=0)
    mean = np.nanmean(stacked, axis=0)
    if stacked.shape[0] == 1:
        std = np.zeros(length)
    else:
        std = np.nanstd(stacked, axis=0, ddof=1)
    std = np.where(count >= 2, std, 0.0)
    mean = np.where(count > 0, mean, np.nan)
    return Summary(mean=mean, std=std, n=count, runs=series)
