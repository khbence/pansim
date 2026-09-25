"""Derived daily series, evaluated per run so the ensemble std stays meaningful."""

from __future__ import annotations

import numpy as np

from .columns import DERIVED
from .parse import Run, RunSet


def _get(run: Run, key: str) -> np.ndarray:
    if key not in run.data:
        return np.full(run.n_days, np.nan)
    return run.data[key]


def _diff(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan)
    if len(values) == 0:
        return out
    out[0] = values[0]
    out[1:] = np.diff(values)
    return out


def add_derived(run: Run) -> None:
    n = run.n_days
    e = _get(run, "E")
    i1 = _get(run, "I1")
    i2 = _get(run, "I2")
    i3 = _get(run, "I3")
    i4 = _get(run, "I4")
    i5 = _get(run, "I5_h")
    i6 = _get(run, "I6_h")
    r = _get(run, "R")
    rh = _get(run, "R_h")
    d1 = _get(run, "D1")
    s = _get(run, "S")
    ni = _get(run, "NI")
    t = _get(run, "T")
    p1 = _get(run, "P1")
    p2 = _get(run, "P2")
    vac = _get(run, "VAC")

    infected = e + i1 + i2 + i3 + i4 + i5 + i6
    weighted = i1 + 0.75 * i2 + i3 + i4 + 0.1 * i5 + 0.1 * i6
    recovered = r + rh
    population = run.population()
    beta = np.full(n, np.nan)
    usable = (weighted > 0) & (s > 0) & np.isfinite(ni)
    beta[usable] = (ni[usable] / weighted[usable]) * (population / s[usable])
    pos = np.full(n, np.nan)
    tested = t > 0
    pos[tested] = p1[tested] / t[tested]

    mut_keys = [k for k in run.columns if k.startswith("MUT") and k[3:].isdigit()]
    if mut_keys:
        wild = 100.0 - np.sum([run.data[k] for k in mut_keys], axis=0)
    else:
        wild = np.full(n, np.nan)

    derived = {
        "I_all": infected,
        "I_sym": i3 + i4,
        "I_beta": weighted,
        "H_covid": i5 + i6,
        "D_new": _diff(d1),
        "R_all": recovered,
        "R_new": _diff(recovered),
        "P_all": p1 + p2,
        "T_all": t + p2,
        "VAC_cum": np.cumsum(np.nan_to_num(vac)),
        "MUT_WT": wild,
        "beta": beta,
        "pos_rate": pos,
    }
    for key, values in derived.items():
        run.data[key] = np.asarray(values, dtype=float)
        if key not in run.columns:
            run.columns.append(key)


def prepare(run_set: RunSet) -> RunSet:
    for run in run_set.runs:
        add_derived(run)
    return run_set


def metric_options(run_set: RunSet) -> list[dict[str, str]]:
    present: set[str] = set()
    for run in run_set.runs:
        present.update(run.data)
    from .columns import all_metric_keys, describe

    ordered = [k for k in all_metric_keys() if k in present]
    extra = sorted(present - set(ordered) - {"date"})
    options = []
    for key in ordered + extra:
        col = describe(key)
        suffix = ""
        if key in DERIVED:
            suffix = ""
        options.append({"label": f"{col.label}  [{key}]", "value": key})
    return options
