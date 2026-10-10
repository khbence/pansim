#!/usr/bin/env python3
"""Offspring dispersion on an undepleted early window.

Index cases are infections acquired on days FIRST..LAST whose infector is known, so the
initial -r seeds (who start partway through their infectious period) are excluded.
Offspring are children infected within HORIZON days of the index infection. The window
is short enough that susceptible depletion does not truncate the tail.
"""

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

from analyze import json_ready, load_trace, mean_var, mom_k, share_for_fraction_of_mass, top_share

FIRST = 1
LAST = 10
HORIZON = 14


def early_counts(trace, first=FIRST, last=LAST, horizon=HORIZON):
    children = defaultdict(list)
    for row in trace:
        if row["infector"] is not None:
            children[row["infector"]].append(row["day"])
    counts = []
    for row in trace:
        if row["infector"] is None:
            continue
        if first <= row["day"] <= last:
            day = row["day"]
            counts.append(sum(1 for d in children.get(row["agent"], ()) if d <= day + horizon))
    return counts


def stats(values):
    if not values:
        return {"n": 0}
    mean, var = mean_var(values)
    k = mom_k(values)
    return {
        "n": len(values),
        "mean": mean,
        "cv2": (var / (mean * mean)) if mean else None,
        "k_mom": "infinity" if (k is not None and math.isinf(k)) else k,
        "zero": sum(v == 0 for v in values) / len(values),
        "top20": top_share(values, 0.20),
        "p80": share_for_fraction_of_mass(values, 0.8),
    }


def bootstrap_k(values, draws=300, seed=7):
    if len(values) < 30:
        return None
    rng = random.Random(seed)
    n = len(values)
    out = []
    for _ in range(draws):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        k = mom_k(sample)
        if k is not None and not math.isinf(k):
            out.append(k)
    if len(out) < 30:
        return {"finite_fraction": len(out) / draws}
    out.sort()
    return {
        "low": out[int(0.025 * len(out))],
        "high": out[int(0.975 * len(out)) - 1],
        "finite_fraction": len(out) / draws,
    }


def cell_summary(cell_dir, bootstrap=True):
    runs = sorted(p.parent for p in cell_dir.glob("rep*/trace.tsv"))
    pooled = []
    per_run = []
    for run in runs:
        trace = load_trace(run / "trace.tsv")
        counts = early_counts(trace)
        pooled.extend(counts)
        item = stats(counts)
        item["run"] = run.name
        per_run.append(item)
    out = stats(pooled)
    out["replicates"] = len(per_run)
    out["per_run"] = per_run
    if bootstrap:
        out["k_ci"] = bootstrap_k(pooled)
    return out


def main():
    results = Path(__file__).resolve().parent / "results"
    out_dir = Path(__file__).resolve().parent / "summaries"
    out_dir.mkdir(exist_ok=True)
    experiments = sys.argv[1:] or ["exp1", "sweep", "exp2"]
    for experiment in experiments:
        payload = {}
        for cell in sorted(p for p in (results / experiment).iterdir() if p.is_dir()):
            print(experiment, cell.name, flush=True)
            payload[cell.name] = cell_summary(cell)
        (out_dir / f"early_{experiment}.json").write_text(json.dumps(json_ready(payload), indent=2) + "\n")


if __name__ == "__main__":
    main()
