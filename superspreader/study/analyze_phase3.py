#!/usr/bin/env python3
"""Tables for the volume-model experiments: early k, household share, fill fraction at infection."""

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analyze_phase2 import RESULTS, OUT, cell_open, load_trace_full, run_dirs  # noqa: E402

INPUTS = RESULTS / "p3" / "inputs"
ORIGINAL = HERE.parents[1] / "inputRealExample" / "locations0.json"


def capacity_file(label):
    if "raw" in label or "fillone" in label:
        return ORIGINAL
    if "tmed" in label:
        return INPUTS / "locations_vol_typemedian.json"
    return INPUTS / "locations_vol_peak.json"


def quantile(values, fraction):
    values = sorted(values)
    return values[int(fraction * (len(values) - 1))] if values else None


def extras(cell, capacities):
    types = Counter()
    fills = []
    total = 0
    for run in run_dirs(cell):
        trace = load_trace_full(run / "trace.tsv")
        for row in trace:
            if row["infector"] is None:
                continue
            total += 1
            types[row["locType"]] += 1
            capacity = capacities[row["location"]] if row["location"] < len(capacities) else 0
            if capacity > 0 and row["occupancy"] > 0 and row["day"] <= 14:
                fills.append(row["occupancy"] / capacity)
    return {
        "household_share": types[2] / total if total else None,
        "type_share": {str(t): round(c / total, 4) for t, c in sorted(types.items())} if total else {},
        "fill_at_infection": {
            "median": quantile(fills, 0.5),
            "p90": quantile(fills, 0.9),
            "share_above_1": sum(f > 1 for f in fills) / len(fills) if fills else None,
        },
    }


def main(prefixes):
    payload = {}
    cache = {}
    for prefix in prefixes:
        for cell in sorted((RESULTS / "p2").glob(prefix + "*")):
            label = cell.name
            path = capacity_file(label)
            if path not in cache:
                cache[path] = [p["capacity"] for p in json.loads(path.read_text())["places"]]
            print("vol", label, flush=True)
            summary = cell_open(cell)
            summary.update(extras(cell, cache[path]))
            payload[label] = summary
    return payload


if __name__ == "__main__":
    prefixes = sys.argv[2:] or ["vol_"]
    name = sys.argv[1] if len(sys.argv) > 1 else "vol"
    payload = main(prefixes)
    OUT.mkdir(exist_ok=True)
    (OUT / f"p3_{name}.json").write_text(json.dumps(payload, indent=1, default=str) + "\n")
    for label, value in payload.items():
        print(label, "R", round(value["mean"], 3), "k", round(value["k_mom"], 3), value["k_ci"], "zero", round(value["zero"], 3),
              "hh", round(value["household_share"], 3), "v10", round(value["top10_venues_early"], 3), "fill", value["fill_at_infection"],
              "k_cal", value.get("calibrated", {}).get("k"))
