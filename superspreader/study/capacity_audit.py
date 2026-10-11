#!/usr/bin/env python3
"""Phase A of 12_capacity_work_package: audit capacity and area in a locations file.

Usage: capacity_audit.py LOCATIONS.json OCCUPANCY.tsv OUT_PREFIX
"""

import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATED_RANGE = {5: (20, 100), 6: (100, 1000), 9: (50, 500)}
AGENTS = ROOT / "inputRealExample" / "agents1.json"


def quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def load_assigned():
    assigned = defaultdict(set)
    people = json.loads(AGENTS.read_text())["people"]
    for index, person in enumerate(people):
        for entry in person["locations"]:
            assigned[entry["locID"]].add(index)
    return {key: len(value) for key, value in assigned.items()}, len(people)


def load_occupancy(path):
    rows = {}
    with open(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            rows[int(parts[0])] = dict(zip(header, parts))
    return rows


def main(locations_path, occupancy_path, prefix):
    places = json.loads(Path(locations_path).read_text())["places"]
    assigned, agent_count = load_assigned()
    occupancy = load_occupancy(occupancy_path)
    per_location = []
    for index, place in enumerate(places):
        row = occupancy.get(index)
        peak = int(row["maxOccupancy"]) if row else 0
        p95 = int(row["dailyMaxP95"]) if row else 0
        capacity = place["capacity"]
        per_location.append(
            {
                "index": index,
                "ID": place["ID"],
                "type": place["type"],
                "capacity": capacity,
                "area": place["area"],
                "assigned_candidates": assigned.get(place["ID"], 0),
                "peak_occupancy": peak,
                "daily_max_p95": p95,
                "peak_over_capacity": peak / capacity if capacity > 0 else None,
            }
        )
    with open(prefix + "_per_location.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_location[0]))
        writer.writeheader()
        writer.writerows(per_location)

    by_type = defaultdict(list)
    for row in per_location:
        by_type[row["type"]].append(row)
    summary = {}
    for kind, rows in sorted(by_type.items()):
        capacities = [r["capacity"] for r in rows]
        positive = [c for c in capacities if c > 0]
        median = statistics.median(positive) if positive else None
        info = {
            "locations": len(rows),
            "capacity_positive": len(positive),
            "area_positive": sum(1 for r in rows if r["area"] > 0),
            "capacity_min": min(capacities),
            "capacity_median": median,
            "capacity_p90": quantile(capacities, 0.9),
            "capacity_max": max(capacities),
            "max_over_median": max(capacities) / median if median else None,
            "distinct_capacities": len(set(capacities)),
            "total_capacity": sum(capacities),
            "total_assigned_candidates": sum(r["assigned_candidates"] for r in rows),
        }
        if kind in STATED_RANGE:
            low, high = STATED_RANGE[kind]
            info["stated_range"] = [low, high]
            info["fraction_in_stated_range"] = sum(low <= c <= high for c in capacities) / len(capacities)
        observed = [r for r in rows if r["peak_occupancy"] > 0]
        info["locations_with_occupancy"] = len(observed)
        if observed:
            ratios = [r["peak_over_capacity"] for r in observed if r["peak_over_capacity"] is not None]
            p95_ratios = [r["daily_max_p95"] / r["capacity"] for r in observed if r["capacity"] > 0]
            info["peak_occupancy_median"] = statistics.median(r["peak_occupancy"] for r in observed)
            info["peak_occupancy_max"] = max(r["peak_occupancy"] for r in observed)
            info["peak_over_capacity_median"] = statistics.median(ratios) if ratios else None
            info["peak_over_capacity_p90"] = quantile(ratios, 0.9)
            info["fraction_peak_above_capacity"] = sum(r > 1 for r in ratios) / len(ratios) if ratios else None
            info["fraction_p95_above_capacity"] = sum(r > 1 for r in p95_ratios) / len(p95_ratios) if p95_ratios else None
            worst = sorted(observed, key=lambda r: -(r["peak_over_capacity"] or 0))[:3]
            info["worst"] = [
                {"ID": r["ID"], "capacity": r["capacity"], "peak": r["peak_occupancy"]} for r in worst
            ]
        summary[kind] = info
    payload = {
        "agents": agent_count,
        "total_capacity_residences": summary.get(2, {}).get("total_capacity"),
        "types": summary,
    }
    Path(prefix + "_summary.json").write_text(json.dumps(payload, indent=1) + "\n")
    for kind, info in summary.items():
        print(
            kind,
            info["locations"],
            "cap med/max",
            info["capacity_median"],
            info["capacity_max"],
            "distinct",
            info["distinct_capacities"],
            "| in range",
            info.get("fraction_in_stated_range"),
            "| peak/cap med",
            info.get("peak_over_capacity_median") and round(info["peak_over_capacity_median"], 3),
            "frac>1",
            info.get("fraction_peak_above_capacity") and round(info["fraction_peak_above_capacity"], 3),
        )


if __name__ == "__main__":
    main(*sys.argv[1:4])
