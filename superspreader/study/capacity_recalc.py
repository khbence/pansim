#!/usr/bin/env python3
"""Phases C and D of 12_capacity_work_package: rewrite capacity and area, then validate.

Design capacity (B1) cannot be reconstructed from this location file: the synthetic population
is a sample, venue types are aggregated (8 restaurants serve all agents), and no dwelling, class
or beds data is available offline. The rule used is therefore peak-anchored (B2):

  residences        capacity = assigned members + 1
  every other type  capacity = ceil(HEADROOM * 95th percentile of daily maximum occupancy)
                    observed in an uncontrolled reference run (min 2), or the assigned count
                    where the location was never occupied
  area              capacity * m2 per person of the type (type table below)

Usage: capacity_recalc.py ORIGINAL.json AUDIT_PER_LOCATION.csv OUT_DIR
"""

import csv
import json
import math
import statistics
import sys
from pathlib import Path

HEADROOM = 1.2
LOSS = 1.0
HOME_REMOVAL = 60.0

# type: (m2 per person, height m, m3 per person, ACH low, mid, high)
TYPE_TABLE = {
    2: (15.4, 2.6, 40, 0.3, 0.5, 1.0),
    3: (5.0, 3.0, 15, 1.0, 2.0, 4.0),
    33: (2.0, 3.0, 6, 0.7, 1.5, 4.0),
    4: (10.0, 3.0, 30, 1.0, 2.0, 4.0),
    13: (10.0, 4.0, 40, 1.0, 2.0, 4.0),
    44: (2.9, 3.5, 10, 1.0, 3.0, 6.0),
    5: (1.9, 3.2, 6, 1.0, 3.0, 6.0),
    6: (1.2, 5.0, 6, 3.0, 6.0, 10.0),
    7: (7.1, 3.5, 25, 2.0, 4.0, 8.0),
    8: (8.6, 3.5, 30, 1.0, 3.0, 8.0),
    9: (1.9, 8.0, 15, 0.3, 1.0, 2.0),
    10: (None, None, 500, 10.0, 20.0, 50.0),
    11: (4.0, 3.0, 12, 0.3, 0.7, 1.5),
    12: (8.3, 3.0, 25, 2.0, 4.0, 6.0),
    14: (4.0, 3.0, 12, 1.0, 3.0, 6.0),
    22: (7.1, 2.8, 20, 0.3, 1.0, 2.0),
}
SKIP_TYPES = {1, 15}
STATED_RANGE = {5: (20, 100), 6: (100, 1000), 9: (50, 500)}


def factor_string(scenario):
    column = {"low": 3, "mid": 4, "high": 5}[scenario]
    parts = []
    for kind, row in sorted(TYPE_TABLE.items()):
        removal = row[2] * (row[column] + LOSS)
        parts.append(f"{kind}:{removal / HOME_REMOVAL:.5g}")
    return ",".join(parts)


def read_audit(path):
    rows = {}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            rows[row["ID"]] = row
    return rows


def recalc(places, audit, mode):
    out = []
    medians = {}
    if mode == "typemedian":
        by_type = {}
        for place in places:
            row = audit[place["ID"]]
            if place["type"] in TYPE_TABLE and place["type"] != 2 and int(row["daily_max_p95"]) > 0:
                by_type.setdefault(place["type"], []).append(int(row["daily_max_p95"]))
        medians = {kind: statistics.median(values) for kind, values in by_type.items()}
    for place in places:
        new = dict(place)
        kind = place["type"]
        row = audit[place["ID"]]
        assigned = int(row["assigned_candidates"])
        if kind in TYPE_TABLE:
            if kind == 2:
                capacity = assigned + 1
            else:
                p95 = int(row["daily_max_p95"])
                if mode == "typemedian" and kind in medians:
                    capacity = math.ceil(HEADROOM * medians[kind])
                elif p95 > 0:
                    capacity = math.ceil(HEADROOM * p95)
                else:
                    capacity = assigned
                capacity = max(2, capacity)
            new["capacity"] = int(capacity)
            m2 = TYPE_TABLE[kind][0]
            if m2 is not None and kind != 2:
                new["area"] = int(round(capacity * m2))
        out.append(new)
    return out


def validate(original, new, audit_second=None):
    report = {}
    by_type = {}
    for old, now in zip(original, new):
        by_type.setdefault(old["type"], []).append((old, now))
    types = {}
    for kind, pairs in sorted(by_type.items()):
        caps_new = [n["capacity"] for _, n in pairs]
        caps_old = [o["capacity"] for o, _ in pairs]
        changed = sum(1 for o, n in pairs if o["capacity"] != n["capacity"])
        median = statistics.median(caps_new)
        types[kind] = {
            "locations": len(pairs),
            "changed": changed,
            "total_capacity_old": sum(caps_old),
            "total_capacity_new": sum(caps_new),
            "median_old": statistics.median(caps_old),
            "median_new": median,
            "max_over_median_new": max(caps_new) / median if median else None,
            "distinct_new": len(set(caps_new)),
            "all_positive": all(c > 0 for c in caps_new),
        }
        if kind in STATED_RANGE:
            low, high = STATED_RANGE[kind]
            types[kind]["fraction_in_stated_range_new"] = sum(low <= c <= high for c in caps_new) / len(caps_new)
    report["types"] = types
    if audit_second is not None:
        exceed = {}
        for place in new:
            row = audit_second.get(place["ID"])
            if row is None or place["type"] in SKIP_TYPES or place["type"] not in TYPE_TABLE:
                continue
            peak = int(row["peak_occupancy"])
            exceed.setdefault(place["type"], [0, 0, 0])
            if peak > 0:
                exceed[place["type"]][1] += 1
                exceed[place["type"]][0] += peak > place["capacity"]
            exceed[place["type"]][2] += 1
        report["held_out_peak_exceeds_capacity"] = {
            kind: {"exceeding": a, "occupied": b, "locations": c, "fraction": a / b if b else None}
            for kind, (a, b, c) in sorted(exceed.items())
        }
    return report


def main(original_path, audit_path, out_dir, held_out_path=None):
    original = json.loads(Path(original_path).read_text())
    audit = read_audit(audit_path)
    held_out = read_audit(held_out_path) if held_out_path else None
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reports = {}
    for mode in ("peak", "typemedian"):
        places = recalc(original["places"], audit, mode)
        payload = dict(original)
        payload["places"] = places
        path = out / f"locations_vol_{mode}.json"
        path.write_text(json.dumps(payload, separators=(",", ":")))
        reports[mode] = validate(original["places"], places, held_out)
    factors = {name: factor_string(name) for name in ("low", "mid", "high")}
    (out / "volume_type_factors.json").write_text(json.dumps(factors, indent=1) + "\n")
    (out / "capacity_validation.json").write_text(json.dumps(reports, indent=1) + "\n")
    print(json.dumps(factors, indent=1))
    for mode, rep in reports.items():
        print(mode)
        for kind, info in rep["types"].items():
            print(" ", kind, info["locations"], "changed", info["changed"], "median", info["median_old"], "->", info["median_new"],
                  "total", info["total_capacity_old"], "->", info["total_capacity_new"], "max/med", round(info["max_over_median_new"], 1))
        if "held_out_peak_exceeds_capacity" in rep:
            for kind, info in rep["held_out_peak_exceeds_capacity"].items():
                print("  heldout", kind, info["exceeding"], "/", info["occupied"])


if __name__ == "__main__":
    main(*sys.argv[1:])
