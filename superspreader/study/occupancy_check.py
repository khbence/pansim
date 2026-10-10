#!/usr/bin/env python3
"""A0: is per-person infection risk independent of venue size under the density kernel?

Venue size is the number of agents that list the venue in their schedule. Infections are
taken from the trace, excluding the initial seeds. If the hazard is
k * sum(I) / N_present, the infections per assigned agent should not rise with N.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from analyze import load_trace

ROOT = Path(__file__).resolve().parents[2]


def main(cell, types, window=(1, 28)):
    people = json.load(open(ROOT / "inputRealExample" / "agents1.json"))["people"]
    locations = json.load(open(ROOT / "inputRealExample" / "locations0.json"))["places"]
    type_of = {p["ID"]: p["type"] for p in locations}
    size = Counter()
    for person in people:
        for loc in person["locations"]:
            size[loc["locID"]] += 1
    index_of = {p["ID"]: i for i, p in enumerate(locations)}
    infections = defaultdict(int)
    for run in sorted((Path(__file__).resolve().parent / "results" / cell).glob("rep*")):
        for row in load_trace(run / "trace.tsv"):
            if row["infector"] is None:
                continue
            if window[0] <= row["day"] <= window[1]:
                infections[row["location"]] += 1
    reps = len(list((Path(__file__).resolve().parent / "results" / cell).glob("rep*")))
    by_type = defaultdict(list)
    for loc_id, n in size.items():
        t = type_of.get(loc_id)
        if t in types:
            by_type[t].append((n, infections.get(index_of[loc_id], 0)))
    out = {}
    for t, items in sorted(by_type.items()):
        items.sort()
        edges = [items[int(len(items) * q / 5)][0] for q in range(5)] + [items[-1][0] + 1]
        rows = []
        for b in range(5):
            chunk = [it for it in items if edges[b] <= it[0] < edges[b + 1]]
            if not chunk:
                continue
            total_n = sum(c[0] for c in chunk)
            total_inf = sum(c[1] for c in chunk)
            rows.append(
                {
                    "size_range": [edges[b], edges[b + 1] - 1],
                    "venues": len(chunk),
                    "mean_size": total_n / len(chunk),
                    "infections_per_assigned_agent": total_inf / reps / total_n,
                    "infections_per_venue": total_inf / reps / len(chunk),
                }
            )
        out[str(t)] = rows
    return out


if __name__ == "__main__":
    cell = sys.argv[1]
    res = main(cell, {3, 4, 7, 8, 33})
    Path(__file__).resolve().parent.joinpath("summaries", "occupancy_" + cell.replace("/", "_") + ".json").write_text(json.dumps(res, indent=2))
    for t, rows in res.items():
        print("type", t)
        for r in rows:
            print(f"  N {r['size_range'][0]:>4}-{r['size_range'][1]:<5} venues {r['venues']:5} mean N {r['mean_size']:7.1f}  inf/agent {r['infections_per_assigned_agent']:.4f}  inf/venue {r['infections_per_venue']:.3f}")
