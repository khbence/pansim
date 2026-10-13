#!/usr/bin/env python3
"""W10. Reassign type-5 slots at the smallest fill target that clears the measured presence rates.

Fill 1.10 cannot absorb the assignments once each venue uses its own presence rate.
The pre-authorised escalation raises the target in steps of 0.05. This script tries
1.20, then 1.25 if the 95th percentile of displacement exceeds 5 km.
"""

import json
import math
import random
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
PRESENCE = STUDY / "results" / "round2" / "w10_type5_presence.csv"
LOCS = ROOT / "inputRealExample" / "locations1.json"
AGENTS = ROOT / "inputRealExample" / "agents2.json"
OUT_LOCS = ROOT / "inputRealExample" / "locations3.json"
OUT_AGENTS = ROOT / "inputRealExample" / "agents4.json"
SEED = 20261013


def percentile(values, q):
    xs = np.sort(np.asarray(values, dtype=float))
    if xs.size == 0:
        return None
    pos = q * (xs.size - 1)
    lo = int(math.floor(pos))
    hi = min(xs.size - 1, lo + 1)
    return float(xs[lo] * (1 - (pos - lo)) + xs[hi] * (pos - lo))


def load_presence():
    import csv
    rows = list(csv.DictReader(PRESENCE.open()))
    return rows


def quotas(rows, phi):
    out = []
    for row in rows:
        pi = float(row["pi_used"])
        cap = float(row["capacity"])
        quota = int(math.floor(phi * cap / pi)) if pi > 0 else 0
        out.append(quota)
    return out


def main():
    presence = load_presence()
    print("load locations", flush=True)
    locations = json.loads(LOCS.read_text())
    id_to_place = {}
    for place in locations["places"]:
        id_to_place.setdefault(place["ID"], place)
    venues = []
    for row in presence:
        place = id_to_place[row["location_id"]]
        venues.append(place)
    north = np.array([place["coordinates"][0] for place in venues], dtype=np.float64)
    east = np.array([place["coordinates"][1] for place in venues], dtype=np.float64)
    print("load agents", flush=True)
    agents = json.loads(AGENTS.read_text())
    people = agents["people"]
    # Import the same anchor rule the file was built with.
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_szeged_locations1 import agent_anchors

    records = []
    for index, person in enumerate(people):
        slots = [slot for slot in person["locations"] if slot.get("typeID") == 5]
        if len(slots) != 3:
            continue
        home, work = agent_anchors(person, id_to_place)
        if home is None and work is None:
            continue
        records.append((index, person, slots, home, work))
    print(f"agents with three type-5 slots {len(records)}", flush=True)
    anchor_n = np.empty(len(records), dtype=np.float64)
    anchor_e = np.empty(len(records), dtype=np.float64)
    for i, (_index, _person, _slots, home, work) in enumerate(records):
        if home is not None and work is not None:
            # Distance used for ranking is the min of home and work, computed per venue below.
            anchor_n[i] = home["coordinates"][0]
            anchor_e[i] = home["coordinates"][1]
        elif home is not None:
            anchor_n[i] = home["coordinates"][0]
            anchor_e[i] = home["coordinates"][1]
        else:
            anchor_n[i] = work["coordinates"][0]
            anchor_e[i] = work["coordinates"][1]
    home_n = np.array([
        (home["coordinates"][0] if home is not None else work["coordinates"][0])
        for _i, _p, _s, home, work in records
    ])
    home_e = np.array([
        (home["coordinates"][1] if home is not None else work["coordinates"][1])
        for _i, _p, _s, home, work in records
    ])
    work_n = np.array([
        (work["coordinates"][0] if work is not None else home["coordinates"][0])
        for _i, _p, _s, home, work in records
    ])
    work_e = np.array([
        (work["coordinates"][1] if work is not None else home["coordinates"][1])
        for _i, _p, _s, home, work in records
    ])
    print("rank venues", flush=True)
    order = np.empty((len(records), len(venues)), dtype=np.int32)
    ages = np.empty(len(records), dtype=np.int16)
    old_ids = []
    old_dist = []
    id_index = {place["ID"]: i for i, place in enumerate(venues)}
    chunk = 4000
    for start in range(0, len(records), chunk):
        stop = min(len(records), start + chunk)
        d_home = np.hypot(home_n[start:stop, None] - north[None, :], home_e[start:stop, None] - east[None, :])
        d_work = np.hypot(work_n[start:stop, None] - north[None, :], work_e[start:stop, None] - east[None, :])
        dist = np.minimum(d_home, d_work)
        order[start:stop] = np.argsort(dist, axis=1).astype(np.int32)
        for row, record in enumerate(records[start:stop]):
            _index, person, slots, _home, _work = record
            ages[start + row] = int(person["age"])
            ids = [slot["locID"] for slot in slots]
            old_ids.append(ids)
            dists = []
            for loc_id in ids:
                venue_i = id_index.get(loc_id)
                dists.append(float(dist[row, venue_i]) if venue_i is not None else None)
            old_dist.append(dists)
        if start % 20000 == 0:
            print(f"  ranked {stop}", flush=True)
    age_ok = np.ones((len(records), len(venues)), dtype=bool)
    for v, place in enumerate(venues):
        lo, hi = place["ageInter"]
        age_ok[:, v] = (ages >= lo) & (ages <= hi)

    def solve(phi):
        quota = quotas(presence, phi)
        remaining = np.array(quota, dtype=np.int32)
        if int(remaining.sum()) < len(records) * 3:
            return None, {"phi": phi, "feasible": False, "quota": int(remaining.sum()), "needed": len(records) * 3}
        rng = random.Random(SEED)
        sequence = list(range(len(records)))
        rng.shuffle(sequence)
        chosen = np.full((len(records), 3), -1, dtype=np.int32)
        failed = 0
        for agent_i in sequence:
            picked = []
            for venue_i in order[agent_i]:
                if not age_ok[agent_i, venue_i]:
                    continue
                if remaining[venue_i] <= 0:
                    continue
                picked.append(int(venue_i))
                if len(picked) == 3:
                    break
            if len(picked) < 3:
                failed += 1
                continue
            for venue_i in picked:
                remaining[venue_i] -= 1
            chosen[agent_i] = picked
        displacements = []
        changed = 0
        new_distances = []
        for agent_i, (_index, _person, _slots, _home, _work) in enumerate(records):
            if chosen[agent_i, 0] < 0:
                continue
            dists = []
            for slot, venue_i in enumerate(chosen[agent_i]):
                d_home = math.hypot(home_n[agent_i] - north[venue_i], home_e[agent_i] - east[venue_i])
                d_work = math.hypot(work_n[agent_i] - north[venue_i], work_e[agent_i] - east[venue_i])
                dists.append(min(d_home, d_work))
            new_distances.append(dists)
            old = [d for d in old_dist[agent_i] if d is not None]
            if old:
                displacements.append(float(np.mean(dists) - np.mean(old)))
            old_set = set(old_ids[agent_i])
            new_set = {venues[v]["ID"] for v in chosen[agent_i]}
            if old_set != new_set:
                changed += 1
        report = {
            "phi": phi,
            "feasible": failed == 0,
            "failed_agents": failed,
            "quota": int(quotas(presence, phi) and sum(quotas(presence, phi))),
            "needed": len(records) * 3,
            "unused_quota": int(remaining.sum()),
            "displacement_median_m": percentile(displacements, 0.5),
            "displacement_p95_m": percentile(displacements, 0.95),
            "agents_whose_venues_changed": changed,
            "share_changed": changed / len(records),
        }
        return chosen, report

    reports = []
    chosen = None
    chosen_phi = None
    best_p95 = None
    for phi in (1.20, 1.25):
        print(f"solve phi {phi}", flush=True)
        solution, report = solve(phi)
        reports.append(report)
        print(report, flush=True)
        if not report["feasible"]:
            continue
        p95 = report["displacement_p95_m"]
        if chosen is None or (p95 is not None and best_p95 is not None and p95 < best_p95):
            chosen, chosen_phi, best_p95 = solution, phi, p95
        if p95 is not None and p95 <= 5000:
            break
    if chosen is None:
        raise SystemExit("no feasible reassignment")
    print(f"chosen phi {chosen_phi}", flush=True)
    # Apply to a copy of the agent slots.
    for agent_i, (_index, _person, slots, _home, _work) in enumerate(records):
        if chosen[agent_i, 0] < 0:
            continue
        for slot, venue_i in zip(slots, chosen[agent_i]):
            slot["locID"] = venues[int(venue_i)]["ID"]
    print("write agents4", flush=True)
    OUT_AGENTS.write_text(json.dumps(agents, separators=(",", ":")))
    if not OUT_LOCS.exists():
        print("copy locations3", flush=True)
        shutil.copyfile(LOCS, OUT_LOCS)
    summary = {
        "solver": "greedy nearest venue with quota remaining",
        "seed": SEED,
        "chosen_phi": chosen_phi,
        "reports": reports,
        "note": "Fill 1.10 and 1.15 fail the 2 percent slack test on measured per-venue presence rates, so they are not solved. locations3.json is a copy of locations1.json.",
    }
    out = STUDY / "results" / "round2" / "w10_solve.json"
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
