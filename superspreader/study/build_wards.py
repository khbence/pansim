#!/usr/bin/env python3
"""Split hospital, nursing-home and health-centre locations into wards.

Each agent assigned to one of those locations is moved, with every schedule
slot at that location, onto a single ward. Total capacity and total visits
stay the same. Only who is co-present changes. Ward size is about 30 assigned
agents, and a location with 40 or fewer agents is left whole.
"""

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCATIONS = ROOT / "inputRealExample" / "locations0.json"
AGENTS = ROOT / "inputRealExample" / "agents1.json"
OUT = Path(__file__).resolve().parent / "inputs" / "x21"
SPLIT_TYPES = {12, 14, 22}  # hospital, health centre, nursing home
TARGET = 30
MAX_WHOLE = 40


def split_int(total, weights):
    weights = [max(w, 0) for w in weights]
    weight_sum = sum(weights) or 1
    raw = [total * w / weight_sum for w in weights]
    floors = [int(v) for v in raw]
    left = int(total) - sum(floors)
    order = sorted(range(len(raw)), key=lambda i: (raw[i] - floors[i], weights[i]), reverse=True)
    for i in range(max(left, 0)):
        floors[order[i % len(floors)]] += 1
    # A zero-capacity ward breaks nothing in the hazard, but keep every ward positive when the parent was.
    if total > 0:
        for i, value in enumerate(floors):
            if value == 0 and floors[order[0]] > 1:
                floors[i] = 1
                floors[order[0]] -= 1
    return floors


def main():
    print("load locations", flush=True)
    locations = json.loads(LOCATIONS.read_text())
    places = locations["places"]
    by_id = {}
    for place in places:
        # work_commuter_box is duplicated identically in the city file. The first copy is enough for type lookup.
        by_id.setdefault(place["ID"], place)
    targets = [place for place in places if place["type"] in SPLIT_TYPES]
    print(f"locations {len(places)} split-candidates {len(targets)}", flush=True)

    print("load agents", flush=True)
    agents = json.loads(AGENTS.read_text())
    people = agents["people"]
    assigned = defaultdict(list)
    for index, person in enumerate(people):
        seen = set()
        for slot in person.get("locations", []):
            loc_id = slot.get("locID")
            if loc_id in by_id and by_id[loc_id]["type"] in SPLIT_TYPES and loc_id not in seen:
                assigned[loc_id].append(index)
                seen.add(loc_id)
        if index and index % 50000 == 0:
            print(f"  scanned {index}", flush=True)
    print(f"agents {len(people)}", flush=True)

    new_places = []
    redirect = defaultdict(dict)  # agent index -> {old id: new id}
    rows = []
    for place in targets:
        members = assigned.get(place["ID"], [])
        n = len(members)
        n_wards = 1 if n <= MAX_WHOLE else max(1, int(round(n / TARGET)))
        if n_wards <= 1:
            rows.append({"id": place["ID"], "type": place["type"], "agents": n, "wards": 1, "capacity": place["capacity"]})
            continue
        chunks = []
        start = 0
        base, extra = divmod(n, n_wards)
        for ward in range(n_wards):
            size = base + (1 if ward < extra else 0)
            chunks.append(members[start:start + size])
            start += size
        capacities = split_int(int(place["capacity"]), [len(chunk) for chunk in chunks])
        areas = split_int(int(place.get("area") or 0), [len(chunk) for chunk in chunks])
        for ward, (chunk, capacity, area) in enumerate(zip(chunks, capacities, areas)):
            new_id = f"{place['ID']}__w{ward:03d}"
            clone = dict(place)
            clone["ID"] = new_id
            clone["capacity"] = capacity
            clone["area"] = area
            new_places.append(clone)
            for agent_index in chunk:
                redirect[agent_index][place["ID"]] = new_id
        rows.append({
            "id": place["ID"],
            "type": place["type"],
            "agents": n,
            "wards": n_wards,
            "capacity": place["capacity"],
            "ward_agents_min": min(len(chunk) for chunk in chunks),
            "ward_agents_max": max(len(chunk) for chunk in chunks),
        })

    split_ids = {row["id"] for row in rows if row["wards"] > 1}
    kept = [place for place in places if place["ID"] not in split_ids]
    locations["places"] = kept + new_places

    reassigned_slots = 0
    for agent_index, mapping in redirect.items():
        for slot in people[agent_index].get("locations", []):
            new_id = mapping.get(slot.get("locID"))
            if new_id:
                slot["locID"] = new_id
                reassigned_slots += 1

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "split_types": sorted(SPLIT_TYPES),
        "target_agents_per_ward": TARGET,
        "left_whole_at_or_below": MAX_WHOLE,
        "locations_split": sum(1 for row in rows if row["wards"] > 1),
        "wards_created": len(new_places),
        "agents_reassigned": len(redirect),
        "schedule_slots_reassigned": reassigned_slots,
        "capacity_before": sum(place["capacity"] for place in places if place["type"] in SPLIT_TYPES),
        "capacity_after": sum(place["capacity"] for place in locations["places"] if place["type"] in SPLIT_TYPES),
        "places_before": len(places),
        "places_after": len(locations["places"]),
        "by_location": rows,
    }
    (OUT / "ward_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(
        f"split {summary['locations_split']} locations into {summary['wards_created']} wards, "
        f"reassigned {summary['agents_reassigned']} agents, capacity {summary['capacity_before']} -> {summary['capacity_after']}",
        flush=True,
    )
    print("write locations", flush=True)
    (OUT / "locations_wards.json").write_text(json.dumps(locations, separators=(",", ":")))
    print("write agents", flush=True)
    (OUT / "agents_wards.json").write_text(json.dumps(agents, separators=(",", ":")))
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
