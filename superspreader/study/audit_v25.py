#!/usr/bin/env python3
"""V0: audit locations1.json and agents2.json against the phase-3 city file."""

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results" / "v25"
FOCUS = (5, 6, 8, 9, 10, 12, 14, 22)


def load_places(path):
    return json.loads(Path(path).read_text())["places"]


def place_table(places):
    rows = {}
    for type_id in FOCUS:
        subset = [place for place in places if place["type"] == type_id]
        caps = sorted(int(place.get("capacity") or 0) for place in subset)
        areas = [float(place.get("area") or 0) for place in subset]
        infectious = Counter(place.get("infectious") for place in subset)
        rows[str(type_id)] = {
            "n": len(subset),
            "capacity_sum": sum(caps),
            "capacity_median": caps[len(caps) // 2] if caps else None,
            "capacity_min": caps[0] if caps else None,
            "capacity_max": caps[-1] if caps else None,
            "area_zero": sum(1 for value in areas if value == 0),
            "area_positive": sum(1 for value in areas if value > 0),
            "infectious": {str(key): value for key, value in infectious.items()},
        }
    return rows


def agent_audit(path, interesting=(5, 9, 10)):
    print("load", path, flush=True)
    people = json.loads(Path(path).read_text())["people"]
    slots = Counter()
    agents_with = Counter()
    assigned = defaultdict(set)
    per_agent_slots = Counter()
    for index, person in enumerate(people):
        seen = set()
        local = Counter()
        for slot in person.get("locations", []):
            type_id = slot["typeID"]
            slots[type_id] += 1
            local[type_id] += 1
            if type_id not in seen:
                agents_with[type_id] += 1
                seen.add(type_id)
            if type_id in interesting:
                assigned[slot["locID"]].add(index)
        for type_id, count in local.items():
            if type_id in interesting:
                per_agent_slots[(type_id, count)] += 1
        if index and index % 50000 == 0:
            print(f"  {index}", flush=True)
    loads = defaultdict(list)
    for loc_id, members in assigned.items():
        loads[loc_id] = len(members)
    return {
        "agents": len(people),
        "slots": {str(key): value for key, value in sorted(slots.items())},
        "agents_with_type": {str(key): value for key, value in sorted(agents_with.items())},
        "slots_per_agent": {
            f"{type_id}x{count}": agents for (type_id, count), agents in sorted(per_agent_slots.items())
        },
        "assignment_loads": loads,
    }


def summarise_loads(loads, places):
    by_type = defaultdict(list)
    type_of = {place["ID"]: place["type"] for place in places}
    for loc_id, count in loads.items():
        by_type[type_of.get(loc_id, -1)].append(count)
    out = {}
    for type_id, values in sorted(by_type.items()):
        values = sorted(values)
        out[str(type_id)] = {
            "locations_with_agents": len(values),
            "agents_assigned_sum": sum(values),
            "min": values[0],
            "median": values[len(values) // 2],
            "p90": values[int(0.9 * (len(values) - 1))],
            "max": values[-1],
        }
    return out


def occupancy_peaks(occupancy_path, places):
    if not Path(occupancy_path).exists():
        return None
    peaks = defaultdict(list)
    with open(occupancy_path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        index = {name: pos for pos, name in enumerate(header)}
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            loc = int(parts[index["location"]])
            if loc >= len(places):
                continue
            peaks[places[loc]["type"]].append(float(parts[index["maxOccupancy"]]))
    out = {}
    for type_id, values in sorted(peaks.items()):
        if type_id not in FOCUS:
            continue
        values = sorted(values)
        out[str(type_id)] = {
            "n": len(values),
            "median_peak": values[len(values) // 2],
            "p90_peak": values[int(0.9 * (len(values) - 1))],
            "max_peak": values[-1],
        }
    return out


def main():
    old_places = load_places(ROOT / "inputRealExample" / "locations0.json")
    new_places = load_places(ROOT / "inputRealExample" / "locations1.json")
    old_ids = [place["ID"] for place in old_places]
    new_ids = [place["ID"] for place in new_places]
    old_agents = agent_audit(ROOT / "inputRealExample" / "agents1.json")
    new_agents = agent_audit(ROOT / "inputRealExample" / "agents2.json")
    slot_delta = {}
    keys = set(old_agents["slots"]) | set(new_agents["slots"])
    for key in sorted(keys, key=lambda item: int(item)):
        slot_delta[key] = new_agents["slots"].get(key, 0) - old_agents["slots"].get(key, 0)
    payload = {
        "locations_old": len(old_places),
        "locations_new": len(new_places),
        "duplicate_ids_old": len(old_ids) - len(set(old_ids)),
        "duplicate_ids_new": len(new_ids) - len(set(new_ids)),
        "places": {"old": place_table(old_places), "new": place_table(new_places)},
        "agents_old": old_agents["agents"],
        "agents_new": new_agents["agents"],
        "slot_delta_new_minus_old": slot_delta,
        "slots_per_agent_old": old_agents["slots_per_agent"],
        "slots_per_agent_new": new_agents["slots_per_agent"],
        "assignment_old": summarise_loads(old_agents["assignment_loads"], old_places),
        "assignment_new": summarise_loads(new_agents["assignment_loads"], new_places),
        "occupancy_old": occupancy_peaks(
            Path(__file__).resolve().parent / "results" / "p3" / "reference_uncontrolled" / "occupancy.tsv",
            old_places,
        ),
    }
    # Drop the raw load maps before writing; they are huge.
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v0_audit.json").write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps({
        "agents": [payload["agents_old"], payload["agents_new"]],
        "slot_delta": slot_delta,
        "places_5_9_10": {
            "old": {key: payload["places"]["old"][key] for key in ("5", "9", "10")},
            "new": {key: payload["places"]["new"][key] for key in ("5", "9", "10")},
        },
        "assignment_new": payload["assignment_new"],
        "occupancy_old": payload["occupancy_old"],
    }, indent=1))


if __name__ == "__main__":
    main()
