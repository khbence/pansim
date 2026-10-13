#!/usr/bin/env python3
"""Add missing Szeged type 5, 9 and 10 locations and reassign agent schedules.

Reads inputRealExample/locations0.json and agents1.json. Writes
inputRealExample/locations1.json and agents2.json.

Type 5 is an evening food or drink venue (restaurant, cafe, pub, bar,
biergarten, fast food, food court). Type 9 is a place of worship. Type 10 is
a park (OpenStreetMap leisure=park). Places already represented by a locations0
point of the same type are not added again.

Every schedule slot that currently points at a type 5, 9 or 10 location is
reassigned. Distance is the minimum of the distance to the agent's real home
and real workplace. The synthetic commuter boxes at the corner of the grid are
not treated as homes or workplaces. Agents who only have that box, and a
school, are anchored on the school. An agent with several slots of one type
receives that many distinct nearest locations, so the new places are used by
the people who live or work near them.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.request
from collections import Counter, defaultdict

import numpy as np

FOOD = {
    "restaurant",
    "cafe",
    "pub",
    "bar",
    "biergarten",
    "fast_food",
    "food_court",
}
FOOD_CAPACITY = {
    "restaurant": 100,
    "pub": 100,
    "bar": 80,
    "biergarten": 80,
    "cafe": 40,
    "fast_food": 40,
    "food_court": 120,
}
FOOD_AGE = {
    "pub": (18, 99),
    "bar": (18, 99),
}
MATCH_M = {5: 80.0, 9: 80.0, 10: 120.0}
SENTINEL = (120000, 740000)


def parse_args():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=root)
    p.add_argument("--osm-dir", default="/tmp/osm_parts")
    p.add_argument("--boundary", default="/tmp/szeged_boundary.geojson")
    p.add_argument("--locations-in", default="inputRealExample/locations0.json")
    p.add_argument("--agents-in", default="inputRealExample/agents1.json")
    p.add_argument("--locations-out", default="inputRealExample/locations1.json")
    p.add_argument("--agents-out", default="inputRealExample/agents2.json")
    p.add_argument("--summary-out", default="scripts/szeged_locations1_summary.json")
    return p.parse_args()


def fetch(url, data=None, timeout=120):
    req = urllib.request.Request(
        url,
        data=data,
        headers={"User-Agent": "pansim-location-review/1.0 (academic; szeged)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def ensure_inputs(osm_dir, boundary_path):
    os.makedirs(osm_dir, exist_ok=True)
    if not os.path.exists(boundary_path):
        raw = fetch(
            "https://nominatim.openstreetmap.org/lookup?osm_ids=R1025105&format=json&polygon_geojson=1"
        )
        geo = json.loads(raw)[0]["geojson"]
        with open(boundary_path, "w") as f:
            json.dump({"type": geo["type"], "coordinates": geo["coordinates"]}, f)
    if not os.path.exists(os.path.join(osm_dir, "amenity.json")) or not os.path.exists(
        os.path.join(osm_dir, "leisure.json")
    ):
        raise SystemExit(
            f"Need amenity.json and leisure.json in {osm_dir}. "
            "Those are Overpass extracts of Szeged amenities and leisure features."
        )


def point_in_ring(lon, lat, ring):
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def element_center(el):
    if "lat" in el and "lon" in el:
        return el["lon"], el["lat"]
    center = el.get("center")
    if center:
        return center["lon"], center["lat"]
    return None


def load_osm_features(osm_dir, ring, to_eov):
    features = []
    seen = set()
    for name in sorted(os.listdir(osm_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(osm_dir, name)
        elements = json.load(open(path)).get("elements", [])
        for el in elements:
            key = (el.get("type"), el.get("id"))
            if key in seen:
                continue
            seen.add(key)
            tags = el.get("tags") or {}
            if tags.get("access") in {"private", "no"}:
                continue
            amenity = tags.get("amenity") or ""
            leisure = tags.get("leisure") or ""
            if amenity in FOOD:
                ptype = 5
                kind = amenity
            elif amenity == "place_of_worship":
                ptype = 9
                kind = amenity
            elif leisure == "park":
                ptype = 10
                kind = "park"
            else:
                continue
            ll = element_center(el)
            if ll is None or not point_in_ring(ll[0], ll[1], ring):
                continue
            east, north = to_eov.transform(ll[0], ll[1])
            features.append(
                {
                    "ptype": ptype,
                    "kind": kind,
                    "name": (tags.get("name") or tags.get("name:hu") or "").strip(),
                    "north": float(north),
                    "east": float(east),
                    "osm": f"{el['type']}/{el['id']}",
                }
            )
    return features


def dist_ne(a_n, a_e, b_n, b_e):
    return math.hypot(a_n - b_n, a_e - b_e)


def dedupe(features):
    """Collapse a node and a building that are the same named place."""
    kept = []
    for feat in sorted(features, key=lambda f: (f["name"] == "", f["name"], f["osm"])):
        name = " ".join(feat["name"].lower().split())
        merged = False
        for prev in kept:
            d = dist_ne(feat["north"], feat["east"], prev["north"], prev["east"])
            prev_name = " ".join(prev["name"].lower().split())
            if name and name == prev_name and d <= 60:
                merged = True
                break
            if not name and d <= 25:
                merged = True
                break
            if name and not prev_name and d <= 30:
                prev["name"] = feat["name"]
                merged = True
                break
        if not merged:
            kept.append(feat)
    return kept


def new_place(feat, used_ids):
    ptype = feat["ptype"]
    osm_tag = feat["osm"].replace("/", "")
    loc_id = f"osm{ptype}_{osm_tag}"
    n = 2
    while loc_id in used_ids:
        loc_id = f"osm{ptype}_{osm_tag}_{n}"
        n += 1
    used_ids.add(loc_id)
    if ptype == 5:
        capacity = FOOD_CAPACITY[feat["kind"]]
        age = list(FOOD_AGE.get(feat["kind"], (0, 99)))
        essential = 0
    elif ptype == 9:
        capacity = 150
        age = [0, 99]
        essential = 2
    else:
        capacity = 300
        age = [0, 99]
        essential = 0
    return {
        "ID": loc_id,
        "type": ptype,
        "essential": essential,
        "coordinates": [int(round(feat["north"])), int(round(feat["east"]))],
        "area": 0,
        "infectious": 1,
        "state": "ON",
        "capacity": capacity,
        "ageInter": age,
        "_kind": feat["kind"],
        "_name": feat["name"],
        "_osm": feat["osm"],
    }


def nearby_existing(feat, grid, places_by_type):
    cell = 200.0
    gn, ge = int(feat["north"] // cell), int(feat["east"] // cell)
    radius = MATCH_M[feat["ptype"]]
    span = int(radius // cell) + 1
    for dn in range(-span, span + 1):
        for de in range(-span, span + 1):
            for idx in grid.get((feat["ptype"], gn + dn, ge + de), ()):
                place = places_by_type[feat["ptype"]][idx]
                north, east = place["coordinates"]
                if dist_ne(feat["north"], feat["east"], north, east) <= radius:
                    return True
    return False


def build_locations(places, features):
    by_type = defaultdict(list)
    for place in places:
        if place["type"] in MATCH_M:
            by_type[place["type"]].append(place)
    grid = defaultdict(list)
    cell = 200.0
    for ptype, group in by_type.items():
        for i, place in enumerate(group):
            north, east = place["coordinates"]
            grid[(ptype, int(north // cell), int(east // cell))].append(i)

    used_ids = {place["ID"] for place in places}
    added = []
    skipped = Counter()
    for feat in features:
        if nearby_existing(feat, grid, by_type):
            skipped[feat["ptype"]] += 1
            continue
        added.append(new_place(feat, used_ids))
    return added, skipped


def is_real_anchor(place):
    if place is None or place["type"] == 15:
        return False
    north, east = place["coordinates"]
    return (north, east) != SENTINEL


def agent_anchors(person, id_to_place):
    home = work = school = None
    for slot in person["locations"]:
        place = id_to_place.get(slot["locID"])
        if place is None:
            continue
        if slot["typeID"] == 2 and home is None and is_real_anchor(place):
            home = place
        elif slot["typeID"] == 4 and work is None and is_real_anchor(place):
            work = place
        elif slot["typeID"] == 3 and school is None and is_real_anchor(place):
            school = place
    if home is None and work is None:
        home = school
    return home, work


def eligible_mask(locations, age):
    mask = np.empty(len(locations), dtype=bool)
    for i, place in enumerate(locations):
        lo, hi = place["ageInter"]
        mask[i] = lo <= age <= hi
    if not mask.any():
        mask[:] = True
    return mask


def assign_type(people, id_to_place, locations, ptype):
    """Reassign slots whose current location is of this type. Returns distances."""
    north = np.array([p["coordinates"][0] for p in locations], dtype=np.float64)
    east = np.array([p["coordinates"][1] for p in locations], dtype=np.float64)
    ids = [p["ID"] for p in locations]
    masks = {age: eligible_mask(locations, age) for age in range(0, 121)}

    groups = defaultdict(list)
    for person in people:
        slots = []
        for slot in person["locations"]:
            place = id_to_place.get(slot["locID"])
            if place is not None and place["type"] == ptype:
                slots.append(slot)
        if not slots:
            continue
        home, work = agent_anchors(person, id_to_place)
        if home is None and work is None:
            continue
        groups[int(person["age"])].append((person, slots, home, work))

    nearest_distances = []
    all_distances = []
    for age, rows in groups.items():
        mask = masks.get(age, masks[100])
        loc_n = north[mask]
        loc_e = east[mask]
        loc_ids = [ids[i] for i, keep in enumerate(mask) if keep]
        m = len(loc_ids)
        kmax = min(4, m)
        chunk = 2500
        for start in range(0, len(rows), chunk):
            batch = rows[start : start + chunk]
            a_n = np.empty((len(batch), 2), dtype=np.float64)
            a_e = np.empty((len(batch), 2), dtype=np.float64)
            for i, (_, _, home, work) in enumerate(batch):
                if home is not None:
                    a_n[i, 0], a_e[i, 0] = home["coordinates"]
                else:
                    a_n[i, 0] = a_e[i, 0] = np.nan
                if work is not None:
                    a_n[i, 1], a_e[i, 1] = work["coordinates"]
                else:
                    a_n[i, 1] = a_e[i, 1] = np.nan
            d_home = np.hypot(a_n[:, 0:1] - loc_n[None, :], a_e[:, 0:1] - loc_e[None, :])
            d_work = np.hypot(a_n[:, 1:2] - loc_n[None, :], a_e[:, 1:2] - loc_e[None, :])
            dist = np.fmin(d_home, d_work)
            pick = np.argpartition(dist, kmax - 1, axis=1)[:, :kmax]
            for i, (_, slots, _, _) in enumerate(batch):
                chosen = pick[i]
                order = chosen[np.argsort(dist[i, chosen], kind="mergesort")]
                # Stable tie break by id when distances match after the sort key.
                k = len(slots)
                for s, loc_index in enumerate(order[:k]):
                    slots[s]["locID"] = loc_ids[int(loc_index)]
                    d = float(dist[i, loc_index])
                    all_distances.append(d)
                    if s == 0:
                        nearest_distances.append(d)
    return nearest_distances, all_distances


def percentile(values, q):
    if not values:
        return None
    arr = np.sort(np.asarray(values, dtype=np.float64))
    return float(arr[min(len(arr) - 1, int(round(q * (len(arr) - 1))))])


def main():
    args = parse_args()
    try:
        from pyproj import Transformer
    except ImportError:
        sys.path.insert(0, "/tmp/pylibs")
        from pyproj import Transformer

    ensure_inputs(args.osm_dir, args.boundary)
    ring = json.load(open(args.boundary))["coordinates"][0]
    to_eov = Transformer.from_crs("EPSG:4326", "EPSG:23700", always_xy=True)

    print("loading OSM", flush=True)
    features = load_osm_features(args.osm_dir, ring, to_eov)
    deduped = []
    for ptype in (5, 9, 10):
        deduped.extend(dedupe([f for f in features if f["ptype"] == ptype]))
    print(
        "OSM in city, deduped",
        {t: sum(1 for f in deduped if f["ptype"] == t) for t in (5, 9, 10)},
        flush=True,
    )

    loc_path = os.path.join(args.root, args.locations_in)
    print("loading", loc_path, flush=True)
    places = json.load(open(loc_path))["places"]
    before = Counter(p["type"] for p in places)
    added, skipped = build_locations(places, deduped)
    print("skipped as already represented", dict(skipped), flush=True)
    print("adding", Counter(p["type"] for p in added), flush=True)

    public_added = []
    for place in added:
        public = {k: v for k, v in place.items() if not k.startswith("_")}
        public_added.append(public)
    out_places = places + public_added
    loc_out = os.path.join(args.root, args.locations_out)
    print("writing", loc_out, flush=True)
    with open(loc_out, "w") as f:
        json.dump({"places": out_places}, f, ensure_ascii=False)
        f.write("\n")

    id_to_place = {}
    for place in out_places:
        # Keep the first record when an id is duplicated (work_commuter_box).
        id_to_place.setdefault(place["ID"], place)

    agent_path = os.path.join(args.root, args.agents_in)
    print("loading", agent_path, flush=True)
    agents = json.load(open(agent_path))
    people = agents["people"]

    distance_stats = {}
    for ptype in (5, 9, 10):
        pool = [p for p in out_places if p["type"] == ptype]
        # Duplicate ids would collapse in id_to_place; the pool itself is unique here.
        print(f"reassigning type {ptype} across {len(pool)} locations", flush=True)
        nearest, all_d = assign_type(people, id_to_place, pool, ptype)
        distance_stats[str(ptype)] = {
            "locations": len(pool),
            "added": sum(1 for p in added if p["type"] == ptype),
            "slots": len(all_d),
            "nearest_median_m": round(percentile(nearest, 0.5) or 0),
            "nearest_p90_m": round(percentile(nearest, 0.9) or 0),
            "assigned_median_m": round(percentile(all_d, 0.5) or 0),
            "assigned_p90_m": round(percentile(all_d, 0.9) or 0),
        }
        print(" ", distance_stats[str(ptype)], flush=True)

    # Load per location after reassignment.
    load = Counter()
    new_ids = {p["ID"] for p in public_added}
    new_load = Counter()
    for person in people:
        for slot in person["locations"]:
            place = id_to_place.get(slot["locID"])
            if place is None or place["type"] not in (5, 9, 10):
                continue
            # Count a slot only once per type assignment. Slots pointing at these
            # types are the reassigned ones.
            if place["type"] == 5 or place["type"] == 9 or place["type"] == 10:
                load[(place["type"], slot["locID"])] += 1
                if slot["locID"] in new_ids:
                    new_load[place["type"]] += 1
    for ptype in (5, 9, 10):
        counts = [c for (t, _), c in load.items() if t == ptype]
        zeros = sum(1 for p in out_places if p["type"] == ptype) - len(counts)
        distance_stats[str(ptype)]["assignment_min"] = min(counts) if counts else 0
        distance_stats[str(ptype)]["assignment_median"] = round(percentile(counts, 0.5) or 0)
        distance_stats[str(ptype)]["assignment_max"] = max(counts) if counts else 0
        distance_stats[str(ptype)]["locations_with_no_slot"] = zeros
        distance_stats[str(ptype)]["slots_on_new_locations"] = new_load[ptype]
        kinds = Counter(p["_kind"] for p in added if p["type"] == ptype)
        distance_stats[str(ptype)]["added_kinds"] = dict(kinds)

    agent_out = os.path.join(args.root, args.agents_out)
    print("writing", agent_out, flush=True)
    with open(agent_out, "w") as f:
        json.dump(agents, f, ensure_ascii=False)
        f.write("\n")

    missing = 0
    for person in people:
        for slot in person["locations"]:
            if slot["locID"] not in id_to_place:
                missing += 1
    summary = {
        "locations_in": before,
        "locations_out": dict(Counter(p["type"] for p in out_places)),
        "skipped_already_represented": dict(skipped),
        "types": distance_stats,
        "dangling_loc_ids": missing,
        "rules": {
            "type_5": "amenity restaurant, cafe, pub, bar, biergarten, fast_food, food_court",
            "type_9": "amenity place_of_worship",
            "type_10": "leisure park",
            "match_radius_m": MATCH_M,
            "anchor": "minimum distance to real home and real work; school if both are the commuter box",
        },
    }
    # Counter keys in locations_in are ints. JSON needs strings or a list.
    summary["locations_in"] = {str(k): v for k, v in before.items()}
    summary["locations_out"] = {str(k): v for k, v in Counter(p["type"] for p in out_places).items()}
    summary_path = os.path.join(args.root, args.summary_out)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print("dangling", missing)
    print("summary", summary_path)


if __name__ == "__main__":
    main()
