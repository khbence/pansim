#!/usr/bin/env python3
"""Round-5 tables. No PanSim runs.

Day numbering follows the earlier notes. Trace day is timestamp // 144.
An index case is infected on trace days 1 to 10, has a known infector, and
its offspring are the infections it causes on or before day+14. "Above 5"
means offspring strictly greater than 5. "At least 20" means offspring >= 20.
Negative-binomial shares use that arm's own mean and method-of-moments k
(sample variance, divisor n-1). Intervals are 90% replicate bootstraps
(1,000 resamples, 5th and 95th percentiles) unless a column is named as a
Poisson band.
"""

import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "superspreader"
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE))

from analyze import load_daily  # noqa: E402
from analyze_round2 import log_nbinom_pmf, mom_k, p_total_gt, simulate_sizes  # noqa: E402
from analyze_round4 import TYPE_NAME  # noqa: E402

UNKNOWN = 4294967295
STEPS = 144
BOOT = 1000
BOOT_SEED = 20261015
SIM_SEED = 20261015
SIM_CHAINS = 100_000
SIM_CAP = 5000

TYPE_NAMES = dict(TYPE_NAME)
CHAIN_ARMS = [
    ("old_map_shape0.03", RESULTS / "x21" / "x3" / "shape0.03"),
    ("new_map_hospitals_unsplit", RESULTS / "v25" / "v4" / "shape0.03"),
    ("hospitals_split", RESULTS / "round2" / "r34" / "shape0.03"),
    ("w3_hospitals_cinemas_churches", RESULTS / "round2" / "w3" / "shape0.03"),
    ("w10_repaired_and_split", RESULTS / "round2" / "w10" / "tail"),
    ("w8_home035", RESULTS / "round2" / "w8" / "home035"),
]
EVENT_ARMS = [
    ("r36_baseline", RESULTS / "round2" / "r36" / "baseline", False),
    ("a_uncapped_f1200", RESULTS / "round2" / "r36" / "a_r0.003_f1200", True),
    ("cap100_f1091", RESULTS / "round2" / "r42" / "cap100_f1091", True),
    ("cap100_f2500", RESULTS / "round2" / "r42" / "cap100_f2500", True),
    ("cap100_f600", RESULTS / "round2" / "r42" / "cap100_f600", True),
    ("E", RESULTS / "round2" / "r3cal" / "r35_E" / "round0", False),
]
PLACE_FILES = {
    "r32_c_locations3": ROOT / "inputRealExample" / "locations3.json",
    "r34_wards": HERE / "inputs" / "v25_wards" / "locations_wards.json",
}
INFECTION_ARMS = [
    ("r32_c_locations3_agents4", RESULTS / "round2" / "r32" / "c", "r32_c_locations3"),
    ("r34_hospitals_split", RESULTS / "round2" / "r34" / "shape0.03", "r34_wards"),
]
PS1_ARMS = [
    ("A0", "A0"),
    ("E", "E"),
    ("P0", "P0"),
    ("P25", "P25"),
    ("P50", "P50"),
    ("P100", "P100"),
    ("P25k", "P25k"),
    ("P100ln", "P100ln"),
    ("P50 wide", "P50a005"),
    ("P100 wide", "P100a005"),
]
BODY_BINS = ((1, 1), (2, 2), (3, 4), (5, 9), (10, 19), (20, 49))
SHAPE003 = {
    "old_map_shape0.03",
    "new_map_hospitals_unsplit",
    "hospitals_split",
    "w3_hospitals_cinemas_churches",
    "w10_repaired_and_split",
}


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path.name} ({path.stat().st_size} bytes)", flush=True)


def quantile_interval(samples):
    values = np.asarray([v for v in samples if v is not None and np.isfinite(v)], dtype=float)
    if values.size == 0:
        return None, None
    return float(np.quantile(values, 0.05)), float(np.quantile(values, 0.95))


def bootstrap(rep_payloads, statistic, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    n = len(rep_payloads)
    samples = []
    for _ in range(BOOT):
        draw = rng.integers(0, n, size=n)
        samples.append(statistic([rep_payloads[i] for i in draw]))
    return quantile_interval(samples)


def count_people(path):
    data = Path(path).read_bytes()
    ids = [int(item) for item in re.findall(br'"typeID"\s*:\s*(\d+)\s*,\s*"locations"', data)]
    counts = Counter(ids)
    return {
        "file": Path(path).name,
        "people": len(ids),
        "type9_tourists": counts.get(9, 0),
        "loaded_when_tourists_disabled": len(ids) - counts.get(9, 0),
        "by_type": {str(key): counts[key] for key in sorted(counts)},
    }


def histogram(values):
    counts = Counter(int(v) for v in values)
    return {count: counts[count] for count in range(0, (max(counts) if counts else 0) + 1)}


def nb_loglik(hist, mu, k):
    if mu <= 0 or k <= 0:
        return -1e300
    p = k / (k + mu)
    log_p = math.log(p)
    log_q = math.log(1.0 - p)
    log_k = math.lgamma(k)
    total = 0.0
    for offspring, number in hist.items():
        if number:
            total += number * (
                math.lgamma(offspring + k) - log_k - math.lgamma(offspring + 1) + k * log_p + offspring * log_q
            )
    return total


def ml_k(hist, mu):
    best_k, best = None, -1e300
    for value in np.geomspace(1e-4, 50.0, 500):
        score = nb_loglik(hist, mu, float(value))
        if score > best:
            best, best_k = score, float(value)
    lo, hi = best_k / 1.2, best_k * 1.2
    for _ in range(60):
        a = math.exp((2 * math.log(lo) + math.log(hi)) / 3)
        b = math.exp((math.log(lo) + 2 * math.log(hi)) / 3)
        if nb_loglik(hist, mu, a) < nb_loglik(hist, mu, b):
            lo = a
        else:
            hi = b
    k = math.exp(0.5 * (math.log(lo) + math.log(hi)))
    return k, nb_loglik(hist, mu, k)


def nb_pmf(mu, k, j):
    p = k / (k + mu)
    return math.exp(log_nbinom_pmf(j, k, p))


def nb_sf_at_least(mu, k, threshold):
    """P(X >= threshold) for the negative binomial with this mean and k."""
    if threshold <= 0:
        return 1.0
    if mu <= 0 or k <= 0 or not math.isfinite(k):
        return None
    p = k / (k + mu)
    log_q = math.log(1.0 - p)
    logpmf = log_nbinom_pmf(threshold, k, p)
    logs = [logpmf]
    j = threshold
    while j < threshold + 200000:
        j += 1
        logpmf += math.log(j - 1 + k) - math.log(j) + log_q
        logs.append(logpmf)
        if logpmf < logs[0] - 45 and logpmf < max(logs) - 45:
            break
    peak = max(logs)
    return math.exp(peak) * sum(math.exp(item - peak) for item in logs)


def nb_shares(mu, k):
    if mu is None or k is None or not math.isfinite(k) or mu < 0 or k <= 0:
        return {name: None for name in ("share_0", "share_1_to_5", "share_above_5", "share_at_least_20", "share_at_least_50", "share_at_least_100", "pmf_5")}
    p0 = nb_pmf(mu, k, 0)
    mid = sum(nb_pmf(mu, k, j) for j in range(1, 6))
    above5 = nb_sf_at_least(mu, k, 6)
    check = p0 + mid + above5
    if abs(check - 1.0) > 1e-6:
        raise RuntimeError(f"negative-binomial probabilities sum to {check} at mean {mu} k {k}")
    return {
        "share_0": p0,
        "share_1_to_5": mid,
        "share_above_5": above5,
        "share_at_least_20": nb_sf_at_least(mu, k, 20),
        "share_at_least_50": nb_sf_at_least(mu, k, 50),
        "share_at_least_100": nb_sf_at_least(mu, k, 100),
        "pmf_5": nb_pmf(mu, k, 5),
    }


def poisson_interval(lam, alpha=0.05):
    """Central probability interval of a Poisson count, as integer endpoints."""
    if lam <= 0:
        return 0, 0
    lo_p, hi_p = alpha / 2.0, 1.0 - alpha / 2.0
    if lam < 200:
        term = math.exp(-lam)
        cdf = term
        k = 0
        low = 0 if cdf >= lo_p else None
        while True:
            if low is None and cdf >= lo_p:
                low = k
            if cdf >= hi_p:
                return (0 if low is None else low), k
            k += 1
            term *= lam / k
            cdf += term
            if k > lam + 80 + 30 * math.sqrt(lam):
                return (0 if low is None else low), k
    sd = math.sqrt(lam)
    start = max(0, int(lam - 12 * sd) - 1)
    logpmf = -lam + (start * math.log(lam) if start else 0.0) - math.lgamma(start + 1)
    # Left tail below `start` is negligible at 12 sd, so the cdf starts there.
    left = 0.0
    cdf = left
    low = None
    high = start
    k = start
    while k < lam + 14 * sd + 5:
        cdf += math.exp(logpmf)
        if low is None and cdf >= lo_p:
            low = k
        if cdf >= hi_p:
            high = k
            break
        k += 1
        logpmf += math.log(lam) - math.log(k)
        high = k
    return (start if low is None else low), high


def load_events(path):
    found = set()
    with open(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        index = {name: i for i, name in enumerate(header)}
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            found.add((int(parts[index["location_index"]]), int(parts[index["day"]])))
    return found


def index_offspring(trace_path, events):
    day = {}
    location = {}
    known = set()
    children = defaultdict(list)
    with open(trace_path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        index = {name: i for i, name in enumerate(header)}
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            agent = int(parts[index["agent"]])
            infector = int(parts[index["infector"]])
            d = int(parts[index["timestamp"]]) // STEPS
            loc = int(parts[index["location"]])
            day[agent] = d
            location[agent] = loc
            if infector != UNKNOWN:
                known.add(agent)
                children[infector].append((d, loc))
    total, event = [], []
    for agent, d in day.items():
        if agent not in known or not (1 <= d <= 10):
            continue
        n = e = 0
        limit = d + 14
        for child_day, child_loc in children.get(agent, ()):
            if child_day <= limit:
                n += 1
                if (child_loc, child_day) in events:
                    e += 1
        total.append(n)
        event.append(e)
    return np.asarray(total, dtype=np.int32), np.asarray(event, dtype=np.int32)


def trace_forest(path):
    meta = {}
    with open(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        index = {name: i for i, name in enumerate(header)}
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            infector = int(parts[index["infector"]])
            agent = int(parts[index["agent"]])
            meta[agent] = None if infector == UNKNOWN else infector
    children = defaultdict(list)
    roots = []
    for agent, parent in meta.items():
        if parent is not None and parent in meta:
            children[parent].append(agent)
        else:
            roots.append(agent)
    offspring = {agent: len(children.get(agent, ())) for agent in meta}
    generation = {root: 1 for root in roots}
    stack = list(roots)
    seen = set(roots)
    while stack:
        node = stack.pop()
        nxt = generation[node] + 1
        for child in children.get(node, ()):
            if child in seen:
                continue
            seen.add(child)
            generation[child] = nxt
            stack.append(child)
    sizes = []
    for root in roots:
        stack = [root]
        in_chain = set()
        size = 0
        while stack:
            node = stack.pop()
            if node in in_chain:
                continue
            in_chain.add(node)
            size += 1
            stack.extend(children.get(node, ()))
        sizes.append(size)
    parent_off, child_mean = [], []
    for agent, kids in children.items():
        # A component that never meets an introduction is a cycle in the trace.
        # It is not a generation, so it stays out of the parent-child correlation.
        if agent not in generation or not kids:
            continue
        forest_kids = [child for child in kids if child in generation]
        if not forest_kids:
            continue
        parent_off.append(offspring[agent])
        child_mean.append(float(np.mean([offspring[child] for child in forest_kids])))
    by_gen = defaultdict(list)
    for agent, gen in generation.items():
        by_gen[gen].append(offspring[agent])
    return {
        "offspring": np.asarray(list(offspring.values()), dtype=np.int32),
        "sizes": np.asarray(sizes, dtype=np.int32),
        "parent_off": np.asarray(parent_off, dtype=float),
        "child_mean": np.asarray(child_mean, dtype=float),
        "by_gen": {gen: np.asarray(values, dtype=np.int32) for gen, values in by_gen.items()},
        "covered": len(generation),
        "nodes": len(meta),
        "unreached": len(meta) - len(generation),
    }


def average_ranks(values):
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=float)
    ordered = arr[order]
    start = 0
    while start < arr.size:
        stop = start
        while stop + 1 < arr.size and ordered[stop + 1] == ordered[start]:
            stop += 1
        ranks[order[start:stop + 1]] = 0.5 * ((start + 1) + (stop + 1))
        start = stop + 1
    return ranks


def spearman(x, y):
    if len(x) < 3:
        return None
    rx = average_ranks(x)
    ry = average_ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float(rx @ rx) * float(ry @ ry))
    if denom == 0:
        return None
    return float(rx @ ry) / denom


def load_places(path):
    places = json.loads(Path(path).read_text())["places"]
    ids, types, caps = [], [], []
    seen = {}
    duplicate_ids = 0
    for place in places:
        ident = place.get("ID", "")
        if ident in seen:
            duplicate_ids += 1
        else:
            seen[ident] = len(ids)
        ids.append(ident)
        types.append(int(place.get("type", -1)))
        caps.append(float(place.get("capacity") or 0))
    return {"id": ids, "type": np.asarray(types), "capacity": np.asarray(caps), "duplicate_ids": duplicate_ids}


def infection_counts(trace_path):
    early, whole = Counter(), Counter()
    with open(trace_path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        index = {name: i for i, name in enumerate(header)}
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            day = int(parts[index["timestamp"]]) // STEPS
            loc = int(parts[index["location"]])
            loc_type = int(parts[index["locType"]])
            whole[(loc, loc_type)] += 1
            if 1 <= day <= 28:
                early[(loc, loc_type)] += 1
    return early, whole


def command_flags(directory):
    path = next(Path(directory).glob("rep*/command.json"), None)
    if path is None:
        return {}
    args = json.loads(path.read_text())
    wanted = {}
    keys = [
        "-a", "-l", "-w", "-k", "--sheddingShape", "--eventRate", "--eventFactor",
        "--eventMaxCapacity", "--eventTypes", "--homeVenueFactor", "--disableTourists",
    ]
    for key in keys:
        if key in args:
            wanted[key] = args[args.index(key) + 1]
    return wanted


def r50():
    print("R5-0", flush=True)
    files = {}
    for name in ("agents2.json", "agents4.json", "agents1.json"):
        files[name] = count_people(ROOT / "inputRealExample" / name)
        print(f"  {name} {files[name]['people']} tourists {files[name]['type9_tourists']}", flush=True)
    hist = defaultdict(int)
    total_n = 0
    with (OUT / "35_r45_offspring_hist.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["arm"] != "E":
                continue
            hist[int(row["offspring"])] = int(row["index_cases"])
            total_n += int(row["index_cases"])
    mean = sum(j * n for j, n in hist.items()) / total_n
    var = sum(n * (j - mean) ** 2 for j, n in hist.items()) / (total_n - 1)
    k_mom = mean * mean / (var - mean)
    k_ml, ll = ml_k(hist, mean)
    published = nb_shares(1.39, 0.31)
    out = {
        "files": files,
        "active_definition": (
            "typeID 9 is named tourist in inputConfigFiles/agentTypes.json. "
            "--disableTourists defaults to 1 and those agents are skipped in initAgents. "
            "The study commands do not set the flag, so the loaded population is the file count minus type 9."
        ),
        "E_histogram_source": "35_r45_offspring_hist.csv",
        "E_index_cases": total_n,
        "E_mean": mean,
        "E_mom_k_ddof1": k_mom,
        "E_ml_k_mean_fixed": k_ml,
        "E_ml_loglik": ll,
        "published_tail_check_mean_1.39_k_0.31": published,
        "five_seed": {
            "note22_figure": 0.0077,
            "note22_configuration": "old map, flat shedding shape 0.03, 40 replicates, 38545 chains, mean 0.363, k 0.034, longest 254",
            "later_file_order_groups": 0.009469451290699182,
            "later_exact_dwass": 0.009438107399729612,
            "definition": "P(total size of 5 introductions > 50). The later figure groups the stored chains in file order, non-overlapping, and discards the leftover chains.",
            "exact_recomputed": p_total_gt(0.3634502006506697, 0.03351811092511302, 5, 50),
        },
        "flags": {
            "eventMaxCapacity": {
                "commit": "c51b3ad",
                "date": "2026-10-15",
                "default": 0,
                "meaning": "If positive, a location is event-eligible only when its type is allowed and its capacity is at or below this value. Zero leaves eligibility unchanged.",
            },
            "homeVenueFactor_flag": {
                "commit": "0184e68",
                "date": "2026-10-14",
                "default": -1,
                "meaning": "If non-negative, replace the venue-profile base of type 2 before venueStrength is applied.",
            },
            "homeVenueFactor_guard": {
                "commit": "c51b3ad",
                "date": "2026-10-15",
                "meaning": "The factor is applied when the venue profile is on, or communityScale is not 1, or homeVenueFactor is non-negative. Before this commit an explicit home factor was ignored while the profile was off.",
            },
        },
    }
    print(f"  E MoM k {k_mom:.6f} ML k {k_ml:.6f} tail50 {published['share_at_least_50']}", flush=True)
    return out


def share_pack(total, event):
    n = int(total.size)
    if n == 0:
        return None
    mean = float(total.mean())
    k = mom_k(total)
    def share(mask):
        return float(np.mean(mask))
    event_children = int(event.sum())
    any_event = event >= 1
    return {
        "index_cases": n,
        "mean_offspring": mean,
        "early_k": None if k is None else float(k),
        "early_R": mean,
        "share_0": share(total == 0),
        "share_1_to_5": share((total >= 1) & (total <= 5)),
        "share_above_5": share(total > 5),
        "share_exactly_5": share(total == 5),
        "share_at_least_20": share(total >= 20),
        "share_at_least_50": share(total >= 50),
        "share_at_least_100": share(total >= 100),
        "maximum": int(total.max()),
        "event_offspring_share": (event_children / float(total.sum())) if total.sum() else None,
        "event_offspring_mean_if_any": float(event[any_event].mean()) if any_event.any() else None,
        "n_with_event_offspring": int(any_event.sum()),
    }


def r51():
    print("R5-1", flush=True)
    joint_rows = []
    summary_rows = []
    details = {}
    reference = {}
    with (OUT / "35_r45_offspring_hist.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["arm"] == "E":
                reference[int(row["offspring"])] = int(row["index_cases"])
    for label, directory, has_events in EVENT_ARMS:
        print(f"  {label}", flush=True)
        reps = []
        pooled_t, pooled_e = [], []
        for trace in sorted(directory.glob("rep*/trace.tsv")):
            events = set()
            if has_events:
                event_path = trace.parent.parent / f"{trace.parent.name}.events.tsv"
                events = load_events(event_path)
            total, event = index_offspring(trace, events)
            if np.any(event > total):
                raise RuntimeError(f"event offspring exceeded total in {trace}")
            reps.append((total, event))
            pooled_t.append(total)
            pooled_e.append(event)
        total = np.concatenate(pooled_t)
        event = np.concatenate(pooled_e)
        counts = Counter(zip(total.tolist(), event.tolist()))
        for (off, ev), number in sorted(counts.items()):
            joint_rows.append({
                "arm": label,
                "offspring_total": off,
                "offspring_at_events": ev,
                "index_cases": number,
            })
        point = share_pack(total, event)
        expected = nb_shares(point["mean_offspring"], point["early_k"])

        def stat(parts, key):
            arrays_t = [item[0] for item in parts]
            arrays_e = [item[1] for item in parts]
            return share_pack(np.concatenate(arrays_t), np.concatenate(arrays_e))[key]

        intervals = {}
        for key in (
            "mean_offspring", "early_k", "share_0", "share_1_to_5", "share_above_5",
            "share_exactly_5", "share_at_least_20", "share_at_least_50", "share_at_least_100",
            "event_offspring_share", "event_offspring_mean_if_any",
        ):
            low, high = bootstrap(reps, lambda parts, key=key: stat(parts, key), seed=BOOT_SEED + len(label))
            intervals[key] = (low, high)
        row = {
            "arm": label,
            "index_cases": point["index_cases"],
            "mean_offspring": point["mean_offspring"],
            "mean_offspring_low": intervals["mean_offspring"][0],
            "mean_offspring_high": intervals["mean_offspring"][1],
            "share_0": point["share_0"],
            "share_0_low": intervals["share_0"][0],
            "share_0_high": intervals["share_0"][1],
            "share_1_to_5": point["share_1_to_5"],
            "share_1_to_5_low": intervals["share_1_to_5"][0],
            "share_1_to_5_high": intervals["share_1_to_5"][1],
            "share_above_5": point["share_above_5"],
            "share_above_5_low": intervals["share_above_5"][0],
            "share_above_5_high": intervals["share_above_5"][1],
            "share_at_least_20": point["share_at_least_20"],
            "share_at_least_20_low": intervals["share_at_least_20"][0],
            "share_at_least_20_high": intervals["share_at_least_20"][1],
            "share_at_least_50": point["share_at_least_50"],
            "share_at_least_50_low": intervals["share_at_least_50"][0],
            "share_at_least_50_high": intervals["share_at_least_50"][1],
            "share_at_least_100": point["share_at_least_100"],
            "share_at_least_100_low": intervals["share_at_least_100"][0],
            "share_at_least_100_high": intervals["share_at_least_100"][1],
            "maximum": point["maximum"],
            "event_offspring_share": point["event_offspring_share"],
            "event_offspring_share_low": intervals["event_offspring_share"][0],
            "event_offspring_share_high": intervals["event_offspring_share"][1],
            "early_k": point["early_k"],
            "early_k_low": intervals["early_k"][0],
            "early_k_high": intervals["early_k"][1],
            "nb_share_0": expected["share_0"],
            "nb_share_1_to_5": expected["share_1_to_5"],
            "nb_share_above_5": expected["share_above_5"],
            "nb_share_at_least_20": expected["share_at_least_20"],
            "nb_share_at_least_50": expected["share_at_least_50"],
            "nb_share_at_least_100": expected["share_at_least_100"],
            "nb_pmf_5": expected["pmf_5"],
            "share_exactly_5": point["share_exactly_5"],
            "n_with_event_offspring": point["n_with_event_offspring"],
            "event_offspring_mean_if_any": point["event_offspring_mean_if_any"],
            "event_offspring_mean_if_any_low": intervals["event_offspring_mean_if_any"][0],
            "event_offspring_mean_if_any_high": intervals["event_offspring_mean_if_any"][1],
            "replicates": len(reps),
            "command": command_flags(directory),
        }
        if expected["share_at_least_50"]:
            row["ratio_at_least_50_to_nb"] = point["share_at_least_50"] / expected["share_at_least_50"]
        if expected["pmf_5"]:
            row["ratio_exactly_5_to_nb"] = point["share_exactly_5"] / expected["pmf_5"]
        if expected["share_above_5"]:
            row["ratio_above_5_to_nb"] = point["share_above_5"] / expected["share_above_5"]
        if expected["share_at_least_20"]:
            row["ratio_at_least_20_to_nb"] = point["share_at_least_20"] / expected["share_at_least_20"]
        summary_rows.append(row)
        details[label] = {
            "index_cases": point["index_cases"],
            "mean": point["mean_offspring"],
            "k": point["early_k"],
            "maximum": point["maximum"],
            "share_0": point["share_0"],
            "share_1_to_5": point["share_1_to_5"],
            "share_above_5": point["share_above_5"],
            "share_exactly_5": point["share_exactly_5"],
            "share_at_least_20": point["share_at_least_20"],
            "share_at_least_50": point["share_at_least_50"],
            "share_at_least_100": point["share_at_least_100"],
            "nb": expected,
            "ratio_at_least_50": row.get("ratio_at_least_50_to_nb"),
            "ratio_exactly_5": row.get("ratio_exactly_5_to_nb"),
            "ratio_above_5": row.get("ratio_above_5_to_nb"),
            "ratio_at_least_20": row.get("ratio_at_least_20_to_nb"),
            "event_share": point["event_offspring_share"],
            "event_mean_if_any": point["event_offspring_mean_if_any"],
            "event_mean_interval": intervals["event_offspring_mean_if_any"],
            "n_with_event_offspring": point["n_with_event_offspring"],
            "share_at_least_50_interval": intervals["share_at_least_50"],
            "command": row["command"],
        }
        print(
            f"    n {point['index_cases']} R {point['mean_offspring']:.3f} k {point['early_k']:.3f} "
            f">5 {point['share_above_5']:.4f} >=50 {point['share_at_least_50']:.6g} "
            f"event mean {point['event_offspring_mean_if_any']}",
            flush=True,
        )
        if label == "E":
            got = histogram(total)
            if got != reference:
                raise RuntimeError("E offspring histogram does not match 35_r45_offspring_hist.csv")
            per_k = [mom_k(item[0]) for item in reps]
            details[label]["mean_of_replicate_k"] = float(np.mean(per_k))
    write_csv(OUT / "38_r51_event_offspring.csv", joint_rows)
    write_csv(OUT / "38_r51_summary.csv", summary_rows)
    return details


def bin_count(sizes, lo, hi):
    return int(np.sum((sizes >= lo) & (sizes <= hi)))


def r53():
    print("R5-3", flush=True)
    hist_rows = []
    iid_rows = []
    corr_rows = []
    details = {}
    arms = list(CHAIN_ARMS) + [
        ("v5_caregiving_check", RESULTS / "v25" / "v5" / "caregiving_shed0.09"),
    ]
    for arm_index, (label, directory) in enumerate(arms):
        print(f"  {label}", flush=True)
        reps = []
        for trace in sorted(directory.glob("rep*/trace.tsv")):
            forest = trace_forest(trace)
            if int(forest["sizes"].sum()) != forest["covered"]:
                raise RuntimeError(f"a node sits in more than one chain in {trace}")
            reps.append(forest)
        offspring = np.concatenate([rep["offspring"] for rep in reps])
        sizes = np.concatenate([rep["sizes"] for rep in reps])
        counts = Counter(int(v) for v in offspring)
        maximum = max(counts) if counts else 0
        if label != "v5_caregiving_check":
            for value in range(0, maximum + 1):
                hist_rows.append({"arm": label, "offspring": value, "infectors": counts.get(value, 0)})
        rng = np.random.default_rng(SIM_SEED + arm_index)
        drawn = simulate_sizes(offspring, SIM_CHAINS, SIM_CAP, rng)
        n_obs = int(sizes.size)
        scale = n_obs / SIM_CHAINS
        observed_counts = Counter(int(v) for v in sizes)
        sim_counts = Counter(int(v) for v in drawn)
        longest = int(sizes.max())
        capped = int(np.sum(drawn >= SIM_CAP))

        def scaled(count):
            return count * scale

        if label != "v5_caregiving_check":
            for size in range(1, 201):
                expected = scaled(sim_counts.get(size, 0))
                low, high = poisson_interval(expected)
                iid_rows.append({
                    "arm": label,
                    "size": size,
                    "observed": observed_counts.get(size, 0),
                    "iid_expected": expected,
                    "iid_low": low,
                    "iid_high": high,
                })
            observed_tail = sum(number for size, number in observed_counts.items() if size > 200)
            expected_tail = scaled(sum(number for size, number in sim_counts.items() if size > 200))
            low, high = poisson_interval(expected_tail)
            iid_rows.append({
                "arm": label,
                "size": -1,
                "observed": observed_tail,
                "iid_expected": expected_tail,
                "iid_low": low,
                "iid_high": high,
            })
        bins = []
        for lo, hi in BODY_BINS:
            observed = bin_count(sizes, lo, hi)
            expected = scaled(bin_count(drawn, lo, hi))
            band_lo, band_hi = poisson_interval(expected)
            bins.append({
                "bin": f"{lo}-{hi}" if lo != hi else str(lo),
                "observed": observed,
                "iid_expected": expected,
                "iid_low": band_lo,
                "iid_high": band_hi,
                "inside": band_lo <= observed <= band_hi,
                "ratio": (observed / expected) if expected else None,
            })
        above10_obs = int(np.sum(sizes > 10))
        above10_iid = scaled(int(np.sum(drawn > 10)))
        parent_x = [rep["parent_off"] for rep in reps]
        parent_y = [rep["child_mean"] for rep in reps]
        spear = spearman(np.concatenate(parent_x), np.concatenate(parent_y))
        spear_lo, spear_hi = bootstrap(
            list(zip(parent_x, parent_y)),
            lambda parts: spearman(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])),
            seed=BOOT_SEED + 100 + arm_index,
        )
        gen_rows = []
        for gen in range(1, 9):
            arrays = [rep["by_gen"].get(gen, np.asarray([], dtype=np.int32)) for rep in reps]
            pooled = np.concatenate(arrays) if any(len(item) for item in arrays) else np.asarray([])
            mean = float(pooled.mean()) if pooled.size else None

            def gen_mean(parts, gen=gen):
                chosen = [part["by_gen"].get(gen, np.asarray([], dtype=np.int32)) for part in parts]
                if not any(len(item) for item in chosen):
                    return None
                return float(np.concatenate(chosen).mean())

            glo, ghi = bootstrap(reps, gen_mean, seed=BOOT_SEED + 200 + 10 * arm_index + gen)
            record = {
                "arm": label,
                "generation": gen,
                "n_nodes": int(pooled.size),
                "mean_offspring": mean,
                "mean_offspring_low": glo,
                "mean_offspring_high": ghi,
                "n_parents_with_children": int(sum(len(item) for item in parent_x)),
                "spearman": spear,
                "spearman_low": spear_lo,
                "spearman_high": spear_hi,
            }
            gen_rows.append(record)
            if label != "v5_caregiving_check":
                corr_rows.append(record)
        later = []
        for gen in range(9, 30):
            n = sum(len(rep["by_gen"].get(gen, ())) for rep in reps)
            if n:
                later.append({"generation": gen, "n_nodes": n})
        groups = [sizes[i:i + 5] for i in range(0, n_obs - (n_obs % 5), 5)]
        five = float(np.mean([group.sum() > 50 for group in groups])) if groups else None
        details[label] = {
            "chains": n_obs,
            "nodes": int(offspring.size),
            "unreached_cycle_nodes": int(sum(rep["unreached"] for rep in reps)),
            "mean_offspring": float(offspring.mean()),
            "k": float(mom_k(offspring)),
            "longest": longest,
            "sim_hit_cap": capped,
            "bins": bins,
            "above_10_observed": above10_obs,
            "above_10_iid": above10_iid,
            "spearman": spear,
            "spearman_interval": [spear_lo, spear_hi],
            "n_parents": int(sum(len(item) for item in parent_x)),
            "generations": [
                {"generation": row["generation"], "n": row["n_nodes"], "mean": row["mean_offspring"]}
                for row in gen_rows
            ],
            "generations_after_8": later,
            "five_seed_file_order": five,
            "five_seed_groups": len(groups),
        }
        print(
            f"    chains {n_obs} R {float(offspring.mean()):.4f} spearman {spear} "
            f"above10 {above10_obs} iid {above10_iid:.1f} five-seed {five}",
            flush=True,
        )
        for item in bins:
            print(f"      bin {item['bin']}: obs {item['observed']} iid {item['iid_expected']:.1f} inside {item['inside']} ratio {item['ratio']}", flush=True)
    write_csv(OUT / "38_r53_offspring_hist_all.csv", hist_rows)
    write_csv(OUT / "38_r53_iid_vs_observed.csv", iid_rows)
    write_csv(OUT / "38_r53_correlation.csv", corr_rows)
    return details


def type_maxima(places):
    by_type = defaultdict(list)
    for cap, loc_type in zip(places["capacity"], places["type"]):
        by_type[int(loc_type)].append(float(cap))
    return {loc_type: max(values) if values else None for loc_type, values in sorted(by_type.items())}


def r54():
    print("R5-4", flush=True)
    places = {key: load_places(path) for key, path in PLACE_FILES.items()}
    maxima = {key: type_maxima(value) for key, value in places.items()}
    print("  type maxima locations3", {t: maxima["r32_c_locations3"][t] for t in sorted(maxima["r32_c_locations3"]) if maxima["r32_c_locations3"][t] >= 1000}, flush=True)
    type_rows = []
    top_rows = []
    details = {"type_max_at_least_1000": {}, "arms": {}}
    for key, mapping in maxima.items():
        details["type_max_at_least_1000"][key] = sorted(t for t, cap in mapping.items() if cap is not None and cap >= 1000)
    all_types = sorted(TYPE_NAMES)
    for label, directory, place_key in INFECTION_ARMS:
        print(f"  {label}", flush=True)
        book = places[place_key]
        early_reps, whole_reps = [], []
        for trace in sorted(directory.glob("rep*/trace.tsv")):
            early, whole = infection_counts(trace)
            early_reps.append(early)
            whole_reps.append(whole)
        arm_detail = {"command": command_flags(directory), "windows": {}}
        n_places = len(book["id"])
        big_types = set(details["type_max_at_least_1000"][place_key])
        selectors = {
            "home_school_work": lambda typ, cap: typ in (2, 3, 33, 4, 13),
            "note_list_whole_6_14_15_44_and_large_7_9": lambda typ, cap: typ in (6, 14, 15, 44) or (typ in (7, 9) and cap >= 1000),
            "locations_capacity_at_least_1000": lambda typ, cap: cap >= 1000,
            "types_whose_max_is_at_least_1000": lambda typ, cap: typ in big_types,
            "type_7_capacity_at_least_1000": lambda typ, cap: typ == 7 and cap >= 1000,
            "type_7_all": lambda typ, cap: typ == 7,
            "type_15": lambda typ, cap: typ == 15,
            "type_6": lambda typ, cap: typ == 6,
            "type_9_capacity_at_least_1000": lambda typ, cap: typ == 9 and cap >= 1000,
        }
        for window, reps in (("days_1_28", early_reps), ("whole_run", whole_reps)):
            n_reps = len(reps)
            rep_total = np.zeros(n_reps, dtype=float)
            type_rep = defaultdict(lambda: np.zeros(n_reps, dtype=float))
            loc_rep = defaultdict(lambda: np.zeros(n_reps, dtype=float))
            group_rep = {name: np.zeros(n_reps, dtype=float) for name in selectors}
            unknown = 0
            disagreements = 0
            for rep_i, counter in enumerate(reps):
                for (loc, loc_type), number in counter.items():
                    rep_total[rep_i] += number
                    type_rep[loc_type][rep_i] += number
                    loc_rep[(loc, loc_type)][rep_i] += number
                    if not (0 <= loc < n_places):
                        unknown += number
                        cap = 0.0
                    else:
                        cap = float(book["capacity"][loc])
                        if int(book["type"][loc]) != loc_type:
                            disagreements += number
                    for name, selector in selectors.items():
                        if selector(loc_type, cap):
                            group_rep[name][rep_i] += number
            total = int(rep_total.sum())
            rng = np.random.default_rng(BOOT_SEED + (0 if window == "days_1_28" else 1))
            picked = rng.integers(0, n_reps, size=(BOOT, n_reps))
            weights = np.zeros((BOOT, n_reps), dtype=float)
            for rep_i in range(n_reps):
                weights[:, rep_i] = np.sum(picked == rep_i, axis=1)
            boot_total = weights @ rep_total
            present_types = set(all_types) | set(type_rep)
            pooled_locs = Counter()
            for (loc, loc_type), counts in loc_rep.items():
                pooled_locs[(loc, loc_type)] = int(counts.sum())
            by_type_counts = defaultdict(list)
            for (loc, loc_type), number in pooled_locs.items():
                by_type_counts[loc_type].append(number)
            for loc_type in sorted(present_types):
                infections = int(type_rep[loc_type].sum()) if loc_type in type_rep else 0
                loc_numbers = by_type_counts.get(loc_type, [])
                distinct = len(loc_numbers)
                largest = max(loc_numbers) if loc_numbers else 0
                largest_share = (largest / infections) if infections else None
                type_share = (infections / total) if total else None
                if infections and np.any(boot_total > 0):
                    boot_type = weights @ type_rep[loc_type]
                    share_lo, share_hi = quantile_interval(boot_type / boot_total)
                    members = [key for key in loc_rep if key[1] == loc_type]
                    member_mat = np.vstack([loc_rep[key] for key in members])
                    boot_locs = weights @ member_mat.T
                    boot_largest = boot_locs.max(axis=1)
                    positive = boot_type > 0
                    large_lo, large_hi = quantile_interval(boot_largest[positive] / boot_type[positive])
                else:
                    share_lo = share_hi = large_lo = large_hi = None
                type_rows.append({
                    "arm": label,
                    "window": window,
                    "type": loc_type,
                    "type_name": TYPE_NAMES.get(loc_type, ""),
                    "infections": infections,
                    "locations_with_infection": distinct,
                    "largest_location_share": largest_share,
                    "largest_location_share_low": large_lo,
                    "largest_location_share_high": large_hi,
                    "type_share": type_share,
                    "type_share_low": share_lo,
                    "type_share_high": share_hi,
                    "capacity_max": maxima[place_key].get(loc_type),
                })
            ranked = []
            for (loc, loc_type), number in pooled_locs.items():
                if 0 <= loc < n_places:
                    ident = book["id"][loc]
                    cap = float(book["capacity"][loc])
                    file_type = int(book["type"][loc])
                else:
                    ident, cap, file_type = "", None, None
                ranked.append((number, loc, loc_type, ident, cap, file_type))
            ranked.sort(key=lambda item: (-item[0], item[1]))
            top = ranked[:20]
            nonhome = [item for item in ranked if item[2] != 2]
            watched = [(item[1], item[2]) for item in top]
            if nonhome:
                watched.append((nonhome[0][1], nonhome[0][2]))
            boot_share = {}
            for key in watched:
                boot_share[key] = (weights @ loc_rep[key]) / boot_total
            for rank, (number, loc, loc_type, ident, cap, file_type) in enumerate(top, start=1):
                lo, hi = quantile_interval(boot_share[(loc, loc_type)])
                top_rows.append({
                    "arm": label,
                    "window": window,
                    "rank": rank,
                    "location_index": loc,
                    "id": ident,
                    "type": loc_type,
                    "type_name": TYPE_NAMES.get(loc_type, ""),
                    "file_type": file_type,
                    "capacity": cap,
                    "infections": number,
                    "share": number / total if total else None,
                    "share_low": lo,
                    "share_high": hi,
                })
            groups = {}
            for name, counts in group_rep.items():
                point = float(counts.sum() / total) if total else None
                lo, hi = quantile_interval((weights @ counts) / boot_total)
                groups[name] = {"share": point, "low": lo, "high": hi}
            if nonhome and total:
                nh_lo, nh_hi = quantile_interval(boot_share[(nonhome[0][1], nonhome[0][2])])
                nonhome_record = {
                    "id": nonhome[0][3],
                    "type": nonhome[0][2],
                    "capacity": nonhome[0][4],
                    "infections": nonhome[0][0],
                    "share": nonhome[0][0] / total,
                    "share_low": nh_lo,
                    "share_high": nh_hi,
                }
            else:
                nonhome_record = None
            arm_detail["windows"][window] = {
                "infections": total,
                "unknown_location_infections": unknown,
                "type_disagreements": disagreements,
                "groups": groups,
                "largest_nonresidence": nonhome_record,
            }
            print(f"    {window} infections {total} home/school/work {groups['home_school_work']['share']}", flush=True)
        details["arms"][label] = arm_detail
    write_csv(OUT / "38_r54_infections_by_type.csv", type_rows)
    write_csv(OUT / "38_r54_top_locations.csv", top_rows)
    details["duplicate_ids"] = {key: value["duplicate_ids"] for key, value in places.items()}
    return details


def r55():
    print("R5-5", flush=True)
    rows = []
    per_arm = {}
    for label, directory_name in PS1_ARMS:
        directory = RESULTS / "ps" / "ps1" / directory_name
        arm_rows = []
        for stdout in sorted(directory.glob("rep*/stdout.txt")):
            daily = load_daily(stdout)
            if not daily:
                raise RuntimeError(f"no daily table in {stdout}")
            def at(index):
                return int(daily[index]["INF"]) if len(daily) > index else None
            record = {
                "arm": label,
                "rep": stdout.parent.name,
                "directory": directory_name,
                "days": len(daily),
                "day28": at(27),
                "day70": at(69),
                "final": int(daily[-1]["INF"]),
            }
            arm_rows.append(record)
            rows.append({
                "arm": label,
                "rep": record["rep"],
                "day28": record["day28"],
                "day70": record["day70"],
                "final": record["final"],
            })
        per_arm[label] = arm_rows
        finals = [item["final"] for item in arm_rows]
        print(f"  {label} n {len(arm_rows)} days {arm_rows[0]['days']} final mean {float(np.mean(finals)):.1f}", flush=True)
    # Within-arm regression of final size on the day-28 count.
    labels = [label for label, _name in PS1_ARMS]
    day = np.asarray([item["day28"] for label in labels for item in per_arm[label]], dtype=float)
    final = np.asarray([item["final"] for label in labels for item in per_arm[label]], dtype=float)
    arm = np.asarray([label for label in labels for _item in per_arm[label]])

    def fit(day, final, arm):
        num = den = within = 0.0
        means_d, means_f = {}, {}
        for label in np.unique(arm):
            mask = arm == label
            means_d[label] = float(day[mask].mean())
            means_f[label] = float(final[mask].mean())
            xd = day[mask] - means_d[label]
            yf = final[mask] - means_f[label]
            num += float(xd @ yf)
            den += float(xd @ xd)
            within += float(yf @ yf)
        beta = num / den if den else None
        residual = within - (beta * num if beta is not None else 0.0)
        r2 = 1.0 - residual / within if within else None
        return beta, means_d, means_f, r2

    beta, means_d, means_f, r2 = fit(day, final, arm)
    target = means_d["E"]
    adjusted = {label: means_f[label] + beta * (target - means_d[label]) for label in means_d}

    def ratio_from(parts):
        # parts is a list of per-arm row lists, in PS1_ARMS order, already resampled inside statistic below
        day_b, final_b, arm_b = [], [], []
        for rows_b in parts:
            for item in rows_b:
                day_b.append(item["day28"])
                final_b.append(item["final"])
                arm_b.append(item["arm"])
        beta_b, means_d_b, means_f_b, _r2 = fit(np.asarray(day_b, float), np.asarray(final_b, float), np.asarray(arm_b))
        target_b = means_d_b["E"]
        adj_e = means_f_b["E"] + beta_b * (target_b - means_d_b["E"])
        adj_p = means_f_b["P25k"] + beta_b * (target_b - means_d_b["P25k"])
        return adj_p / adj_e if adj_e else None

    payloads = [per_arm[label] for label in labels]
    # Resample within arm: each payload is one arm's rows. bootstrap() resamples the list of arms, which is wrong.
    rng = np.random.default_rng(BOOT_SEED + 55)
    ratio_samples = []
    beta_samples = []
    for _ in range(BOOT):
        drawn = []
        for rows_arm in payloads:
            pick = rng.integers(0, len(rows_arm), size=len(rows_arm))
            drawn.append([rows_arm[i] for i in pick])
        ratio_samples.append(ratio_from(drawn))
        day_b, final_b, arm_b = [], [], []
        for rows_b in drawn:
            for item in rows_b:
                day_b.append(item["day28"])
                final_b.append(item["final"])
                arm_b.append(item["arm"])
        beta_b, _d, _f, _r = fit(np.asarray(day_b, float), np.asarray(final_b, float), np.asarray(arm_b))
        beta_samples.append(beta_b)
    ratio_lo, ratio_hi = quantile_interval(ratio_samples)
    beta_lo, beta_hi = quantile_interval(beta_samples)
    summary = {
        "mapping": {label: directory_name for label, directory_name in PS1_ARMS},
        "days": {label: sorted({item["days"] for item in per_arm[label]}) for label in labels},
        "final_equals_day70": all(item["final"] == item["day70"] for label in labels for item in per_arm[label]),
        "slope": beta,
        "slope_interval": [beta_lo, beta_hi],
        "within_r2": r2,
        "reference_day28": target,
        "reference": "E mean day-28 count",
        "arms": {},
        "adjusted_P25k_over_E": adjusted["P25k"] / adjusted["E"],
        "adjusted_ratio_interval": [ratio_lo, ratio_hi],
        "raw_P25k_over_E": means_f["P25k"] / means_f["E"],
    }
    for label in labels:
        summary["arms"][label] = {
            "n": len(per_arm[label]),
            "day28": means_d[label],
            "day70": float(np.mean([item["day70"] for item in per_arm[label]])),
            "final": means_f[label],
            "adjusted_final": adjusted[label],
            "adjusted_over_E": adjusted[label] / adjusted["E"],
        }
    write_csv(OUT / "38_r55_ps1_cumulative.csv", rows)
    print(
        f"  slope {beta:.3f} P25k/E raw {summary['raw_P25k_over_E']:.3f} "
        f"adjusted {summary['adjusted_P25k_over_E']:.3f} interval {ratio_lo:.3f}-{ratio_hi:.3f}",
        flush=True,
    )
    return summary


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    summary_path = RESULTS / "round2" / "r5_summary.json"
    summary = {}
    if summary_path.exists() and which != "all":
        summary = json.loads(summary_path.read_text())
    if which in ("all", "r50"):
        summary["r50"] = r50()
    if which in ("all", "r51"):
        summary["r51"] = r51()
    if which in ("all", "r53"):
        summary["r53"] = r53()
    if which in ("all", "r54"):
        summary["r54"] = r54()
    if which in ("all", "r55"):
        summary["r55"] = r55()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(json_ready(summary), indent=1) + "\n")
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
