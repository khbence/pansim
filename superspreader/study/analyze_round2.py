#!/usr/bin/env python3
"""Tier-1 analyses for note 28: W1, W2, W4a, W6, and the W10 presence-rate measurement.

No PanSim runs. Exact negative-binomial expectations use the Dwass total-progeny
formula. Chain thresholds are "above" (size > cap), stated in the outputs.
"""

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STUDY = HERE
RESULTS = STUDY / "results" / "round2"
V25 = STUDY / "results" / "v25"
X21 = STUDY / "results" / "x21"
OUT_DIR = ROOT / "superspreader"
UNKNOWN = 4294967295
STEPS = 144

TYPE_GROUP = {
    5: "restaurant",
    6: "cinema",
    7: "shop",
    8: "shop",
    9: "church",
    10: "park",
    4: "work",
    13: "work",
    3: "school",
    33: "school",
    44: "school",
    12: "hospital",
    2: "home",
}
AUDIT_TYPES = (5, 6, 7, 8, 9, 10, 12, 14, 22)


def log_nbinom_pmf(failures, successes, p):
    if failures < 0 or successes <= 0 or p <= 0 or p >= 1:
        return float("-inf")
    return (
        math.lgamma(failures + successes)
        - math.lgamma(successes)
        - math.lgamma(failures + 1)
        + successes * math.log(p)
        + failures * math.log(1.0 - p)
    )


def p_total_gt(mean_r, k, introductions, cap):
    """P(total chain size > cap) from `introductions` introductions."""
    if mean_r <= 0 or k <= 0:
        return 0.0
    p = k / (k + mean_r)
    logs = []
    for n in range(introductions, cap + 1):
        logs.append(math.log(introductions / n) + log_nbinom_pmf(n - introductions, n * k, p))
    m = max(logs)
    cdf = math.exp(m) * sum(math.exp(lp - m) for lp in logs)
    return max(0.0, 1.0 - cdf)


def expected_above(mean_r, k, n_chains, cap):
    return n_chains * p_total_gt(mean_r, k, 1, cap)


def p_any_at_least(mean_r, k, n_chains, threshold):
    """P(at least one chain has size >= threshold)."""
    p1 = p_total_gt(mean_r, k, 1, threshold - 1)
    if p1 <= 0:
        return 0.0
    if n_chains * p1 < 1e-8:
        return n_chains * p1
    return 1.0 - math.exp(n_chains * math.log1p(-p1))


def poisson_cdf_le(count, lam):
    if count < 0:
        return 0.0
    if lam <= 0:
        return 1.0
    term = math.exp(-lam)
    total = term
    for j in range(1, count + 1):
        term *= lam / j
        total += term
        if term < 1e-18 * total and j > lam:
            break
    return min(1.0, total)


def garwood_interval(count, alpha=0.05):
    """95% Garwood interval for a Poisson mean given an observed count."""
    if count < 0:
        return (None, None)
    upper_tail = alpha / 2

    def find(target_cdf, guess):
        lo, hi = 0.0, max(guess * 4 + 10.0, 1.0)
        while poisson_cdf_le(count if target_cdf < 0.5 else count - 1, hi) > target_cdf * 0.5 and hi < 1e9:
            # expand until the cdf at hi is on the correct side
            hi *= 2
            if hi > 1e7:
                break
        # lower bound of the mean: P(X <= count-1 | lam) = 1 - alpha/2
        # upper bound of the mean: P(X <= count | lam) = alpha/2
        return lo, hi

    if count == 0:
        low = 0.0
    else:
        # P(X <= count-1 | lam) decreases in lam. Want it equal to 1 - alpha/2.
        target = 1.0 - upper_tail
        lo, hi = 0.0, max(count * 2.0, 1.0)
        while poisson_cdf_le(count - 1, hi) > target:
            hi *= 2
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if poisson_cdf_le(count - 1, mid) > target:
                lo = mid
            else:
                hi = mid
        low = 0.5 * (lo + hi)
    target = upper_tail
    lo, hi = 0.0, max((count + 1) * 4.0, 1.0)
    while poisson_cdf_le(count, hi) > target:
        hi *= 2
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if poisson_cdf_le(count, mid) > target:
            lo = mid
        else:
            hi = mid
    high = 0.5 * (lo + hi)
    return low, high


def mom_k(values):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return None
    mean = float(values.mean())
    var = float(values.var(ddof=1))
    if var <= mean or mean <= 0:
        return math.inf if var <= mean else None
    return mean * mean / (var - mean)


def percentile(values, q):
    if not len(values):
        return None
    xs = np.sort(np.asarray(values, dtype=float))
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = min(len(xs) - 1, lo + 1)
    return float(xs[lo] * (1 - (pos - lo)) + xs[hi] * (pos - lo))


def load_arm(directory):
    """Every infected node, its offspring, and each introduction's chain."""
    offspring = []
    chains = []
    for path in sorted(Path(directory).glob("rep*/trace.tsv")):
        meta = {}
        with open(path) as handle:
            header = handle.readline().rstrip("\n").split("\t")
            index = {name: i for i, name in enumerate(header)}
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                infector = int(parts[index["infector"]])
                agent = int(parts[index["agent"]])
                meta[agent] = (
                    None if infector == UNKNOWN else infector,
                    int(parts[index["locType"]]),
                    int(parts[index["location"]]),
                )
        children = defaultdict(list)
        roots = []
        for agent, (parent, _type, _loc) in meta.items():
            if parent is not None and parent in meta:
                children[parent].append(agent)
            else:
                roots.append(agent)
        for agent in meta:
            offspring.append(len(children.get(agent, ())))
        for root in roots:
            types = Counter()
            groups = Counter()
            locs = Counter()
            by_gen = defaultdict(Counter)
            stack = [(root, 0)]
            size = 0
            depth = 0
            while stack:
                node, gen = stack.pop()
                size += 1
                depth = max(depth, gen)
                _parent, loc_type, loc = meta[node]
                types[loc_type] += 1
                groups[TYPE_GROUP.get(loc_type, "other")] += 1
                locs[loc] += 1
                by_gen[gen][loc_type] += 1
                for child in children.get(node, ()):
                    stack.append((child, gen + 1))
            chains.append({
                "size": size,
                "generations": depth + 1,
                "types": types,
                "groups": groups,
                "locations": locs,
                "by_gen": {str(g): dict(counts) for g, counts in by_gen.items()},
                "rep": path.parent.name,
            })
    return np.asarray(offspring, dtype=np.int32), chains


def simulate_sizes(dist, n_chains, cap, rng):
    sizes = np.ones(n_chains, dtype=np.int32)
    frontier = np.ones(n_chains, dtype=np.int32)
    active = np.ones(n_chains, dtype=bool)
    for _ in range(100000):
        if not active.any():
            break
        idx = np.flatnonzero(active)
        frontier_now = frontier[idx]
        total = int(frontier_now.sum())
        draws = dist[rng.integers(0, dist.size, size=total)]
        cuts = np.cumsum(frontier_now)
        starts = np.empty_like(cuts)
        starts[0] = 0
        starts[1:] = cuts[:-1]
        kids = np.add.reduceat(draws, starts)
        new_size = sizes[idx] + kids
        capped = new_size >= cap
        sizes[idx] = np.minimum(new_size, cap)
        frontier[idx] = np.where(capped, 0, kids)
        active[idx] = frontier[idx] > 0
    return sizes


def count_above(sizes, caps):
    return {f"above_{cap}": int(np.sum(sizes > cap)) for cap in caps}


def scale_counts(counts, n_sim, n_obs):
    return {key: value * n_obs / n_sim for key, value in counts.items()}


def w1():
    arms = {
        "v4_shape0.03": V25 / "v4" / "shape0.03",
        "v5_flat": V25 / "v5" / "flat_shed0.04",
        "v5_caregiving": V25 / "v5" / "caregiving_shed0.09",
        "x1_caregiving": X21 / "x1" / "caregiving_shed0.09",
    }
    caps = (10, 50, 100, 200, 500)
    rows = []
    detail = {}
    rng = np.random.default_rng(20261013)
    for label, directory in arms.items():
        print(f"W1 load {label}", flush=True)
        offspring, chains = load_arm(directory)
        sizes = np.asarray([chain["size"] for chain in chains], dtype=np.int32)
        n_obs = int(sizes.size)
        mean_r = float(offspring.mean())
        k = mom_k(offspring)
        observed = count_above(sizes, caps)
        observed["longest"] = int(sizes.max())
        observed["chains"] = n_obs
        bands = {key: garwood_interval(value) for key, value in observed.items() if key.startswith("above_")}
        nb = {}
        if k and math.isfinite(k):
            for cap in caps:
                nb[f"above_{cap}"] = expected_above(mean_r, k, n_obs, cap)
        print(f"  nodes {offspring.size} chains {n_obs} R {mean_r:.4f} k {k:.4f} longest {observed['longest']}", flush=True)
        # Empirical iid. V4 also estimates P(any chain >= 585) over blocks of the study size.
        if label == "v4_shape0.03":
            n_blocks = 300
            block = n_obs
            accum = {f"above_{cap}": 0 for cap in caps}
            longest = 0
            hits_585 = 0
            capped = 0
            for block_i in range(n_blocks):
                drawn = simulate_sizes(offspring, block, 5000, rng)
                for cap in caps:
                    accum[f"above_{cap}"] += int(np.sum(drawn > cap))
                longest = max(longest, int(drawn.max()))
                if np.any(drawn >= 585):
                    hits_585 += 1
                capped += int(np.sum(drawn >= 5000))
                if (block_i + 1) % 50 == 0:
                    print(f"  block {block_i + 1}/{n_blocks} hit585 {hits_585}", flush=True)
            n_sim = n_blocks * block
            iid = {key: value * n_obs / n_sim for key, value in accum.items()}
            iid["longest_in_one_study_max_seen"] = longest
            iid["blocks"] = n_blocks
            iid["block_size"] = block
            iid["share_blocks_with_chain_at_least_585"] = hits_585 / n_blocks
            iid["chains_simulated"] = n_sim
            iid["hit_cap_5000"] = capped
        else:
            n_sim = 100000
            drawn = simulate_sizes(offspring, n_sim, 5000, rng)
            sim_counts = count_above(drawn, caps)
            iid = scale_counts(sim_counts, n_sim, n_obs)
            iid["longest_among_simulated"] = int(drawn.max())
            iid["chains_simulated"] = n_sim
            iid["hit_cap_5000"] = int(np.sum(drawn >= 5000))
        nb_p585 = None
        if label == "v4_shape0.03" and k and math.isfinite(k):
            nb_p585 = p_any_at_least(mean_r, k, n_obs, 585)
        record = {
            "arm": label,
            "nodes": int(offspring.size),
            "chains": n_obs,
            "mean_offspring": mean_r,
            "k": k,
            "zero_share": float(np.mean(offspring == 0)),
            "observed": observed,
            "poisson_band_95": {key: {"low": low, "high": high} for key, (low, high) in bands.items()},
            "negative_binomial": nb,
            "empirical_iid": iid,
            "nb_p_any_chain_at_least_585": nb_p585,
        }
        detail[label] = record
        for source, payload in (("observed", observed), ("negative_binomial", nb), ("empirical_iid", iid)):
            row = {"arm": label, "source": source, "chains": n_obs, "mean_offspring": mean_r, "k": k}
            for cap in caps:
                row[f"above_{cap}"] = payload.get(f"above_{cap}")
            if source == "observed":
                row["longest"] = observed["longest"]
                for cap in caps:
                    low, high = bands[f"above_{cap}"]
                    row[f"above_{cap}_poisson_low"] = low
                    row[f"above_{cap}_poisson_high"] = high
            elif source == "empirical_iid":
                row["longest"] = payload.get("longest_in_one_study_max_seen", payload.get("longest_among_simulated"))
                row["chains_simulated"] = payload.get("chains_simulated")
                row["share_blocks_chain_at_least_585"] = payload.get("share_blocks_with_chain_at_least_585")
            rows.append(row)
        print(f"  observed above10 {observed['above_10']} iid {iid.get('above_10')} nb {nb.get('above_10')}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "w1.json").write_text(json.dumps(detail, indent=1) + "\n")
    write_csv(OUT_DIR / "29_chain_counts.csv", rows)
    return detail


def composition_rows(label, chains):
    chains = sorted(chains, key=lambda item: item["size"], reverse=True)
    rows = []

    def add(kind, chain, rank):
        size = chain["size"]
        locs = chain["locations"]
        largest = max(locs.values()) if locs else 0
        largest_loc = max(locs, key=locs.get) if locs else None
        row = {
            "arm": label,
            "row_kind": kind,
            "rank": rank,
            "rep": chain.get("rep"),
            "size": size,
            "generations": chain["generations"],
            "distinct_locations": len(locs),
            "largest_location_share": largest / size if size else None,
            "largest_location_index": largest_loc,
        }
        for name in ("restaurant", "church", "park", "cinema", "shop", "work", "school", "hospital", "home", "other"):
            row[f"share_{name}"] = chain["groups"].get(name, 0) / size if size else None
        rows.append(row)
        if kind == "longest_20":
            for gen, counts in sorted(chain["by_gen"].items(), key=lambda item: int(item[0])):
                gen_n = sum(counts.values())
                grow = {
                    "arm": label,
                    "row_kind": "generation",
                    "rank": rank,
                    "rep": chain.get("rep"),
                    "size": size,
                    "generation": int(gen),
                    "generation_infections": gen_n,
                }
                for loc_type, count in sorted(counts.items(), key=lambda item: int(item[0])):
                    grow[f"type_{loc_type}"] = count
                rows.append(grow)

    for rank, chain in enumerate(chains[:20], start=1):
        add("longest_20", chain, rank)
    above_200 = [chain for chain in chains if chain["size"] > 200]
    band = [chain for chain in chains if 51 <= chain["size"] <= 200]
    for kind, group in (("pooled_above_200", above_200), ("pooled_51_to_200", band)):
        if not group:
            rows.append({"arm": label, "row_kind": kind, "chains": 0})
            continue
        groups = Counter()
        size = 0
        distinct = []
        largest_shares = []
        for chain in group:
            groups.update(chain["groups"])
            size += chain["size"]
            distinct.append(len(chain["locations"]))
            largest_shares.append(max(chain["locations"].values()) / chain["size"])
        row = {
            "arm": label,
            "row_kind": kind,
            "chains": len(group),
            "infections": size,
            "median_distinct_locations": percentile(distinct, 0.5),
            "median_largest_location_share": percentile(largest_shares, 0.5),
        }
        for name in ("restaurant", "church", "park", "cinema", "shop", "work", "school", "hospital", "home", "other"):
            row[f"share_{name}"] = groups.get(name, 0) / size if size else None
        rows.append(row)
    summary = {
        "chains": len(chains),
        "above_100": sum(chain["size"] > 100 for chain in chains),
        "above_200": sum(chain["size"] > 200 for chain in chains),
        "above_500": sum(chain["size"] > 500 for chain in chains),
        "longest": chains[0]["size"] if chains else None,
    }
    return rows, summary


def w2():
    arms = {
        "v4_shape0.03_new": V25 / "v4" / "shape0.03",
        "x3_shape0.03_old": X21 / "x3" / "shape0.03",
    }
    rows = []
    summary = {}
    for label, directory in arms.items():
        print(f"W2 {label}", flush=True)
        _offspring, chains = load_arm(directory)
        arm_rows, arm_summary = composition_rows(label, chains)
        rows.extend(arm_rows)
        summary[label] = arm_summary
        print(" ", arm_summary, flush=True)
    write_csv(OUT_DIR / "29_long_chain_composition.csv", rows)
    (RESULTS / "w2.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def index_cases(path):
    day = {}
    shed = {}
    known = set()
    children = defaultdict(list)
    with open(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        index = {name: i for i, name in enumerate(header)}
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            agent = int(parts[index["agent"]])
            infector = int(parts[index["infector"]])
            d = int(parts[index["timestamp"]]) // STEPS
            day[agent] = d
            shed[agent] = float(parts[index["shed"]])
            if infector != UNKNOWN:
                known.add(agent)
                children[infector].append(d)
    sheds, offs = [], []
    for agent, d in day.items():
        if agent not in known or not (1 <= d <= 10):
            continue
        sheds.append(shed[agent])
        offs.append(sum(1 for child_day in children.get(agent, ()) if child_day <= d + 14))
    return np.asarray(sheds, dtype=float), np.asarray(offs, dtype=float)


def curve_stats(sheds, offs):
    n = len(offs)
    if n == 0:
        return None
    early_r = float(offs.mean())
    order = np.argsort(sheds)
    top_n = max(1, int(round(0.01 * n)))
    top = order[-top_n:]
    top_shed = float(sheds[top].mean())
    top_off = float(offs[top].mean())
    ratio = top_off / (early_r * top_shed) if early_r > 0 and top_shed > 0 else None
    median = float(np.median(sheds))
    low = sheds <= median
    low_shed = float(sheds[low].mean()) if low.any() else 0.0
    low_off = float(offs[low].mean()) if low.any() else 0.0
    low_slope = low_off / low_shed if low_shed > 0 else None
    ratio_lower = top_off / (low_slope * top_shed) if low_slope and top_shed > 0 else None
    mid = (sheds >= 0.5) & (sheds < 1.0)
    if mid.sum() >= 5 and float(sheds[mid].mean()) > 0:
        mod_slope = float(offs[mid].mean()) / float(sheds[mid].mean())
        ratio_mod = top_off / (mod_slope * top_shed) if top_shed > 0 else None
    else:
        mod_slope = None
        ratio_mod = None
    return {
        "n": n,
        "early_R": early_r,
        "top_n": top_n,
        "top_shed": top_shed,
        "top_offspring": top_off,
        "reference_free_ratio": ratio,
        "lower_half_slope": low_slope,
        "lower_half_ratio": ratio_lower,
        "moderate_bin_n": int(mid.sum()),
        "moderate_bin_slope": mod_slope,
        "moderate_bin_ratio": ratio_mod,
    }


def log_bins(sheds):
    positive = sheds[sheds > 0]
    if positive.size == 0:
        return np.array([0.0, 1.0])
    lo = float(positive.min())
    hi = float(positive.max())
    if hi <= lo:
        hi = lo * 1.01
    return np.geomspace(lo, hi, 26)


def bin_table(reps, edges):
    rows = []
    # zero bin then 25 positive bins
    for sheds, offs in reps:
        pass
    pooled_s = np.concatenate([s for s, _o in reps])
    pooled_o = np.concatenate([o for _s, o in reps])
    masks = [(pooled_s <= 0)]
    for lo, hi in zip(edges[:-1], edges[1:]):
        masks.append((pooled_s >= lo) & (pooled_s < hi if hi < edges[-1] else pooled_s <= hi))
    labels = ["zero"] + [f"log_{i:02d}" for i in range(1, 26)]
    point = []
    for label, mask in zip(labels, masks):
        n = int(mask.sum())
        mean_s = float(pooled_s[mask].mean()) if n else None
        mean_o = float(pooled_o[mask].mean()) if n else None
        per = mean_o / mean_s if n and mean_s and mean_s > 0 else None
        point.append((label, n, mean_s, mean_o, per, mask))
    # replicate bootstrap of offspring per unit multiplier
    boot = {label: [] for label, *_ in point}
    ratio_boot = []
    rng = np.random.default_rng(28)
    n_rep = len(reps)
    for _ in range(1000):
        choice = rng.integers(0, n_rep, size=n_rep)
        s = np.concatenate([reps[i][0] for i in choice])
        o = np.concatenate([reps[i][1] for i in choice])
        stats = curve_stats(s, o)
        if stats and stats["reference_free_ratio"] is not None:
            ratio_boot.append(stats["reference_free_ratio"])
        for label, _n, _ms, _mo, _per, _mask in point:
            if label == "zero":
                mask = s <= 0
            else:
                i = int(label.split("_")[1]) - 1
                lo, hi = edges[i], edges[i + 1]
                mask = (s >= lo) & (s < hi if hi < edges[-1] else s <= hi)
            if mask.sum() < 5 or float(s[mask].mean()) <= 0:
                continue
            boot[label].append(float(o[mask].mean()) / float(s[mask].mean()))
    rows = []
    for label, n, mean_s, mean_o, per, _mask in point:
        samples = sorted(boot[label])
        if len(samples) >= 40:
            low = samples[int(0.025 * len(samples))]
            high = samples[int(0.975 * len(samples)) - 1]
        else:
            low = high = None
        lo_edge = 0.0 if label == "zero" else float(edges[int(label.split("_")[1]) - 1])
        hi_edge = 0.0 if label == "zero" else float(edges[int(label.split("_")[1])])
        rows.append({
            "bin": label,
            "bin_lo": lo_edge,
            "bin_hi": hi_edge,
            "infectors": n,
            "mean_multiplier": mean_s,
            "mean_offspring": mean_o,
            "offspring_per_multiplier": per,
            "offspring_per_multiplier_low": low,
            "offspring_per_multiplier_high": high,
        })
    ratio_boot.sort()
    ratio_ci = None
    if len(ratio_boot) >= 40:
        ratio_ci = (ratio_boot[int(0.025 * len(ratio_boot))], ratio_boot[int(0.975 * len(ratio_boot)) - 1])
    return rows, ratio_ci


def w4a():
    shapes = ("0.3", "0.08", "0.03")
    all_rows = []
    summary = {}
    for shape in shapes:
        directory = V25 / "v3" / f"shape{shape}"
        reps = []
        for path in sorted(directory.glob("rep*/trace.tsv")):
            reps.append(index_cases(path))
        pooled_s = np.concatenate([s for s, _o in reps])
        pooled_o = np.concatenate([o for _s, o in reps])
        stats = curve_stats(pooled_s, pooled_o)
        edges = log_bins(pooled_s)
        rows, ratio_ci = bin_table(reps, edges)
        for row in rows:
            row["map"] = "new"
            row["shape"] = shape
            all_rows.append(row)
        stats["reference_free_low"] = ratio_ci[0] if ratio_ci else None
        stats["reference_free_high"] = ratio_ci[1] if ratio_ci else None
        summary[shape] = stats
        print(f"W4a shape {shape} R {stats['early_R']:.3f} ratio {stats['reference_free_ratio']:.3f} ci {ratio_ci}", flush=True)
    # Existing PS1 shape 0.3 is reported with the new-map table's summary columns.
    ps1 = STUDY / "results" / "ps" / "ps1" / "E"
    if ps1.exists():
        reps = [index_cases(path) for path in sorted(ps1.glob("rep*/trace.tsv"))]
        if reps:
            stats = curve_stats(np.concatenate([s for s, _o in reps]), np.concatenate([o for _s, o in reps]))
            summary["old_ps1_shape0.3"] = stats
            print(f"W4a old PS1 E ratio {stats['reference_free_ratio']:.3f}", flush=True)
    write_csv(OUT_DIR / "29_offspring_curve.csv", all_rows)
    (RESULTS / "w4a.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def load_daily_hospital(path):
    sys.path.insert(0, str(STUDY))
    from analyze import hospital, load_daily

    rows = load_daily(path)
    return [hospital(row) for row in rows]


def welch_mdd(values_a, values_b, alpha=0.05, power=0.8):
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    n1, n2 = len(a), len(b)
    v1, v2 = a.var(ddof=1) / n1, b.var(ddof=1) / n2
    se = math.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    # Normal approximation with a small-sample correction from the t critical values' leading term.
    z_a, z_b = 1.95996398454, 0.841621233573
    def t_approx(z):
        return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df * df)
    return {
        "n_old": n1,
        "n_new": n2,
        "df": df,
        "mdd_80": (t_approx(z_a) + t_approx(z_b)) * se,
        "mean_old": float(a.mean()),
        "mean_new": float(b.mean()),
        "sd_old": float(a.std(ddof=1)),
        "sd_new": float(b.std(ddof=1)),
    }


def w6_and_presence():
    print("W6 locations and occupancy", flush=True)
    locations = json.loads((ROOT / "inputRealExample" / "locations1.json").read_text())
    places = locations["places"]
    print("W6 agents", flush=True)
    agents = json.loads((ROOT / "inputRealExample" / "agents2.json").read_text())
    assigned = defaultdict(int)
    for person in agents["people"]:
        seen = set()
        for slot in person.get("locations", []):
            if slot.get("typeID") in AUDIT_TYPES:
                loc_id = slot.get("locID")
                if loc_id not in seen:
                    seen.add(loc_id)
                    assigned[loc_id] += 1
    peaks = {}
    p95 = {}
    days = {}
    with open(V25 / "v0" / "occupancy.tsv") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        index = {name: i for i, name in enumerate(header)}
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            loc = int(parts[index["location"]])
            peaks[loc] = float(parts[index["maxOccupancy"]])
            p95[loc] = float(parts[index["dailyMaxP95"]])
            days[loc] = int(float(parts[index["daysObserved"]]))
    fill_rows = []
    assign_rows = []
    for loc_type in AUDIT_TYPES:
        idxs = [i for i, place in enumerate(places) if place.get("type") == loc_type]
        fills = []
        above = 0
        caps = []
        assigns = []
        for i in idxs:
            cap = places[i].get("capacity") or 0
            peak = peaks.get(i, 0.0)
            caps.append(cap)
            assigns.append(assigned.get(places[i]["ID"], 0))
            if cap > 0:
                fills.append(peak / cap)
                if peak > cap:
                    above += 1
        fill_rows.append({
            "type": loc_type,
            "locations": len(idxs),
            "capacity_median": percentile(caps, 0.5),
            "capacity_max": max(caps) if caps else None,
            "distinct_capacities": len(set(caps)),
            "peak_median": percentile([peaks.get(i, 0.0) for i in idxs], 0.5),
            "fill_median": percentile(fills, 0.5),
            "fill_p90": percentile(fills, 0.9),
            "fill_max": max(fills) if fills else None,
            "share_peak_above_capacity": above / len(fills) if fills else None,
        })
        assign_rows.append({
            "type": loc_type,
            "locations": len(idxs),
            "assigned_p05": percentile(assigns, 0.05),
            "assigned_p25": percentile(assigns, 0.25),
            "assigned_p50": percentile(assigns, 0.50),
            "assigned_p75": percentile(assigns, 0.75),
            "assigned_p95": percentile(assigns, 0.95),
            "assigned_p99": percentile(assigns, 0.99),
            "assigned_max": max(assigns) if assigns else None,
            "correlation_assigned_capacity": float(np.corrcoef(assigns, caps)[0, 1]) if len(assigns) > 2 else None,
        })
    # Per type-5 venue presence rate.
    venue_rows = []
    stable_pi = []
    for i, place in enumerate(places):
        if place.get("type") != 5:
            continue
        cap = place.get("capacity") or 0
        n_assigned = assigned.get(place["ID"], 0)
        peak = peaks.get(i, 0.0)
        row = {
            "index": i,
            "location_id": place["ID"],
            "assigned": n_assigned,
            "daily_peak_p95": p95.get(i, 0.0),
            "peak_occupancy": peak,
            "capacity": cap,
            "days_observed": days.get(i, 0),
        }
        row["fill"] = peak / cap if cap else None
        row["pi"] = peak / n_assigned if n_assigned else None
        stable = n_assigned >= 50 and days.get(i, 0) >= 5 and peak > 0
        row["pi_stable"] = stable
        if stable:
            stable_pi.append(row["pi"])
        venue_rows.append(row)
    pooled_pi = float(np.median(stable_pi)) if stable_pi else None
    used_pooled = 0
    absorb = 0.0
    for row in venue_rows:
        pi = row["pi"] if row["pi_stable"] else pooled_pi
        if not row["pi_stable"]:
            used_pooled += 1
        row["pi_used"] = pi
        if pi and pi > 0:
            row["assignment_cap_fill_1_10"] = 1.10 * row["capacity"] / pi
            absorb += row["capacity"] * 1.10 / pi
        else:
            row["assignment_cap_fill_1_10"] = None
    # Waves
    def series_stats(paths):
        areas1, areas2, peaks2 = [], [], []
        for path in paths:
            series = load_daily_hospital(path)
            areas1.append(sum(series[:105]))
            areas2.append(sum(series[105:]))
            peaks2.append(max(series[140:]) if len(series) > 140 else None)
        return {"wave1_area": areas1, "wave2_area_after_day_105": areas2, "wave2_peak_after_day_140": peaks2}

    old_paths = sorted((STUDY / "results" / "p2" / "waves_baseline").glob("rep*/stdout.txt"))
    new_paths = sorted((V25 / "v1" / "waves_new").glob("rep*/stdout.txt"))
    old = series_stats(old_paths)
    new = series_stats(new_paths)
    mdd_area1 = welch_mdd(old["wave1_area"], new["wave1_area"])
    mdd_area2 = welch_mdd(old["wave2_area_after_day_105"], new["wave2_area_after_day_105"])
    mdd_peak2 = welch_mdd(old["wave2_peak_after_day_140"], new["wave2_peak_after_day_140"])
    payload = {
        "fill": fill_rows,
        "assignment": assign_rows,
        "presence": {
            "venues": len(venue_rows),
            "stable_pi_venues": len(stable_pi),
            "pooled_pi_median_of_stable": pooled_pi,
            "venues_using_pooled_pi": used_pooled,
            "pi_p05": percentile(stable_pi, 0.05),
            "pi_p50": percentile(stable_pi, 0.50),
            "pi_p95": percentile(stable_pi, 0.95),
            "absorbable_assignments_at_fill_1_10": absorb,
            "assignments_needed": 461130,
            "feasible_at_1_10": absorb >= 461130 * 1.02,
            "slack": absorb - 461130,
        },
        "waves": {"old": old, "new": new, "mdd_wave1_area": mdd_area1, "mdd_wave2_area": mdd_area2, "mdd_wave2_peak": mdd_peak2},
    }
    (RESULTS / "w6.json").write_text(json.dumps(payload, indent=1) + "\n")
    write_csv(OUT_DIR / "29_audit_fill_and_assignment.csv", fill_rows + assign_rows)
    write_csv(RESULTS / "w10_type5_presence.csv", venue_rows)
    print("presence", json.dumps(payload["presence"], indent=1), flush=True)
    print("mdd wave1", mdd_area1, flush=True)
    print("mdd wave2 area", mdd_area2, flush=True)
    print("mdd wave2 peak", mdd_peak2, flush=True)
    return payload


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}", flush=True)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    steps = sys.argv[1:] or ["w1", "w2", "w4a", "w6"]
    for step in steps:
        {"w1": w1, "w2": w2, "w4a": w4a, "w6": w6_and_presence}[step]()
    print("done", flush=True)


if __name__ == "__main__":
    main()
