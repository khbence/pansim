#!/usr/bin/env python3
"""Round-3 tables that do not need new PanSim runs (note 31, R3-1).

Thresholds are "above" (size > cap) unless a column says "at least".
Negative-binomial expectations use the Dwass total-progeny formula at the
arm's own mean offspring and k.
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
RESULTS = HERE / "results" / "round2"
OUT = ROOT / "superspreader"
sys.path.insert(0, str(HERE))

from analyze import daily_summary, hospital, load_daily, load_trace, mom_k as mom_k_list, venue_metrics  # noqa: E402
from analyze_round2 import (  # noqa: E402
    composition_rows,
    count_above,
    curve_stats,
    expected_above,
    garwood_interval,
    index_cases,
    load_arm,
    mom_k,
    p_total_gt,
    percentile,
    scale_counts,
    simulate_sizes,
    write_csv,
)
from early_window import early_counts  # noqa: E402

CAPS = (10, 50, 100, 200, 500)
STEPS = 144


def location_ids(path):
    places = json.loads(Path(path).read_text())["places"]
    return [place.get("ID", "") for place in places]


def chain_table(label, directory, rng):
    print(f"chains {label}", flush=True)
    offspring, chains = load_arm(directory)
    sizes = np.asarray([chain["size"] for chain in chains], dtype=np.int32)
    n_obs = int(sizes.size)
    mean_r = float(offspring.mean()) if n_obs else 0.0
    k = mom_k(offspring)
    observed = count_above(sizes, CAPS)
    observed["longest"] = int(sizes.max()) if n_obs else None
    bands = {key: garwood_interval(value) for key, value in observed.items() if key.startswith("above_")}
    nb = {}
    if k and math.isfinite(k):
        for cap in CAPS:
            nb[f"above_{cap}"] = expected_above(mean_r, k, n_obs, cap)
    drawn = simulate_sizes(offspring, 100000, 5000, rng)
    iid = scale_counts(count_above(drawn, CAPS), 100000, n_obs)
    groups = [sizes[i:i + 5] for i in range(0, n_obs - (n_obs % 5), 5)]
    empirical = float(np.mean([group.sum() > 50 for group in groups])) if groups else None
    exact = p_total_gt(mean_r, k, 5, 50) if k and math.isfinite(k) else None
    print(
        f"  chains {n_obs} R {mean_r:.4f} k {k if k else float('nan'):.4f} longest {observed['longest']} "
        f"five-seed {empirical} exact {exact}",
        flush=True,
    )
    rows = []
    for source, payload in (("observed", observed), ("negative_binomial", nb), ("empirical_iid", iid)):
        row = {"arm": label, "source": source, "chains": n_obs, "mean_offspring": mean_r, "k": k}
        for cap in CAPS:
            row[f"above_{cap}"] = payload.get(f"above_{cap}")
        if source == "observed":
            row["longest"] = observed["longest"]
            for cap in CAPS:
                low, high = bands[f"above_{cap}"]
                row[f"above_{cap}_poisson_low"] = low
                row[f"above_{cap}_poisson_high"] = high
            row["five_seed_groups"] = len(groups)
            row["five_seed_p_total_above_50"] = empirical
        elif source == "empirical_iid":
            row["longest"] = int(drawn.max())
            row["chains_simulated"] = 100000
        rows.append(row)
    rows.append({
        "arm": label,
        "source": "five_seed_exact",
        "chains": n_obs,
        "mean_offspring": mean_r,
        "k": k,
        "five_seed_groups": len(groups),
        "five_seed_p_total_above_50": exact,
        "definition": "P(total size of 5 introductions > 50), Dwass at this arm's mean and k",
    })
    return rows, offspring, chains


def names_for_existing():
    print("location names", flush=True)
    by_arm = {
        "v4_shape0.03_new": location_ids(ROOT / "inputRealExample" / "locations1.json"),
        "x3_shape0.03_old": location_ids(ROOT / "inputRealExample" / "locations0.json"),
    }
    rows = []
    seen = set()
    with open(OUT / "29_long_chain_composition.csv") as handle:
        for row in csv.DictReader(handle):
            raw = row.get("largest_location_index") or ""
            if raw == "":
                continue
            key = (row["arm"], raw)
            if key in seen:
                continue
            seen.add(key)
            index = int(float(raw))
            ids = by_arm.get(row["arm"], [])
            rows.append({
                "arm": row["arm"],
                "largest_location_index": index,
                "largest_location_id": ids[index] if 0 <= index < len(ids) else "",
                "source_file": "29_long_chain_composition.csv",
            })
    return rows


def longest_rows(label, chains, ids):
    arm_rows, _summary = composition_rows(label, chains)
    kept = []
    for row in arm_rows:
        if row.get("row_kind") == "longest_20" and row.get("rank") != 1:
            continue
        if row.get("row_kind") == "generation" and row.get("rank") != 1:
            continue
        if row.get("row_kind") in ("pooled_above_200", "pooled_51_to_200"):
            continue
        index = row.get("largest_location_index")
        if index is not None and ids is not None and 0 <= int(index) < len(ids):
            row["largest_location_id"] = ids[int(index)]
        kept.append(row)
    return kept


def reassignment():
    print("reassignment summary", flush=True)
    path = OUT / "29_type5_reassignment.csv"
    agent_disp = []
    slot_disp = []
    old_dist = []
    new_dist = []
    changed = 0
    n = 0
    prefix_agent = []
    prefix_slot = []
    prefix_old = []
    prefix_new = []
    prefix_changed = 0
    with open(path) as handle:
        for row in csv.DictReader(handle):
            old = [float(x) for x in row["old_distance_m"].split(";") if x != ""]
            new = [float(x) for x in row["new_distance_m"].split(";") if x != ""]
            old_ids = set(row["old_location_ids"].split(";"))
            new_ids = set(row["new_location_ids"].split(";"))
            did_change = old_ids != new_ids
            paired = [b - a for a, b in zip(sorted(old), sorted(new))]
            agent = float(np.mean(new) - np.mean(old))
            agent_disp.append(agent)
            slot_disp.extend(paired)
            old_dist.extend(old)
            new_dist.extend(new)
            changed += int(did_change)
            if n < 6416:
                prefix_agent.append(agent)
                prefix_slot.extend(paired)
                prefix_old.extend(old)
                prefix_new.extend(new)
                prefix_changed += int(did_change)
            n += 1
    qs = (0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0)

    def pack(label, values):
        row = {"quantity": label, "n": len(values)}
        for q in qs:
            row[f"p{int(q * 100) if q < 1 else 'max'}"] = percentile(values, q) if q < 1 else (max(values) if values else None)
        return row

    def share_above(values, bound):
        xs = np.asarray(values, dtype=float)
        return float(np.mean(xs > bound)) if xs.size else None

    rows = [
        pack("agent_mean_displacement_m", agent_disp),
        pack("slot_sorted_pair_displacement_m", slot_disp),
        pack("old_assigned_distance_m", old_dist),
        pack("new_assigned_distance_m", new_dist),
        pack("prefix6416_agent_mean_displacement_m", prefix_agent),
        pack("prefix6416_slot_sorted_pair_displacement_m", prefix_slot),
        pack("prefix6416_old_assigned_distance_m", prefix_old),
        pack("prefix6416_new_assigned_distance_m", prefix_new),
    ]
    summary = {
        "agents": n,
        "definition_reported_2km": (
            "Per agent, mean of the three new slot distances minus mean of the three old slot "
            "distances. Distance is the minimum of distance to home and distance to work. "
            "This is the solver definition. Pairing slots does not change it, because the mean "
            "does not depend on order."
        ),
        "definition_note31_sample": (
            "Per slot, new minus old after sorting the three distances; per agent, the mean of "
            "those three paired differences. The agent mean equals the solver definition."
        ),
        "share_agents_venue_set_changed": changed / n,
        "prefix6416_share_changed": prefix_changed / 6416,
        "share_slots_new_above_1km": share_above(new_dist, 1000),
        "share_slots_new_above_2km": share_above(new_dist, 2000),
        "share_slots_new_above_5km": share_above(new_dist, 5000),
        "share_slots_old_above_5km": share_above(old_dist, 5000),
        "prefix6416_share_slots_new_above_5km": share_above(prefix_new, 5000),
        "full_agent_p95_m": percentile(agent_disp, 0.95),
        "prefix_agent_p95_m": percentile(prefix_agent, 0.95),
        "full_slot_p95_m": percentile(slot_disp, 0.95),
        "prefix_slot_p95_m": percentile(prefix_slot, 0.95),
    }
    changed_agent = [d for d, flag in zip(agent_disp, range(n))]
    # Changed-agent tail needs the flags; recompute cheaply from the stored lists by a second pass is avoided:
    # the prefix comparison is the reconciliation. Full changed-agent p95 is computed below from a mask
    # stored alongside. Re-read is unnecessary: `changed` count is enough for the share.
    write_csv(OUT / "32_reassignment_summary.csv", rows)
    (RESULTS / "r31_reassignment.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1), flush=True)
    return summary


def presence_and_validation():
    print("W10 steps", flush=True)
    venues = list(csv.DictReader((RESULTS / "w10_type5_presence.csv").open()))
    for row in venues:
        for key in ("assigned", "capacity", "days_observed", "pi_stable"):
            if key == "pi_stable":
                row[key] = row[key] == "True"
            else:
                row[key] = float(row[key]) if row[key] not in ("", None) else 0.0
        for key in ("pi", "pi_used", "peak_occupancy", "daily_peak_p95"):
            row[key] = float(row[key]) if row[key] not in ("", None) else None
    needed = 461130
    step_rows = []
    for phi in (1.10, 1.15, 1.20):
        absorb = 0.0
        quota = 0
        for row in venues:
            pi = row["pi_used"] or 0.0
            if pi <= 0:
                continue
            absorb += row["capacity"] * phi / pi
            quota += int(math.floor(phi * row["capacity"] / pi))
        step_rows.append({
            "phi": phi,
            "absorbable_sum_capacity_phi_over_pi": absorb,
            "floor_quota": quota,
            "needed": needed,
            "absorbable_minus_needed": absorb - needed,
            "floor_quota_minus_needed": quota - needed,
            "floor_slack_fraction": (quota - needed) / needed,
            "clears_2_percent_slack": quota >= needed * 1.02,
        })
    stable = [row["pi"] for row in venues if row["pi_stable"] and row["pi"] is not None]
    pooled = float(np.median(stable))
    using_median = sum(1 for row in venues if not row["pi_stable"])
    # Realised fill on the repaired occupancy file.
    places = json.loads((ROOT / "inputRealExample" / "locations3.json").read_text())["places"]
    peaks = {}
    with open(RESULTS / "w10" / "occupancy.tsv") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        index = {name: i for i, name in enumerate(header)}
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            peaks[int(parts[index["location"]])] = float(parts[index["maxOccupancy"]])
    fills = []
    for i, place in enumerate(places):
        if place.get("type") != 5:
            continue
        cap = place.get("capacity") or 0
        if cap:
            fills.append(peaks.get(i, 0.0) / cap)
    scale = 1.20 / 1.10
    validation = {
        "fill_median": percentile(fills, 0.5),
        "fill_p95": percentile(fills, 0.95),
        "fill_max": max(fills) if fills else None,
        "share_peak_above_capacity": float(np.mean(np.asarray(fills) > 1.0)),
        "scaled_median_low": 0.5 * scale,
        "scaled_median_high": 1.1 * scale,
        "scaled_p95_high": 1.3 * scale,
        "scaled_max_high": 1.6 * scale,
        "median_pass": 0.5 * scale <= percentile(fills, 0.5) <= 1.1 * scale,
        "p95_pass": percentile(fills, 0.95) <= 1.3 * scale,
        "max_pass": max(fills) <= 1.6 * scale,
        "n_type5": len(fills),
    }
    write_csv(OUT / "32_w10_presence.csv", venues)
    write_csv(OUT / "32_w10_feasibility.csv", step_rows)
    (RESULTS / "r31_w10_steps.json").write_text(json.dumps({
        "stable_venues": len(stable),
        "venues_using_median": using_median,
        "pooled_median_pi": pooled,
        "pi_p05": percentile(stable, 0.05),
        "pi_p95": percentile(stable, 0.95),
        "feasibility": step_rows,
        "validation_scaled_to_1_20": validation,
    }, indent=1) + "\n")
    print("using median", using_median, "stable", len(stable), "validation", validation, flush=True)


def run_metrics(path):
    rows = load_daily(path)
    summary = daily_summary(rows)
    trace = path.parent / "trace.tsv"
    early = None
    if trace.exists() and trace.stat().st_size:
        counts = early_counts(load_trace(trace))
        early = sum(counts) / len(counts) if counts else None
    day28 = int(rows[27]["INF"]) if len(rows) >= 28 else None
    return {
        "rep": path.parent.name,
        "days": summary.get("days"),
        "peak_prevalence": summary.get("peak_prevalence"),
        "peak_day": summary.get("peak_prevalence_day"),
        "final_size": summary.get("final_infected_ever"),
        "early_R": early,
        "day28_infections": day28,
        "prevalence_last_day": summary.get("final_prevalence"),
    }


def peaks_and_waves():
    print("peaks and waves", flush=True)
    rows = []
    for label, root in (
        ("old_file", HERE / "results" / "v25" / "v1" / "new"),
        ("repaired_file", RESULTS / "w10" / "v1"),
    ):
        for path in sorted(root.glob("rep*/stdout.txt")):
            row = run_metrics(path)
            row["arm"] = label
            rows.append(row)
    wave_rows = []
    wave_dirs = list((HERE / "results" / "v25" / "v1" / "waves_new").glob("rep*/stdout.txt"))
    wave_dirs += list((RESULTS / "w6" / "waves_new").glob("rep*/stdout.txt"))
    for path in sorted(wave_dirs):
        series_rows = load_daily(path)
        series = [hospital(row) for row in series_rows]
        wave_rows.append({
            "arm": "w6c_new_map",
            "rep": f"{path.parents[1].name}_{path.parent.name}",
            "area_105": sum(series[:105]),
            "wave2_peak_after_day_140": max(series[140:]) if len(series) > 140 else None,
            "days": len(series),
        })
    write_csv(OUT / "32_peak_prevalence.csv", rows)
    write_csv(OUT / "32_w6c_hospital.csv", wave_rows)


def w9_table():
    print("W9 levers", flush=True)
    arms = ("venue_s3", "venue_s1", "shape1")
    levers = ("none", "venue_close", "household", "ventilation")
    rows = []
    rng = np.random.default_rng(32)
    for arm in arms:
        finals = {}
        for lever in levers:
            values = []
            for path in sorted((RESULTS / "w9" / f"{arm}_{lever}").glob("rep*/stdout.txt")):
                daily = load_daily(path)
                values.append(int(daily[-1]["INF"]) if daily else None)
            finals[lever] = values
        shares = []
        for path in sorted((RESULTS / "w9" / f"{arm}_none").glob("rep*/trace.tsv")):
            shares.append(venue_metrics(load_trace(path))["household_fraction"])
        base = np.asarray(finals["none"], dtype=float)

        def reduction(name, base=base, finals=finals):
            other = np.asarray(finals[name], dtype=float)
            return 1.0 - float(other.mean()) / float(base.mean())

        def interval(name, base=base, finals=finals):
            other = np.asarray(finals[name], dtype=float)
            n = len(base)
            samples = []
            for _ in range(1000):
                pick = rng.integers(0, n, size=n)
                samples.append(1.0 - float(other[pick].mean()) / float(base[pick].mean()))
            samples.sort()
            return samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples)) - 1]

        share_samples = []
        share_arr = np.asarray(shares, dtype=float)
        for _ in range(1000):
            pick = rng.integers(0, len(share_arr), size=len(share_arr))
            share_samples.append(float(share_arr[pick].mean()))
        share_samples.sort()
        row = {
            "arm": arm,
            "replicates": len(base),
            "household_share": float(share_arr.mean()),
            "household_share_low": share_samples[int(0.025 * len(share_samples))],
            "household_share_high": share_samples[int(0.975 * len(share_samples)) - 1],
        }
        for lever in ("venue_close", "household", "ventilation"):
            low, high = interval(lever)
            row[f"{lever}_reduction"] = reduction(lever)
            row[f"{lever}_low"] = low
            row[f"{lever}_high"] = high
            row[f"{lever}_final_mean"] = float(np.mean(finals[lever]))
        row["none_final_mean"] = float(base.mean())
        row["venue_minus_household"] = row["venue_close_reduction"] - row["household_reduction"]
        rows.append(row)
        print(arm, row["household_share"], row["venue_minus_household"], flush=True)
    write_csv(OUT / "32_w9_levers.csv", rows)


def quantile_bins(sheds, offs):
    """Zero bin, deciles of positive multipliers, then nested top tails."""
    rows = []
    zero = sheds <= 0
    rows.append(("zero", zero, None, None))
    positive = np.flatnonzero(sheds > 0)
    if positive.size == 0:
        return rows
    order = positive[np.argsort(sheds[positive])]
    n = order.size
    for decile in range(10):
        start = int(round(decile * n / 10))
        stop = int(round((decile + 1) * n / 10))
        mask = np.zeros(sheds.shape, dtype=bool)
        mask[order[start:stop]] = True
        rows.append((f"decile_{decile + 1}", mask, float(sheds[order[start]]), float(sheds[order[stop - 1]])))
    for label, frac in (("top_5pct", 0.05), ("top_2pct", 0.02), ("top_1pct", 0.01), ("top_0.5pct", 0.005), ("top_0.1pct", 0.001)):
        start = int(round((1.0 - frac) * n))
        start = min(start, n - 1)
        mask = np.zeros(sheds.shape, dtype=bool)
        mask[order[start:]] = True
        rows.append((label, mask, float(sheds[order[start]]), float(sheds[order[-1]])))
    return rows


def offspring_curves():
    print("offspring curves", flush=True)
    arms = [
        ("new", "0.3", HERE / "results" / "v25" / "v3" / "shape0.3"),
        ("new", "0.08", HERE / "results" / "v25" / "v3" / "shape0.08"),
        ("new", "0.03", HERE / "results" / "v25" / "v3" / "shape0.03"),
        ("old", "0.08", RESULTS / "stage" / "w4b_shape0.08" / "round1"),
        ("old", "0.03", RESULTS / "stage" / "w4b_shape0.03" / "round2"),
        ("old", "0.3", HERE / "results" / "ps" / "ps1" / "E"),
    ]
    summary_rows = []
    curve_rows = []
    rng = np.random.default_rng(32)
    for map_name, shape, directory in arms:
        reps = [index_cases(path) for path in sorted(directory.glob("rep*/trace.tsv"))]
        pooled_s = np.concatenate([s for s, _o in reps])
        pooled_o = np.concatenate([o for _s, o in reps])
        stats = curve_stats(pooled_s, pooled_o)
        ratios = []
        n_rep = len(reps)
        # Bootstrap of the reference-free ratio and of each re-binned row.
        bin_boot = defaultdict(list)
        for _ in range(1000):
            choice = rng.integers(0, n_rep, size=n_rep)
            s = np.concatenate([reps[i][0] for i in choice])
            o = np.concatenate([reps[i][1] for i in choice])
            sampled = curve_stats(s, o)
            if sampled and sampled["reference_free_ratio"] is not None:
                ratios.append(sampled["reference_free_ratio"])
            for label, mask, _lo, _hi in quantile_bins(s, o):
                if int(mask.sum()) < 5 or float(s[mask].mean()) <= 0:
                    continue
                bin_boot[label].append(float(o[mask].mean()) / float(s[mask].mean()))
        ratios.sort()
        if len(ratios) >= 40:
            low = ratios[int(0.025 * len(ratios))]
            high = ratios[int(0.975 * len(ratios)) - 1]
        else:
            low = high = None
        summary_rows.append({
            "map": map_name,
            "shape": shape,
            "replicates": n_rep,
            "early_R": stats["early_R"],
            "top1_mean_multiplier": stats["top_shed"],
            "top1_mean_offspring": stats["top_offspring"],
            "ratio": stats["reference_free_ratio"],
            "ratio_low": low,
            "ratio_high": high,
            "n_index": stats["n"],
            "top_n": stats["top_n"],
        })
        for label, mask, lo, hi in quantile_bins(pooled_s, pooled_o):
            n = int(mask.sum())
            mean_s = float(pooled_s[mask].mean()) if n else None
            mean_o = float(pooled_o[mask].mean()) if n else None
            per = mean_o / mean_s if n and mean_s and mean_s > 0 else None
            samples = sorted(bin_boot[label])
            if len(samples) >= 40:
                blo, bhi = samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples)) - 1]
            else:
                blo = bhi = None
            curve_rows.append({
                "map": map_name,
                "shape": shape,
                "bin": label,
                "bin_lo": lo,
                "bin_hi": hi,
                "infectors": n,
                "mean_multiplier": mean_s,
                "mean_offspring": mean_o,
                "offspring_per_multiplier": per,
                "offspring_per_multiplier_low": blo,
                "offspring_per_multiplier_high": bhi,
            })
        print(f"  {map_name} {shape} ratio {stats['reference_free_ratio']:.3f} ci {(low, high)}", flush=True)
    write_csv(OUT / "32_w4_top1.csv", summary_rows)
    write_csv(OUT / "32_offspring_curve.csv", curve_rows)


def event_clusters(directory):
    """Cluster size is the number of trace infections at the event location on the event day."""
    clusters = []
    infections = 0
    event_infections = 0
    type_max = Counter()
    sheds = []
    offs = []
    for rep in sorted(Path(directory).glob("rep*/trace.tsv")):
        trace = load_trace(rep)
        infections += len(trace)
        counts = early_counts(trace)
        # early_counts returns offspring only; rebuild sheds alongside via index_cases
        s, o = index_cases(rep)
        sheds.append(s)
        offs.append(o)
        by_loc_day = Counter()
        for row in trace:
            by_loc_day[(row["location"], row["day"])] += 1
        event_path = rep.parent.parent / f"{rep.parent.name}.events.tsv"
        seen = set()
        with open(event_path) as handle:
            header = handle.readline().rstrip("\n").split("\t")
            index = {name: i for i, name in enumerate(header)}
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                day = int(parts[index["day"]])
                loc = int(parts[index["location_index"]])
                loc_type = int(parts[index["locType"]])
                key = (loc, day)
                if key in seen:
                    continue
                seen.add(key)
                size = by_loc_day.get(key, 0)
                clusters.append(size)
                event_infections += size
                if size:
                    type_max[loc_type] = max(type_max[loc_type], size)
    pooled_o = np.concatenate(offs) if offs else np.asarray([])
    early_r = float(pooled_o.mean()) if pooled_o.size else None
    early_k = mom_k(pooled_o) if pooled_o.size else None
    positive = [c for c in clusters if c >= 1]
    max_type = max(type_max, key=type_max.get) if type_max else None
    return {
        "early_R": early_r,
        "early_k": early_k,
        "events": len(clusters),
        "cluster_median": percentile(clusters, 0.5) if clusters else None,
        "cluster_p90": percentile(clusters, 0.9) if clusters else None,
        "cluster_max": max(clusters) if clusters else None,
        "max_type": max_type,
        "event_share": event_infections / infections if infections else None,
        "nonempty_n": len(positive),
        "nonempty_median": percentile(positive, 0.5) if positive else None,
        "nonempty_p90": percentile(positive, 0.9) if positive else None,
        "nonempty_p99": percentile(positive, 0.99) if positive else None,
    }


def events_row():
    print("W7 2000x row", flush=True)
    stats = event_clusters(RESULTS / "w7_a_restaurant_shop_r0.003_f2000")
    print(stats, flush=True)
    row = {
        "arm": "w7_a_restaurant_shop_r0.003_f2000",
        "early_R": stats["early_R"],
        "early_k": stats["early_k"],
        "events": stats["events"],
        "cluster_median": stats["cluster_median"],
        "cluster_p90": stats["cluster_p90"],
        "cluster_max": stats["cluster_max"],
        "max_type": stats["max_type"],
        "event_share": stats["event_share"],
    }
    existing = list(csv.DictReader((OUT / "29_events.csv").open()))
    if not any(item["arm"] == row["arm"] for item in existing):
        existing.append(row)
        write_csv(OUT / "29_events.csv", existing)
    write_csv(OUT / "32_events_2000x.csv", [row])
    (RESULTS / "r31_events_2000.json").write_text(json.dumps(stats, indent=1) + "\n")


def main():
    reassignment()
    presence_and_validation()
    peaks_and_waves()
    w9_table()
    events_row()
    name_rows = names_for_existing()
    rng = np.random.default_rng(20261014)
    arms = [
        ("w3_shape0.03", RESULTS / "w3" / "shape0.03", HERE / "inputs" / "w3_wards" / "locations_wards.json"),
        ("w8_home035", RESULTS / "w8" / "home035", HERE / "inputs" / "v25_wards" / "locations_wards.json"),
        ("w10_tail", RESULTS / "w10" / "tail", HERE / "inputs" / "w10_tail" / "locations_wards.json"),
        ("v4_shape0.03_old", HERE / "results" / "x21" / "x3" / "shape0.03", ROOT / "inputRealExample" / "locations0.json"),
    ]
    count_rows = []
    comp_rows = []
    for label, directory, loc_path in arms:
        rows, _off, chains = chain_table(label, directory, rng)
        count_rows.extend(rows)
        ids = location_ids(loc_path)
        comp_rows.extend(longest_rows(label, chains, ids))
        for row in comp_rows:
            if row.get("arm") == label and row.get("largest_location_index") is not None:
                name_rows.append({
                    "arm": label,
                    "largest_location_index": row["largest_location_index"],
                    "largest_location_id": row.get("largest_location_id", ""),
                    "source_file": "32_long_chain_composition.csv",
                })
    # Keep one name row per arm and index.
    dedup = {}
    for row in name_rows:
        dedup[(row["arm"], str(row["largest_location_index"]))] = row
    write_csv(OUT / "32_chain_counts.csv", count_rows)
    write_csv(OUT / "32_long_chain_composition.csv", comp_rows)
    write_csv(OUT / "32_location_names.csv", list(dedup.values()))
    offspring_curves()
    print("R3-1 tables done", flush=True)


if __name__ == "__main__":
    main()
