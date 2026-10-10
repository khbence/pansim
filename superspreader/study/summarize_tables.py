#!/usr/bin/env python3
"""Method-of-moments tables for the ACT-049 runs. Does not fit an MLE."""

import math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from analyze import (
    HOME,
    case_table,
    chain_sizes_fixed,
    daily_summary,
    generation1_r,
    json_ready,
    load_daily,
    load_trace,
    mean_var,
    mom_k,
    offspring_counts,
    share_for_fraction_of_mass,
    top_share,
    venue_metrics,
    write_json,
)

ROOT = Path(__file__).resolve().parent / "results"
OUT = Path(__file__).resolve().parent / "summaries"
START = date(2020, 9, 23)


def finite(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isinf(value):
        return "infinity"
    return value


def bundle(values):
    mean, var = mean_var(values)
    if mean is None:
        return {"n": 0}
    zeros = sum(value == 0 for value in values) / len(values)
    return {
        "n": len(values),
        "mean": mean,
        "variance": var,
        "k_mom": finite(mom_k(values)),
        "zero_fraction": zeros,
        "top10": top_share(values, 0.10),
        "top20": top_share(values, 0.20),
        "p80": share_for_fraction_of_mass(values, 0.8),
    }


def mean_of(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def runs_of(cell):
    return sorted(path.parent for path in cell.glob("rep*/stdout.txt") if (path.parent / "trace.tsv").exists())


def complete_counts(trace):
    """Offspring of every node, including terminal cases. Used for short extinct chains."""
    nodes = set()
    children = defaultdict(int)
    for row in trace:
        nodes.add(row["agent"])
        if row["infector"] is not None:
            nodes.add(row["infector"])
            children[row["infector"]] += 1
    return [children.get(agent, 0) for agent in nodes]


def window_counts(by_agent, children, index, start, end, horizon):
    counts = []
    for agent in index:
        row = by_agent[agent]
        day = row["day"]
        if day < start or day > end:
            continue
        if horizon is None:
            counts.append(len(children.get(agent, [])))
        else:
            counts.append(sum(1 for child in children.get(agent, []) if child["day"] <= day + horizon))
    return counts


def split_setting_counts(by_agent, children, index):
    home = []
    community = []
    for agent in index:
        kids = children.get(agent, [])
        home.append(sum(1 for child in kids if child["locType"] == HOME))
        community.append(sum(1 for child in kids if child["locType"] not in (HOME, -1)))
    return home, community


def pure_backward(by_agent, children, index):
    counts = []
    for agent in index:
        kids = children.get(agent, [])
        if any(child["diagnosed"] for child in kids):
            counts.append(len(kids))
    return counts


def observation_models(trace):
    by_agent, children, index, _last, _cutoff = case_table(trace)
    diagnosed_index = [agent for agent in index if by_agent[agent].get("diagnosed")]
    forward = [
        sum(1 for child in children.get(agent, []) if child["diagnosed"]) for agent in diagnosed_index
    ]
    backward = []
    for agent in index:
        kids = children.get(agent, [])
        diagnosed_children = sum(1 for child in kids if child["diagnosed"])
        if by_agent[agent].get("diagnosed") or diagnosed_children > 0:
            if by_agent[agent].get("diagnosed"):
                backward.append(len(kids))
            else:
                backward.append(diagnosed_children)
    venue_counts = defaultdict(int)
    venue_diagnosed = defaultdict(int)
    for row in trace:
        venue_counts[row["location"]] += 1
        if row["diagnosed"]:
            venue_diagnosed[row["location"]] += 1
    detected = {
        venue
        for venue, count in venue_counts.items()
        if count >= 3 and venue_diagnosed[venue] >= 1
    }
    observed = {row["agent"] for row in trace if row["location"] in detected}
    cluster_index = [agent for agent in index if agent in observed or by_agent[agent].get("synthetic")]
    cluster = [
        sum(1 for child in children.get(agent, []) if child["agent"] in observed) for agent in cluster_index
    ]
    home, community = split_setting_counts(by_agent, children, index)
    infections_in_detected = sum(1 for row in trace if row["location"] in detected)
    return {
        "true": bundle(offspring_counts(children, index)),
        "symptom_forward": bundle(forward),
        "backward_mixed": bundle(backward),
        "pure_backward_true_offspring": bundle(pure_backward(by_agent, children, index)),
        "cluster": bundle(cluster),
        "home_offspring": bundle(home),
        "community_offspring": bundle(community),
        "diagnosed_index_fraction": (len(diagnosed_index) / len(index)) if index else None,
        "detected_venues": len(detected),
        "infections_in_detected_venues": infections_in_detected / len(trace) if trace else None,
    }


def venue_clusters(trace):
    counts = defaultdict(int)
    types = {}
    for row in trace:
        if row["infector"] is None and row["day"] == 0 and row["infectedCount"] == 0:
            continue
        counts[row["location"]] += 1
        types[row["location"]] = row["locType"]
    by_type = defaultdict(list)
    for location, count in counts.items():
        by_type[types[location]].append(count)
    out = {}
    for loc_type, values in sorted(by_type.items()):
        if len(values) < 5:
            continue
        mean, _var = mean_var(values)
        out[str(loc_type)] = {
            "venues": len(values),
            "mean": mean,
            "k_mom": finite(mom_k(values)),
        }
    return out


def peak_in(series, start, end):
    window = series[start:end]
    if not window:
        return None, None
    offset = max(range(len(window)), key=lambda i: window[i])
    return window[offset], start + offset


def wave_peaks(stdout_path):
    rows = load_daily(stdout_path)
    daily = daily_summary(rows)
    from analyze import hospital, prevalence

    hosp = [hospital(row) for row in rows]
    prev = [prevalence(row) for row in rows]
    split = 140
    h1, d1 = peak_in(hosp, 0, split)
    h2, d2 = peak_in(hosp, split, len(hosp))
    p1, pd1 = peak_in(prev, 0, split)
    p2, pd2 = peak_in(prev, split, len(prev))

    def dated(day):
        if day is None:
            return None
        return (START + timedelta(days=day)).isoformat()

    return {
        "daily": daily,
        "wave1_hospital": h1,
        "wave1_hospital_day": d1,
        "wave1_hospital_date": dated(d1),
        "wave2_hospital": h2,
        "wave2_hospital_day": d2,
        "wave2_hospital_date": dated(d2),
        "wave1_prevalence": p1,
        "wave1_prevalence_day": pd1,
        "wave1_prevalence_date": dated(pd1),
        "wave2_prevalence": p2,
        "wave2_prevalence_day": pd2,
        "wave2_prevalence_date": dated(pd2),
        "day14_prevalence": prev[14] if len(prev) > 14 else None,
    }


def summarize_cell(cell, early=False, chains=False, waves=False):
    pooled = []
    complete = []
    early_7 = []
    early_3 = []
    per_run = []
    venue_location = defaultdict(int)
    venue_type = defaultdict(int)
    chain_sizes = []
    gen_seeds = 0
    gen_children = 0
    for run in runs_of(cell):
        print(f"  {run.parent.name}/{run.name}", flush=True)
        trace = load_trace(run / "trace.tsv")
        by_agent, children, index, _last, _cutoff = case_table(trace)
        counts = offspring_counts(children, index)
        pooled.extend(counts)
        if chains:
            complete.extend(complete_counts(trace))
        daily_path = run / "stdout.txt"
        if waves:
            daily = wave_peaks(daily_path)
        else:
            daily = {"daily": daily_summary(load_daily(daily_path))}
            rows = load_daily(daily_path)
            from analyze import prevalence

            prev = [prevalence(row) for row in rows]
            daily["day14_prevalence"] = prev[14] if len(prev) > 14 else None
        venues = venue_metrics(trace)
        item = {
            "run": run.name,
            "infections": len(trace),
            "index_cases": len(index),
            "offspring": bundle(counts),
            "household_fraction": venues["household_fraction"],
            "top10_venues": venues["top10_venues"],
            "generation1_r": generation1_r(trace),
            "peak_prevalence": daily["daily"].get("peak_prevalence"),
            "peak_hospital": daily["daily"].get("peak_hospital"),
            "quarantine_person_days": daily["daily"].get("quarantine_person_days"),
            "day14_prevalence": daily.get("day14_prevalence"),
        }
        if waves:
            item.update({key: daily[key] for key in daily if key != "daily"})
        if early:
            c7 = window_counts(by_agent, children, index, 0, 7, 14)
            c3 = window_counts(by_agent, children, index, 0, 3, 14)
            early_7.extend(c7)
            early_3.extend(c3)
            item["early_0_7"] = bundle(c7)
            item["early_0_3"] = bundle(c3)
        if chains:
            chain = chain_sizes_fixed(trace)
            chain_sizes.extend(chain.get("sizes", []))
            seeds = [row["agent"] for row in trace if row["infector"] is None]
            gen_seeds += len(seeds)
            child_n = defaultdict(int)
            for row in trace:
                if row["infector"] is not None:
                    child_n[row["infector"]] += 1
            gen_children += sum(child_n.get(seed, 0) for seed in seeds)
            item["chains"] = {key: chain[key] for key in ("chains", "max_chain", "mean_chain", "p_gt_10", "p_gt_50")}
        for row in trace:
            if row["infector"] is None and row["day"] == 0 and row["infectedCount"] == 0:
                continue
            venue_location[row["location"]] += 1
            venue_type[row["locType"]] += 1
        per_run.append(item)
    venue_values = list(venue_location.values())
    type_total = sum(venue_type.values()) or 1
    payload = {
        "replicates": len(per_run),
        "offspring": bundle(pooled),
        "mean_infections": mean_of(per_run, "infections"),
        "mean_peak_prevalence": mean_of(per_run, "peak_prevalence"),
        "mean_peak_hospital": mean_of(per_run, "peak_hospital"),
        "mean_quarantine_person_days": mean_of(per_run, "quarantine_person_days"),
        "mean_day14_prevalence": mean_of(per_run, "day14_prevalence"),
        "household_fraction": venue_type.get(HOME, 0) / type_total,
        "top10_venues_pooled": top_share(venue_values, 0.10),
        "top20_venues_pooled": top_share(venue_values, 0.20),
        "mean_top10_venues": mean_of(per_run, "top10_venues"),
        "runs": per_run,
    }
    if chains:
        payload["complete_offspring"] = bundle(complete)
    if early:
        payload["early_0_7"] = bundle(early_7)
        payload["early_0_3"] = bundle(early_3)
    if chains and chain_sizes:
        payload["chains"] = {
            "chains": len(chain_sizes),
            "max_chain": max(chain_sizes),
            "mean_chain": sum(chain_sizes) / len(chain_sizes),
            "p_gt_10": sum(size > 10 for size in chain_sizes) / len(chain_sizes),
            "p_gt_50": sum(size > 50 for size in chain_sizes) / len(chain_sizes),
        }
        payload["generation1_r_pooled"] = (gen_children / gen_seeds) if gen_seeds else None
        payload["seeds"] = gen_seeds
    return payload


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for experiment, kwargs in (
        ("exp1", {}),
        ("sweep", {}),
        ("waves", {"waves": True}),
        ("exp2", {"early": True}),
        ("exp4", {"chains": True}),
        ("exp4cal", {"chains": True}),
        ("exp4pilot", {"chains": True}),
    ):
        folder = ROOT / experiment
        print(experiment, flush=True)
        cells = {}
        for cell in sorted(path for path in folder.iterdir() if path.is_dir()):
            cells[cell.name] = summarize_cell(cell, **kwargs)
        write_json(OUT / f"{experiment}.json", cells)

    print("exp3", flush=True)
    audit = {}
    targets = [
        "exp1/shedoff_shuf0_venueoff/rep01",
        "exp1/shed0.1_shuf0_venueoff/rep01",
        "exp1/shedoff_shuf0_venuecovid/rep01",
        "exp1/shed0.1_shuf0_venuecovid/rep01",
        "waves/baseline/rep01",
    ]
    for relative in targets:
        print(f"  {relative}", flush=True)
        trace = load_trace(ROOT / relative / "trace.tsv")
        audit[relative] = observation_models(trace)
        audit[relative]["venue_clusters"] = venue_clusters(trace)
    write_json(OUT / "exp3.json", audit)
    print("done", flush=True)


if __name__ == "__main__":
    main()
